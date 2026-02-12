import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

MODEL_ID = "deepseek-ai/DeepSeek-OCR"
PROMPT_FREE_OCR = "<image>\nFree OCR. Preserve original line breaks."
EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def clean_ds_output(raw) -> str:
    if raw is None:
        return ""
    s = str(raw)
    s = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", s, flags=re.DOTALL)
    s = re.sub(r"<\|det\|>.*?<\|/det\|>", "", s, flags=re.DOTALL)
    return s.strip()


def ocr_deepseek_vllm(llm: LLM, img_path: Path, max_tokens: int) -> str:
    img = Image.open(img_path).convert("RGB")
    model_input = [{
        "prompt": PROMPT_FREE_OCR,
        "multi_modal_data": {"image": img},
    }]
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},
        ),
        skip_special_tokens=False,
    )
    out = llm.generate(model_input, sampling)
    return clean_ds_output(out[0].outputs[0].text)


def _edge_density(edges: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad: int = 3) -> float:
    xa, ya, xb, yb = x1 + pad, y1 + pad, x2 - pad, y2 - pad
    if xb <= xa or yb <= ya:
        return 1.0
    roi = edges[ya:yb, xa:xb]
    if roi.size == 0:
        return 1.0
    return float(np.mean(roi > 0))


def _binarize_dark(gray: np.ndarray) -> np.ndarray:
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hard_inv = ((gray < 170).astype(np.uint8) * 255)
    return cv2.bitwise_or(otsu_inv, hard_inv)


def _inner_region(x1: int, y1: int, x2: int, y2: int, pad: int) -> tuple[int, int, int, int]:
    return x1 + pad, y1 + pad, x2 - pad, y2 - pad


def _interior_stats(gray: np.ndarray, dark: np.ndarray, box: tuple[int, int, int, int], pad: int = 3) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    xa, ya, xb, yb = _inner_region(x1, y1, x2, y2, pad)
    if xb <= xa or yb <= ya:
        return 0.0, 1.0
    interior_gray = gray[ya:yb, xa:xb]
    interior_dark = dark[ya:yb, xa:xb]
    if interior_gray.size == 0:
        return 0.0, 1.0
    return float(interior_gray.mean()), float(np.mean(interior_dark > 0))


def _side_dark_frac(dark: np.ndarray, box: tuple[int, int, int, int], thickness: int = 2) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    t = max(1, thickness)
    top = dark[y1:y1 + t, x1:x2]
    bot = dark[y2 - t:y2, x1:x2]
    lef = dark[y1:y2, x1:x1 + t]
    rig = dark[y1:y2, x2 - t:x2]
    if min(top.size, bot.size, lef.size, rig.size) == 0:
        return 0.0, 0.0, 0.0, 0.0

    def frac(mask: np.ndarray) -> float:
        return float(np.mean(mask > 0))

    return frac(top), frac(bot), frac(lef), frac(rig)


def _split_stacked_boxes(h_lines: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    roi = h_lines[y1:y2, x1:x2]
    if roi.size == 0:
        return [box]

    row_strength = (roi > 0).mean(axis=1)
    seams = np.where(row_strength > 0.60)[0]
    seams = seams[(seams > 4) & (seams < (y2 - y1 - 5))]
    if seams.size == 0:
        return [box]

    groups = []
    start = int(seams[0])
    prev = int(seams[0])
    for s in seams[1:]:
        s = int(s)
        if s == prev + 1:
            prev = s
            continue
        groups.append((start, prev))
        start = s
        prev = s
    groups.append((start, prev))

    cuts = [y1]
    for a, b in groups:
        cuts.append(y1 + int(round((a + b) / 2)))
    cuts.append(y2)

    parts: list[tuple[int, int, int, int]] = []
    for ya, yb in zip(cuts[:-1], cuts[1:]):
        if yb - ya >= 8:
            parts.append((x1, ya, x2, yb))

    return parts if parts else [box]


def _is_redaction_like(gray: np.ndarray, dark: np.ndarray, edges: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w < 30 or h < 10:
        return False
    if w * h < 300:
        return False

    interior_mean, interior_dark = _interior_stats(gray, dark, box, pad=3)
    if interior_mean < 220:
        return False
    if interior_dark > 0.12:
        return False

    if _edge_density(edges, x1, y1, x2, y2, pad=3) > 0.08:
        return False

    topf, botf, leff, rigf = _side_dark_frac(dark, box, thickness=2)
    if min(topf, botf, leff, rigf) < 0.08:
        return False

    return True


def _detect_from_lines(gray: np.ndarray, dark: np.ndarray) -> tuple[list[tuple[int, int, int, int]], np.ndarray, np.ndarray]:
    h, w = gray.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(24, w // 10), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 4)))

    h_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, v_kernel, iterations=1)

    lines = cv2.bitwise_or(h_lines, v_lines)
    lines = cv2.morphologyEx(
        lines,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        candidate = (x, y, x + bw, y + bh)
        for sub in _split_stacked_boxes(h_lines, candidate):
            if _is_redaction_like(gray, dark, edges, sub):
                boxes.append(sub)

    return boxes, lines, edges


def _detect_from_contours(
    gray: np.ndarray,
    dark: np.ndarray,
    edges: np.ndarray,
) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    mask = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) < 4 or len(approx) > 8:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 30 or bh < 10:
            continue

        rect_area = float(bw * bh)
        if rect_area <= 0:
            continue

        fill_ratio = float(cv2.contourArea(cnt) / rect_area)
        if fill_ratio < 0.55:
            continue

        box = (x, y, x + bw, y + bh)
        if _is_redaction_like(gray, dark, edges, box):
            boxes.append(box)

    return boxes, mask


def _area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0

    union = _area(a) + _area(b) - inter
    if union <= 0:
        return 0.0

    return inter / union


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int], margin: int = 2) -> bool:
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    return (
        ox1 <= ix1 + margin
        and oy1 <= iy1 + margin
        and ox2 >= ix2 - margin
        and oy2 >= iy2 - margin
    )


def _dedupe_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    clean = [tuple(int(v) for v in box) for box in boxes if _area(box) > 0]
    clean.sort(key=_area)

    kept: list[tuple[int, int, int, int]] = []
    for box in clean:
        skip = False
        for prev in kept:
            if _iou(box, prev) >= 0.65:
                skip = True
                break
            if _contains(box, prev, margin=2):
                skip = True
                break
        if not skip:
            kept.append(box)

    kept.sort(key=lambda b: (b[1], b[0]))
    return kept


def detect_redaction_boxes(gray: np.ndarray, debug_dir: Path | None = None) -> list[tuple[int, int, int, int]]:
    dark = _binarize_dark(gray)
    line_boxes, lines_mask, edges = _detect_from_lines(gray, dark)
    contour_boxes, contour_mask = _detect_from_contours(gray, dark, edges)
    boxes = _dedupe_boxes(line_boxes + contour_boxes)

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "debug_dark.png"), dark)
        cv2.imwrite(str(debug_dir / "debug_lines.png"), lines_mask)
        cv2.imwrite(str(debug_dir / "debug_contours.png"), contour_mask)
        cv2.imwrite(str(debug_dir / "debug_edges.png"), edges)

        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / "debug_boxes.png"), vis)

    return boxes


def estimate_text_block(gray: np.ndarray) -> tuple[int, int, int, int]:
    h, w = gray.shape
    content = (gray < 245).astype(np.uint8) * 255
    coords = cv2.findNonZero(content)
    if coords is None:
        return 0, 0, w, h
    x, y, bw, bh = cv2.boundingRect(coords)
    return x, y, x + bw, y + bh


def _snap_to_word_boundary(line: str, pos: int) -> int:
    if not line:
        return 0
    pos = max(0, min(len(line), pos))

    left = line.rfind(" ", 0, pos + 1)
    right = line.find(" ", pos)

    candidates = []
    if left != -1:
        candidates.append(left + 1)
    if right != -1:
        candidates.append(right)

    if not candidates:
        return pos

    return min(candidates, key=lambda x: abs(x - pos))


def insert_placeholders(
    ocr_text: str,
    boxes: list[tuple[int, int, int, int]],
    text_block: tuple[int, int, int, int],
) -> str:
    lines = ocr_text.splitlines()
    if not lines or not boxes:
        return ocr_text.rstrip()

    nonempty = [idx for idx, line in enumerate(lines) if line.strip()]
    if not nonempty:
        return ocr_text.rstrip()

    x_left, y_top, x_right, y_bottom = text_block
    count = len(nonempty)

    def line_for_y(y: float) -> int:
        t = (y - y_top) / max((y_bottom - y_top), 1)
        j = int(round(t * (count - 1)))
        return max(0, min(count - 1, j))

    def pos_for_x(line: str, x: float) -> int:
        if not line:
            return 0
        t = (x - x_left) / max((x_right - x_left), 1)
        raw = int(round(t * len(line)))
        raw = max(0, min(len(line), raw))
        return _snap_to_word_boundary(line, raw)

    inserts_by_line: dict[int, list[tuple[float, str]]] = {}
    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        token = f"[REDACTED_{idx}]"
        y_anchor = (y1 + y2) / 2.0
        x_anchor = (x1 + x2) / 2.0
        nonempty_idx = line_for_y(y_anchor)
        line_idx = nonempty[nonempty_idx]
        inserts_by_line.setdefault(line_idx, []).append((x_anchor, token))

    for line_idx, placements in inserts_by_line.items():
        line = lines[line_idx]
        placements.sort(key=lambda item: item[0], reverse=True)
        for x_anchor, token in placements:
            pos = pos_for_x(line, x_anchor)
            line = f"{line[:pos]} {token} {line[pos:]}"
        line = re.sub(r"\s{2,}", " ", line).strip()
        lines[line_idx] = line

    return "\n".join(lines).rstrip()


def build_placeholders(boxes: list[tuple[int, int, int, int]]) -> list[dict[str, object]]:
    items = []
    for idx, box in enumerate(boxes, start=1):
        items.append({"id": idx, "token": f"[REDACTED_{idx}]", "bbox_xyxy": list(box)})
    return items


def process_image(
    llm: LLM,
    img_path: Path,
    out_dir: Path,
    max_tokens: int,
    debug: bool,
) -> tuple[str, list[tuple[int, int, int, int]]]:
    ocr_text = ocr_deepseek_vllm(llm, img_path, max_tokens=max_tokens)

    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        boxes: list[tuple[int, int, int, int]] = []
        text_block = (0, 0, 1, 1)
    else:
        debug_dir = (out_dir / "debug" / img_path.stem) if debug else None
        boxes = detect_redaction_boxes(gray, debug_dir=debug_dir)
        text_block = estimate_text_block(gray)

    with_placeholders = insert_placeholders(ocr_text, boxes, text_block)

    out_txt = out_dir / f"{img_path.stem}.redacted_with_placeholders.txt"
    out_txt.write_text(with_placeholders, encoding="utf-8")

    out_json = out_dir / f"{img_path.stem}.redaction_boxes.json"
    out_json.write_text(
        json.dumps(
            {
                "image": str(img_path),
                "boxes_xyxy": [list(box) for box in boxes],
                "placeholders": build_placeholders(boxes),
                "note": "boxes are detected from redacted outlines; OCR includes only visible text",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return with_placeholders, boxes


def _collect_images(in_path: Path) -> list[Path]:
    if in_path.is_file() and in_path.suffix.lower() in EXTS:
        return [in_path]
    if in_path.is_dir():
        return [p for p in sorted(in_path.iterdir()) if p.is_file() and p.suffix.lower() in EXTS]
    raise SystemExit(f"Input not found or not an image/folder: {in_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image path OR folder")
    parser.add_argument("--out", default="outputs", help="Output folder")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = _collect_images(in_path)

    llm = LLM(
        model=MODEL_ID,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
    )

    for img_path in images:
        print(f"\n[DOC] {img_path}")
        text, boxes = process_image(
            llm=llm,
            img_path=img_path,
            out_dir=out_dir,
            max_tokens=args.max_tokens,
            debug=args.debug,
        )
        print(f"[INFO] Detected {len(boxes)} redaction box(es).")
        print(f"[INFO] Saved: {out_dir / (img_path.stem + '.redacted_with_placeholders.txt')}")
        if not text:
            print("[WARN] OCR output was empty after cleaning.")


if __name__ == "__main__":
    main()
