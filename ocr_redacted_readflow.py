import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

from ocr_redacted_placeholders import EXTS, MODEL_ID, PROMPT_FREE_OCR, _binarize_dark, detect_redaction_boxes

LINE_COVERAGE_MIN = 0.50
FIRST_LINE_INDENT_SPACES = 5
BOX_LINE_CHARS_DEFAULT = 64


def clean_ds_output(raw) -> str:
    if raw is None:
        return ""
    s = str(raw)
    s = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", s, flags=re.DOTALL)
    s = re.sub(r"<\|det\|>.*?<\|/det\|>", "", s, flags=re.DOTALL)
    return s.strip()


def build_llm(model_id: str, max_model_len: int | None) -> LLM:
    kwargs = dict(
        model=model_id,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
    )
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    return LLM(**kwargs)


def ocr_with_model(
    llm: LLM,
    image_path: Path,
    prompt: str,
    max_tokens: int,
    temperature: float,
    ngram_size: int,
    window_size: int,
) -> str:
    img = Image.open(image_path).convert("RGB")
    model_input = [{
        "prompt": prompt,
        "multi_modal_data": {"image": img},
    }]
    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        extra_args=dict(
            ngram_size=ngram_size,
            window_size=window_size,
            whitelist_token_ids={128821, 128822},
        ),
        skip_special_tokens=False,
    )
    out = llm.generate(model_input, sampling)
    return clean_ds_output(out[0].outputs[0].text)


def _collect_images(in_path: Path) -> list[Path]:
    if in_path.is_file() and in_path.suffix.lower() in EXTS:
        return [in_path]
    if in_path.is_dir():
        return [p for p in sorted(in_path.iterdir()) if p.is_file() and p.suffix.lower() in EXTS]
    raise SystemExit(f"Input not found or not an image/folder: {in_path}")


def build_text_merged_mask(gray: np.ndarray, redaction_boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    dark = _binarize_dark(gray)
    mask = dark.copy()

    # Remove redaction regions from text-mask estimation.
    for x1, y1, x2, y2 in redaction_boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), color=0, thickness=-1)

    merged = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)), iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return merged


def build_structural_lines_mask(gray: np.ndarray) -> np.ndarray:
    # Matches the line-extraction logic used by redaction detection/debug_lines.
    dark = _binarize_dark(gray)
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
    return lines


def percentile_region(mask: np.ndarray, q: float = 2.5) -> tuple[int, int, int, int]:
    h, w = mask.shape
    ys, xs = np.where(mask > 0)

    if xs.size == 0 or ys.size == 0:
        return 0, 0, max(1, w - 1), max(1, h - 1)

    q = float(max(0.0, min(49.9, q)))
    left = int(np.floor(np.percentile(xs, q)))
    right = int(np.ceil(np.percentile(xs, 100.0 - q)))
    top = int(np.floor(np.percentile(ys, q)))
    bottom = int(np.ceil(np.percentile(ys, 100.0 - q)))

    left = max(0, min(w - 1, left))
    right = max(left + 1, min(w - 1, right))
    top = max(0, min(h - 1, top))
    bottom = max(top + 1, min(h - 1, bottom))

    return left, top, right, bottom


def normalize_ocr_lines(ocr_text: str) -> list[str]:
    lines = ocr_text.splitlines()
    while lines and lines[-1] == "":
        lines.pop()
    return lines if lines else [""]


def build_theoretical_line_boxes(
    ocr_lines: list[str],
    text_region: tuple[int, int, int, int],
) -> tuple[list[tuple[int, int, int, int]], float]:
    x1, y1, x2, y2 = text_region
    n = max(1, len(ocr_lines))
    total_h = max(1, (y2 - y1 + 1))
    avg_h = total_h / n

    line_boxes: list[tuple[int, int, int, int]] = []
    for i in range(n):
        ly1 = int(round(y1 + i * avg_h))
        ly2 = int(round(y1 + (i + 1) * avg_h)) - 1
        ly1 = max(y1, min(y2, ly1))
        ly2 = max(ly1, min(y2, ly2))
        line_boxes.append((x1, ly1, x2, ly2))

    return line_boxes, float(avg_h)


def overlap_ratio_y(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    _, ay1, _, ay2 = box_a
    _, by1, _, by2 = box_b
    inter = max(0, min(ay2, by2) - max(ay1, by1) + 1)
    den = max(1, by2 - by1 + 1)
    return inter / den


def map_x_to_char(x: int, left: int, right: int, capacity: int) -> int:
    width = max(1, right - left)
    t = (x - left) / width
    j = int(round(t * capacity))
    return max(0, min(capacity, j))


def _line_mapping_bounds(
    line_idx: int,
    left: int,
    right: int,
    max_chars: int,
    first_line_indent_spaces: int,
) -> tuple[int, int]:
    if line_idx != 0 or first_line_indent_spaces <= 0:
        return left, right

    char_px = max(1.0, (right - left) / max(1, max_chars))
    shift_px = int(round(first_line_indent_spaces * char_px))
    left_adj = min(right - 1, left + max(0, shift_px))
    return left_adj, right


def _snap_insert_to_word_start(line: str, pos: int) -> int:
    if not line:
        return 0

    pos = max(0, min(len(line), pos))
    if pos == 0 or pos == len(line):
        return pos
    if line[pos - 1].isspace() or line[pos].isspace():
        return pos

    left_space = line.rfind(" ", 0, pos)
    if left_space == -1:
        return 0
    return left_space + 1


def project_redactions_to_lines(
    redaction_boxes: list[tuple[int, int, int, int]],
    line_boxes: list[tuple[int, int, int, int]],
    ocr_lines: list[str],
    text_region: tuple[int, int, int, int],
    first_line_indent_spaces: int,
    box_line_chars: int,
) -> list[dict[str, object]]:
    left, _, right, _ = text_region
    line_count = len(line_boxes)
    max_chars = max([len(s) for s in ocr_lines] + [1])

    spans: list[dict[str, object]] = []

    for rid, red in enumerate(redaction_boxes, start=1):
        overlaps = [
            (idx, overlap_ratio_y(red, line_boxes[idx]))
            for idx in range(line_count)
        ]

        # A box spans a line only if it covers more than 50% of that line's height.
        hit_lines = [idx for idx, r in overlaps if r > LINE_COVERAGE_MIN]

        if not hit_lines:
            yc = (red[1] + red[3]) / 2.0
            nearest = min(
                range(line_count),
                key=lambda idx: abs((line_boxes[idx][1] + line_boxes[idx][3]) / 2.0 - yc),
            )
            hit_lines = [nearest]

        for line_idx in hit_lines:
            line_text = ocr_lines[line_idx]
            char_capacity = len(line_text) if line_text else max_chars
            eff_left, eff_right = _line_mapping_bounds(
                line_idx=line_idx,
                left=left,
                right=right,
                max_chars=max_chars,
                first_line_indent_spaces=first_line_indent_spaces,
            )

            clipped_x1 = max(eff_left, min(eff_right, red[0]))
            clipped_x2 = max(eff_left, min(eff_right, red[2]))
            touches_left = red[0] <= eff_left
            touches_right = red[2] >= eff_right

            if touches_left:
                c1 = 0
            else:
                c1 = map_x_to_char(clipped_x1, eff_left, eff_right, char_capacity)

            if touches_right:
                c2 = char_capacity
            else:
                c2 = map_x_to_char(clipped_x2, eff_left, eff_right, char_capacity)

            if c2 <= c1:
                c2 = min(char_capacity, c1 + 1)
            span_len = max(1, c2 - c1)

            line_width_px = max(1, eff_right - eff_left)
            clipped_span_px = max(0, clipped_x2 - clipped_x1)
            line_cover_ratio_x = max(0.0, clipped_span_px / line_width_px)

            # Stable cross-document sizing: full line corresponds to `box_line_chars`.
            # Token budget includes 'BOXn', so underscores are reduced by that width.
            box_label = f"BOX{rid}"
            total_budget = max(1, int(round(line_cover_ratio_x * box_line_chars)))
            underscore_len = max(1, total_budget - len(f"[{box_label}]"))
            full_line = touches_left and touches_right

            spans.append(
                {
                    "id": rid,
                    "line_1based": line_idx + 1,
                    "line_char_capacity": char_capacity,
                    "char_start": c1,
                    "char_end": c2,
                    "char_len": span_len,
                    "full_line": bool(full_line),
                    "bbox_xyxy": [int(v) for v in red],
                    "placeholder_token": f"[REDACTED_{rid}]",
                    "layout_token_underscore": f"[{box_label}_{'_' * underscore_len}]",
                    "layout_token_pixel": f"[{box_label}__{clipped_span_px}/{line_width_px}]",
                    "line_overlap_ratio": round(float(overlaps[line_idx][1]), 4),
                    "line_cover_ratio_x": round(float(line_cover_ratio_x), 4),
                    "clipped_span_px": int(clipped_span_px),
                    "line_width_effective_px": int(line_width_px),
                    "touches_left_border": bool(touches_left),
                    "touches_right_border": bool(touches_right),
                }
            )

    spans.sort(key=lambda s: (s["line_1based"], s["char_start"], s["id"]))
    return spans


def render_with_spans(
    ocr_lines: list[str],
    redaction_spans: list[dict[str, object]],
    token_key: str,
) -> str:
    by_line: dict[int, list[dict[str, object]]] = {}
    for span in redaction_spans:
        idx = int(span["line_1based"]) - 1
        by_line.setdefault(idx, []).append(span)

    rendered = list(ocr_lines)
    for line_idx, spans in by_line.items():
        line = rendered[line_idx]
        spans_sorted = sorted(spans, key=lambda s: int(s["char_start"]), reverse=True)
        for span in spans_sorted:
            if bool(span.get("touches_right_border")) and not bool(span.get("touches_left_border")):
                pos = len(line)
            elif bool(span.get("touches_left_border")) and not bool(span.get("touches_right_border")):
                pos = 0
            else:
                pos = _snap_insert_to_word_start(line, int(span["char_start"]))
            token = str(span[token_key])
            line = f"{line[:pos]} {token} {line[pos:]}"
        rendered[line_idx] = re.sub(r"\s{2,}", " ", line).strip()

    return "\n".join(rendered).rstrip()


def line_model_summary(
    ocr_lines: list[str],
    line_boxes: list[tuple[int, int, int, int]],
) -> list[dict[str, object]]:
    out = []
    for idx, (text, box) in enumerate(zip(ocr_lines, line_boxes), start=1):
        out.append(
            {
                "line_1based": idx,
                "text": text,
                "char_len": len(text),
                "line_box_xyxy": [int(v) for v in box],
            }
        )
    return out


def _line_color(i: int, total: int) -> tuple[int, int, int]:
    # Distinct-ish BGR colors for line guides.
    palette = [
        (255, 160, 0),
        (255, 80, 80),
        (0, 200, 255),
        (120, 255, 120),
        (220, 120, 255),
        (255, 220, 80),
        (120, 180, 255),
    ]
    if total <= 0:
        return palette[0]
    return palette[i % len(palette)]


def save_debug_line_model(
    gray: np.ndarray,
    merged: np.ndarray,
    merged_region_mask: np.ndarray,
    structural_lines_mask: np.ndarray,
    text_region: tuple[int, int, int, int],
    line_boxes: list[tuple[int, int, int, int]],
    redaction_boxes: list[tuple[int, int, int, int]],
    debug_dir: Path,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)

    x1, y1, x2, y2 = text_region

    vis_mask = cv2.cvtColor(merged_region_mask, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis_mask, (x1, y1), (x2, y2), (255, 0, 255), 1)
    cv2.imwrite(str(debug_dir / "debug_text_merged.png"), merged)
    cv2.imwrite(str(debug_dir / "debug_percentile_region.png"), vis_mask)
    cv2.imwrite(str(debug_dir / "debug_structural_lines_mask.png"), structural_lines_mask)
    cv2.imwrite(str(debug_dir / "debug_text_merged_no_structural_lines.png"), merged_region_mask)

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 1)
    line_contours, _ = cv2.findContours(structural_lines_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, line_contours, -1, (180, 180, 180), 1)

    for i, (lx1, ly1, lx2, ly2) in enumerate(line_boxes, start=1):
        color = _line_color(i - 1, len(line_boxes))
        cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), color, 1)
        cv2.putText(
            vis,
            f"L{i}",
            (lx1 + 3, min(ly2, ly1 + 11)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            color,
            1,
            cv2.LINE_AA,
        )

    for rid, (rx1, ry1, rx2, ry2) in enumerate(redaction_boxes, start=1):
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"R{rid}",
            (rx1 + 2, max(10, ry1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(debug_dir / "debug_line_model_overlay.png"), vis)


def process_image(
    llm: LLM,
    model_id: str,
    img_path: Path,
    out_dir: Path,
    prompt: str,
    max_tokens: int,
    temperature: float,
    ngram_size: int,
    window_size: int,
    debug: bool,
    print_raw_ocr: bool,
    print_projection: bool,
    percentile_q: float,
    first_line_indent_spaces: int,
    box_label_mode: str,
    box_line_chars: int,
) -> None:
    ocr_text = ocr_with_model(
        llm=llm,
        image_path=img_path,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        ngram_size=ngram_size,
        window_size=window_size,
    )

    if print_raw_ocr:
        print("\n=== RAW OCR (LINE PRESERVED) ===")
        print(ocr_text)

    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    debug_dir = (out_dir / "debug" / img_path.stem) if debug else None
    redaction_boxes = detect_redaction_boxes(gray, debug_dir=debug_dir)

    merged = build_text_merged_mask(gray, redaction_boxes)
    structural_lines = build_structural_lines_mask(gray)
    lines_exclusion = cv2.dilate(
        structural_lines,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    merged_region_mask = cv2.bitwise_and(merged, cv2.bitwise_not(lines_exclusion))
    if np.count_nonzero(merged_region_mask) == 0:
        merged_region_mask = merged

    text_region = percentile_region(merged_region_mask, q=percentile_q)
    ocr_lines = normalize_ocr_lines(ocr_text)
    line_boxes, avg_line_height = build_theoretical_line_boxes(ocr_lines, text_region)

    redaction_spans = project_redactions_to_lines(
        redaction_boxes=redaction_boxes,
        line_boxes=line_boxes,
        ocr_lines=ocr_lines,
        text_region=text_region,
        first_line_indent_spaces=first_line_indent_spaces,
        box_line_chars=box_line_chars,
    )

    readflow_text = render_with_spans(ocr_lines, redaction_spans, token_key="placeholder_token")
    if box_label_mode == "pixel":
        layout_token_key = "layout_token_pixel"
    else:
        layout_token_key = "layout_token_underscore"
    layout_projection_text = render_with_spans(ocr_lines, redaction_spans, token_key=layout_token_key)

    if print_projection:
        print("\n=== READFLOW OCR WITH REDACTIONS ===")
        print(readflow_text)
        print("\n=== LAYOUT PROJECTION ===")
        print(layout_projection_text)

    if debug_dir is not None:
        save_debug_line_model(
            gray=gray,
            merged=merged,
            merged_region_mask=merged_region_mask,
            structural_lines_mask=structural_lines,
            text_region=text_region,
            line_boxes=line_boxes,
            redaction_boxes=redaction_boxes,
            debug_dir=debug_dir,
        )

    out_raw = out_dir / f"{img_path.stem}.raw_ocr.txt"
    out_readflow = out_dir / f"{img_path.stem}.readflow_with_placeholders.txt"
    out_layout = out_dir / f"{img_path.stem}.layout_projection.txt"
    out_json = out_dir / f"{img_path.stem}.analysis.json"

    out_raw.write_text(ocr_text.rstrip() + "\n", encoding="utf-8")
    out_readflow.write_text(readflow_text.rstrip() + "\n", encoding="utf-8")
    out_layout.write_text(layout_projection_text.rstrip() + "\n", encoding="utf-8")

    x1, y1, x2, y2 = text_region
    out_json.write_text(
        json.dumps(
            {
                "image": str(img_path),
                "model": model_id,
                "prompt": prompt,
                "redaction_boxes_xyxy": [list(b) for b in redaction_boxes],
                "text_region_percentile_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "percentile_q": percentile_q,
                "region_masking": "percentile computed from merged_text minus structural_lines_mask",
                "merged_nonzero_px": int(np.count_nonzero(merged)),
                "merged_region_nonzero_px": int(np.count_nonzero(merged_region_mask)),
                "line_width_px": int(max(1, x2 - x1)),
                "line_count": len(ocr_lines),
                "avg_line_height_px": round(avg_line_height, 4),
                "lines": line_model_summary(ocr_lines, line_boxes),
                "redaction_line_spans": redaction_spans,
                "params": {
                    "line_coverage_min": LINE_COVERAGE_MIN,
                    "first_line_indent_spaces": first_line_indent_spaces,
                    "box_label_mode": box_label_mode,
                    "box_line_chars": box_line_chars,
                },
                "note": "line model uses percentile text region + uniform line boxes from OCR line count",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n[INFO] Detected {len(redaction_boxes)} redaction box(es)")
    print(f"[INFO] OCR line count: {len(ocr_lines)}")
    print(f"[INFO] Avg line height (px): {avg_line_height:.2f}")
    print(f"[INFO] Saved: {out_raw}")
    print(f"[INFO] Saved: {out_readflow}")
    print(f"[INFO] Saved: {out_layout}")
    print(f"[INFO] Saved: {out_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image path or folder")
    parser.add_argument("--out", default="outputs", help="Output folder")

    parser.add_argument("--model", default=MODEL_ID, help="Model id or local path")
    parser.add_argument("--prompt", default=PROMPT_FREE_OCR, help="OCR prompt")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Generation max tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--ngram_size", type=int, default=30, help="N-gram no-repeat size")
    parser.add_argument("--window_size", type=int, default=90, help="N-gram no-repeat window")
    parser.add_argument("--max_model_len", type=int, default=None, help="Override vLLM max_model_len")

    parser.add_argument("--percentile_q", type=float, default=0.5, help="Percentile for left/right/top/bottom bounds")
    parser.add_argument(
        "--first_line_indent_spaces",
        type=int,
        default=FIRST_LINE_INDENT_SPACES,
        help="Indentation slack in spaces applied to line 1 x-mapping",
    )
    parser.add_argument(
        "--box_label_mode",
        choices=["underscore", "pixel"],
        default="underscore",
        help="BOX label style for layout projection output",
    )
    parser.add_argument(
        "--box_line_chars",
        type=int,
        default=BOX_LINE_CHARS_DEFAULT,
        help="Normalized full-line character budget used to scale underscore BOX length",
    )

    parser.add_argument("--debug", action="store_true", help="Save debug images")
    parser.add_argument("--no_print_raw_ocr", action="store_true", help="Disable raw OCR console print")
    parser.add_argument("--no_print_projection", action="store_true", help="Disable projection console print")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = _collect_images(in_path)
    llm = build_llm(args.model, args.max_model_len)

    for img_path in images:
        print(f"\n[DOC] {img_path}")
        process_image(
            llm=llm,
            model_id=args.model,
            img_path=img_path,
            out_dir=out_dir,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            ngram_size=args.ngram_size,
            window_size=args.window_size,
            debug=args.debug,
            print_raw_ocr=not args.no_print_raw_ocr,
            print_projection=not args.no_print_projection,
            percentile_q=args.percentile_q,
            first_line_indent_spaces=args.first_line_indent_spaces,
            box_label_mode=args.box_label_mode,
            box_line_chars=args.box_line_chars,
        )


if __name__ == "__main__":
    main()
