from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ocr_shared import MODEL_ID, PROMPT_FREE_OCR, build_llm, collect_images
from core.redaction_detection import binarize_dark, detect_redaction_boxes
from core.readflow_pipeline import (
    BOX_LINE_CHARS_DEFAULT,
    FIRST_LINE_INDENT_SPACES,
    build_structural_lines_mask,
    percentile_region,
    process_image as legacy_process_image,
)

LINE_COVERAGE_MIN = 0.50


def _box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 < ix1 or iy2 < iy1:
        return 0
    return (ix2 - ix1 + 1) * (iy2 - iy1 + 1)


def _build_text_pixels_mask(
    gray: np.ndarray,
    redaction_boxes: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    dark = binarize_dark(gray)
    mask = dark.copy()
    for x1, y1, x2, y2 in redaction_boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), color=0, thickness=-1)
    structural_lines = build_structural_lines_mask(gray)
    text_pixels = cv2.bitwise_and(mask, cv2.bitwise_not(structural_lines))
    return text_pixels, structural_lines


def _detect_line_boxes(
    text_pixels_mask: np.ndarray,
    text_region: tuple[int, int, int, int],
    row_ink_threshold: float = 0.05,
) -> list[tuple[int, int, int, int]]:
    left, top, right, bottom = text_region
    roi = text_pixels_mask[top : bottom + 1, left : right + 1]
    if roi.size == 0:
        return []

    roi_smooth = cv2.dilate(roi, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)
    row_ratio = np.mean(roi_smooth > 0, axis=1)
    non_empty = row_ratio > row_ink_threshold

    line_boxes: list[tuple[int, int, int, int]] = []
    i = 0
    n = len(non_empty)
    while i < n:
        if not bool(non_empty[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(non_empty[j + 1]):
            j += 1
        y1 = top + i
        y2 = top + j
        seg = roi[i : j + 1, :]
        if (y2 - y1 + 1) >= 2 and int(np.count_nonzero(seg)) >= 12:
            line_boxes.append((left, y1, right, y2))
        i = j + 1
    return line_boxes


def _detect_indentation_flags(
    text_pixels_mask: np.ndarray,
    line_boxes: list[tuple[int, int, int, int]],
    text_region: tuple[int, int, int, int],
    redaction_boxes: list[tuple[int, int, int, int]],
    box_line_chars: int,
    indent_chars: int = 5,
) -> list[bool]:
    left, _, right, _ = text_region
    line_width = max(1, right - left + 1)
    probe_w = max(6, int(round((indent_chars / max(1, box_line_chars)) * line_width)))
    probe_x2 = min(right, left + probe_w)
    threshold_px = max(4, int(round(0.8 * probe_w)))

    flags: list[bool] = []
    for _, ly1, _, ly2 in line_boxes:
        seg = text_pixels_mask[ly1 : ly2 + 1, left : right + 1]
        ys, xs = np.where(seg > 0)
        if xs.size == 0:
            flags.append(False)
            continue
        first_x = int(left + np.min(xs))
        indented = (first_x - left) >= threshold_px
        if indented:
            probe_box = (left, ly1, probe_x2, ly2)
            probe_area = max(1, _box_area(probe_box))
            overlap_ratio = 0.0
            for red in redaction_boxes:
                overlap_ratio = max(overlap_ratio, _intersection_area(probe_box, red) / probe_area)
            if overlap_ratio > LINE_COVERAGE_MIN:
                indented = False
        flags.append(bool(indented))
    return flags


def _build_paragraph_boxes(
    line_boxes: list[tuple[int, int, int, int]],
    indentation_flags: list[bool],
    text_region: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> tuple[list[tuple[int, int, int, int]], list[int]]:
    if not line_boxes:
        return [], []

    h, w = image_shape
    starts = [0] + [i for i, f in enumerate(indentation_flags) if f and i > 0]
    starts = sorted(set(starts))

    left, _, right, _ = text_region
    paragraph_boxes: list[tuple[int, int, int, int]] = []
    pb_line_indices: list[int] = []
    for idx, s in enumerate(starts):
        e = (starts[idx + 1] - 1) if (idx + 1 < len(starts)) else (len(line_boxes) - 1)
        if e < s:
            continue
        y1 = int(line_boxes[s][1])
        y2 = int(line_boxes[e][3])
        bw = max(1, right - left + 1)
        bh = max(1, y2 - y1 + 1)
        pad_x = max(1, int(round(0.10 * bw)))
        pad_y = max(1, int(round(0.10 * bh)))
        xa = max(0, left - pad_x)
        ya = max(0, y1 - pad_y)
        xb = min(w - 1, right + pad_x)
        yb = min(h - 1, y2 + pad_y)
        paragraph_boxes.append((xa, ya, xb, yb))
        pb_line_indices.append(s + 1)
    return paragraph_boxes, pb_line_indices


def _save_structure_debug_images(
    gray: np.ndarray,
    text_pixels: np.ndarray,
    structural_lines: np.ndarray,
    text_region: tuple[int, int, int, int],
    line_boxes: list[tuple[int, int, int, int]],
    indentation_flags: list[bool],
    paragraph_boxes: list[tuple[int, int, int, int]],
    pb_line_indices: list[int],
    debug_dir: Path,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)

    x1, y1, x2, y2 = text_region

    cv2.imwrite(str(debug_dir / "11_text_pixels_mask.png"), text_pixels)
    cv2.imwrite(str(debug_dir / "12_structural_lines_mask.png"), structural_lines)

    vis_lines = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis_lines, (x1, y1), (x2, y2), (255, 0, 255), 1)
    for i, (lx1, ly1, lx2, ly2) in enumerate(line_boxes, start=1):
        cv2.rectangle(vis_lines, (lx1, ly1), (lx2, ly2), (255, 180, 0), 1)
        cv2.putText(
            vis_lines,
            f"L{i}",
            (lx1 + 3, min(ly2, ly1 + 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 180, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(debug_dir / "13_line_boxes_overlay.png"), vis_lines)

    vis_pb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis_pb, (x1, y1), (x2, y2), (255, 0, 255), 1)
    for i, (lx1, ly1, lx2, ly2) in enumerate(line_boxes, start=1):
        if i - 1 < len(indentation_flags) and indentation_flags[i - 1]:
            cv2.rectangle(vis_pb, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)
            cv2.putText(
                vis_pb,
                f"PB{i}",
                (lx1 + 3, min(ly2, ly1 + 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.rectangle(vis_pb, (lx1, ly1), (lx2, ly2), (255, 180, 0), 1)
    cv2.imwrite(str(debug_dir / "14_indent_pb_overlay.png"), vis_pb)

    vis_para = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis_para, (x1, y1), (x2, y2), (255, 0, 255), 1)
    for idx, (px1, py1, px2, py2) in enumerate(paragraph_boxes, start=1):
        cv2.rectangle(vis_para, (px1, py1), (px2, py2), (0, 0, 255), 2)
        label = f"P{idx}"
        if idx - 1 < len(pb_line_indices):
            label = f"P{idx}/PB{pb_line_indices[idx - 1]}"
        cv2.putText(
            vis_para,
            label,
            (px1 + 3, max(12, py1 + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(debug_dir / "15_paragraph_boxes_overlay.png"), vis_para)

    vis_split = vis_para.copy()
    for i in range(1, len(paragraph_boxes)):
        prev_bottom = paragraph_boxes[i - 1][3]
        next_top = paragraph_boxes[i][1]
        split_y = int(round((prev_bottom + next_top) / 2.0))
        split_y = max(0, min(gray.shape[0] - 1, split_y))
        cv2.line(vis_split, (0, split_y), (gray.shape[1] - 1, split_y), (0, 255, 255), 1)
        cv2.putText(
            vis_split,
            f"SPLIT{i}",
            (6, max(12, split_y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(debug_dir / "16_paragraph_split_boundaries.png"), vis_split)


def _shift_tokens(text: str, offset: int) -> str:
    if offset <= 0:
        return text

    out = re.sub(
        r"\[REDACTED_(\d+)\]",
        lambda m: f"[REDACTED_{int(m.group(1)) + offset}]",
        text,
    )
    out = re.sub(
        r"\[BOX(\d+)([^\]]*)\]",
        lambda m: f"[BOX{int(m.group(1)) + offset}{m.group(2)}]",
        out,
    )
    return out


def _count_redactions(segment_analysis_path: Path, segment_readflow_text: str) -> int:
    if segment_analysis_path.exists():
        try:
            data = json.loads(segment_analysis_path.read_text(encoding="utf-8"))
            boxes = data.get("redaction_boxes_xyxy", [])
            if isinstance(boxes, list):
                return len(boxes)
        except Exception:
            pass
    ids = [int(x) for x in re.findall(r"\[REDACTED_(\d+)\]", segment_readflow_text)]
    return max(ids) if ids else 0


def process_image_multiparagraph(
    llm,
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
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    redaction_boxes = detect_redaction_boxes(gray, debug_dir=None)
    text_pixels, structural_lines = _build_text_pixels_mask(gray, redaction_boxes)

    if np.count_nonzero(text_pixels) > 0:
        text_region = percentile_region(text_pixels, q=percentile_q)
    else:
        text_region = percentile_region(binarize_dark(gray), q=percentile_q)

    line_boxes = _detect_line_boxes(text_pixels, text_region=text_region)
    if not line_boxes:
        line_boxes = [text_region]
    indent_flags = _detect_indentation_flags(
        text_pixels_mask=text_pixels,
        line_boxes=line_boxes,
        text_region=text_region,
        redaction_boxes=redaction_boxes,
        box_line_chars=box_line_chars,
    )
    paragraph_boxes, pb_line_indices = _build_paragraph_boxes(
        line_boxes=line_boxes,
        indentation_flags=indent_flags,
        text_region=text_region,
        image_shape=gray.shape,
    )
    if not paragraph_boxes:
        paragraph_boxes = [text_region]
        pb_line_indices = [1]

    if debug:
        _save_structure_debug_images(
            gray=gray,
            text_pixels=text_pixels,
            structural_lines=structural_lines,
            text_region=text_region,
            line_boxes=line_boxes,
            indentation_flags=indent_flags,
            paragraph_boxes=paragraph_boxes,
            pb_line_indices=pb_line_indices,
            debug_dir=out_dir / "debug" / img_path.stem,
        )

    split_dir = out_dir / "paragraph_splits" / img_path.stem
    split_dir.mkdir(parents=True, exist_ok=True)
    for old in split_dir.glob("paragraph_*.png"):
        old.unlink(missing_ok=True)

    paragraph_paths: list[Path] = []
    for i, (x1, y1, x2, y2) in enumerate(paragraph_boxes, start=1):
        crop = gray[y1 : y2 + 1, x1 : x2 + 1]
        p_path = split_dir / f"paragraph_{i:03d}.png"
        cv2.imwrite(str(p_path), crop)
        paragraph_paths.append(p_path)

    segment_out_dir = out_dir / "paragraph_runs" / img_path.stem
    segment_out_dir.mkdir(parents=True, exist_ok=True)

    segment_raw_texts: list[str] = []
    segment_readflow_texts: list[str] = []
    segment_layout_texts: list[str] = []
    segment_rows: list[dict[str, object]] = []

    redaction_offset = 0
    for i, p_path in enumerate(paragraph_paths, start=1):
        seg_stem = f"{img_path.stem}__p{i:03d}"
        seg_image = segment_out_dir / f"{seg_stem}.png"
        cv2.imwrite(str(seg_image), cv2.imread(str(p_path), cv2.IMREAD_GRAYSCALE))

        legacy_process_image(
            llm=llm,
            model_id=model_id,
            img_path=seg_image,
            out_dir=segment_out_dir,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            ngram_size=ngram_size,
            window_size=window_size,
            debug=debug,
            print_raw_ocr=False,
            print_projection=False,
            percentile_q=percentile_q,
            first_line_indent_spaces=first_line_indent_spaces,
            box_label_mode=box_label_mode,
            box_line_chars=box_line_chars,
        )

        seg_raw_path = segment_out_dir / f"{seg_stem}.raw_ocr.txt"
        seg_readflow_path = segment_out_dir / f"{seg_stem}.readflow_with_placeholders.txt"
        seg_layout_path = segment_out_dir / f"{seg_stem}.layout_projection.txt"
        seg_analysis_path = segment_out_dir / f"{seg_stem}.analysis.json"

        seg_raw = seg_raw_path.read_text(encoding="utf-8").strip() if seg_raw_path.exists() else ""
        seg_readflow = seg_readflow_path.read_text(encoding="utf-8").strip() if seg_readflow_path.exists() else ""
        seg_layout = seg_layout_path.read_text(encoding="utf-8").strip() if seg_layout_path.exists() else ""

        seg_readflow_shifted = _shift_tokens(seg_readflow, redaction_offset)
        seg_layout_shifted = _shift_tokens(seg_layout, redaction_offset)
        redaction_count = _count_redactions(seg_analysis_path, seg_readflow)
        redaction_offset += redaction_count

        segment_raw_texts.append(seg_raw)
        segment_readflow_texts.append(seg_readflow_shifted)
        segment_layout_texts.append(seg_layout_shifted)
        segment_rows.append(
            {
                "paragraph_1based": i,
                "pb_line_1based": int(pb_line_indices[i - 1]) if (i - 1) < len(pb_line_indices) else None,
                "paragraph_box_xyxy": [int(v) for v in paragraph_boxes[i - 1]],
                "segment_image": str(seg_image),
                "segment_raw_path": str(seg_raw_path),
                "segment_readflow_path": str(seg_readflow_path),
                "segment_layout_path": str(seg_layout_path),
                "redaction_offset_start": int(redaction_offset - redaction_count),
                "redaction_count": int(redaction_count),
            }
        )

    out_raw = out_dir / f"{img_path.stem}.raw_ocr.txt"
    out_readflow = out_dir / f"{img_path.stem}.readflow_with_placeholders.txt"
    out_layout = out_dir / f"{img_path.stem}.layout_projection.txt"
    out_json = out_dir / f"{img_path.stem}.analysis.json"

    final_raw = "\n\n".join([t for t in segment_raw_texts]).rstrip()
    final_readflow = "\n\n".join([t for t in segment_readflow_texts]).rstrip()
    final_layout = "\n\n".join([t for t in segment_layout_texts]).rstrip()

    out_raw.write_text(final_raw + "\n", encoding="utf-8")
    out_readflow.write_text(final_readflow + "\n", encoding="utf-8")
    out_layout.write_text(final_layout + "\n", encoding="utf-8")

    out_json.write_text(
        json.dumps(
            {
                "image": str(img_path),
                "model": model_id,
                "mode": "multiparagraph_via_legacy",
                "paragraph_split_count": len(paragraph_boxes),
                "paragraph_boxes_xyxy": [[int(v) for v in b] for b in paragraph_boxes],
                "pb_line_indices_1based": [int(v) for v in pb_line_indices],
                "split_dir": str(split_dir),
                "segment_out_dir": str(segment_out_dir),
                "segments": segment_rows,
                "note": "Page split into paragraph crops; legacy single-paragraph OCR pipeline run per crop; outputs concatenated with one blank line between paragraphs.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if print_raw_ocr:
        print("\n=== RAW OCR (PARAGRAPH CONCAT) ===")
        print(final_raw)
    if print_projection:
        print("\n=== READFLOW OCR WITH REDACTIONS (PARAGRAPH CONCAT) ===")
        print(final_readflow)
        print("\n=== LAYOUT PROJECTION (PARAGRAPH CONCAT) ===")
        print(final_layout)

    print(f"\n[INFO] Paragraph splits: {len(paragraph_boxes)}")
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

    parser.add_argument("--percentile_q", type=float, default=0.5, help="Percentile for text region bounds")
    parser.add_argument(
        "--first_line_indent_spaces",
        type=int,
        default=FIRST_LINE_INDENT_SPACES,
        help="Indentation slack in spaces for legacy line projection",
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
        help="Normalized full-line character budget for BOX underscore sizing",
    )
    parser.add_argument("--debug", action="store_true", help="Save debug images from legacy per-paragraph runs")
    parser.add_argument("--no_print_raw_ocr", action="store_true", help="Disable raw OCR console print")
    parser.add_argument("--no_print_projection", action="store_true", help="Disable projection console print")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(in_path)
    llm = build_llm(args.model, args.max_model_len)

    for img_path in images:
        print(f"\n[DOC] {img_path}")
        process_image_multiparagraph(
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
