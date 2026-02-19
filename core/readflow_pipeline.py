import argparse
import json
import re
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np
from PIL import Image

from core.ocr_shared import MODEL_ID, PROMPT_FREE_OCR, build_llm, collect_images, ocr_with_model, ocr_with_pil
from core.redaction_detection import binarize_dark, build_lines_mask, detect_redaction_boxes

LINE_COVERAGE_MIN = 0.50
FIRST_LINE_INDENT_SPACES = 5
BOX_LINE_CHARS_DEFAULT = 64
LINEBREAK_COLOR = (255, 80, 0)  # BGR (blue-ish)
BLANKLINE_COLOR = (0, 220, 255)  # BGR (yellow)
PARAGRAPH_OCR_PROMPT = (
    "<image>\nFree OCR. Preserve original line breaks and punctuation.\n"
    "Return only the transcribed text."
)
PARAGRAPH_OCR_PROMPT_STRICT = (
    "<image>\nTranscribe exactly the visible text.\n"
    "Output only the document text, no explanations or instructions."
)


def build_text_merged_mask(gray: np.ndarray, redaction_boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    dark = binarize_dark(gray)
    mask = dark.copy()

    # Remove redaction regions from text-mask estimation.
    for x1, y1, x2, y2 in redaction_boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), color=0, thickness=-1)

    merged = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)), iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return merged


def build_structural_lines_mask(
    gray_shape: tuple[int, int],
    redaction_boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    # Force structural lines to be derived only from redaction borders so
    # unrelated underlines/rules never contaminate downstream region modeling.
    h, w = gray_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in redaction_boxes:
        ix1 = max(0, min(w - 1, x1 + 1))
        iy1 = max(0, min(h - 1, y1 + 1))
        ix2 = max(0, min(w - 1, x2 - 1))
        iy2 = max(0, min(h - 1, y2 - 1))
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        cv2.rectangle(mask, (ix1, iy1), (ix2, iy2), color=255, thickness=1)
    return mask


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


def fit_lines_to_count(lines: list[str], target_count: int) -> list[str]:
    n = max(0, int(target_count))
    if n == 0:
        return []
    normalized = [_normalize_line_text(s) for s in lines]
    if len(normalized) >= n:
        return normalized[:n]
    return normalized + ([""] * (n - len(normalized)))


def _normalize_line_text(text: str) -> str:
    # Single-line OCR for crops should not carry accidental internal newlines.
    return re.sub(r"\s+", " ", text).strip()


def sanitize_paragraph_ocr_text(text: str) -> str:
    raw = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw.split("\n")]
    return "\n".join(lines).strip()


def _is_instruction_like_line(line: str) -> bool:
    s = line.strip().lower()
    if not s:
        return False
    if s.startswith("do not return"):
        return True
    if s.startswith("return only"):
        return True
    if s.startswith("output only"):
        return True
    if s.startswith("press the"):
        return True
    if s.startswith("line breaks are"):
        return True
    if s.startswith("preserve line breaks"):
        return True
    if s.startswith("the text is in"):
        return True
    if s.startswith("mifaqhama:"):
        return True
    return False


def _strip_instruction_like_lines(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    kept = [ln for ln in lines if ln and (not _is_instruction_like_line(ln))]
    return "\n".join(kept).strip()


def _looks_like_instructional_noise(text: str) -> bool:
    s = text.strip().lower()
    if not s:
        return False
    patterns = [
        "do not return",
        "return only",
        "preserve line breaks",
        "line breaks are",
        "press the",
        "the text is in",
        "output only",
    ]
    return any(p in s for p in patterns)


def _choose_ocr_candidate(
    candidates: list[str],
    max_plausible_chars: int,
) -> str:
    cleaned = []
    for c in candidates:
        if not str(c).strip():
            continue
        x = sanitize_paragraph_ocr_text(c)
        x = _strip_instruction_like_lines(x)
        if x:
            cleaned.append(x)
    if not cleaned:
        return ""

    # Best case: non-noisy and plausible length.
    for c in cleaned:
        if (not _looks_like_instructional_noise(c)) and (len(c) <= max_plausible_chars):
            return c

    # Next best: any plausible-length candidate.
    plausible = [c for c in cleaned if len(c) <= max_plausible_chars]
    if plausible:
        # Prefer longest plausible (usually captures most text on real paragraph crops).
        return max(plausible, key=len)

    # Last resort: shortest candidate to avoid massive chatter.
    return min(cleaned, key=len)


def build_text_pixels_mask(
    dark_mask_01: np.ndarray,
    lines_mask_02: np.ndarray,
    structural_lines_mask_12: np.ndarray,
) -> np.ndarray:
    # "Text pixels" = white in 01_dark_mask and not in 02_lines_mask nor 12_structural_lines_mask.
    excluded = cv2.bitwise_or(lines_mask_02, structural_lines_mask_12)
    return cv2.bitwise_and(dark_mask_01, cv2.bitwise_not(excluded))


def _close_small_runs(values: np.ndarray, target: bool, max_run: int) -> np.ndarray:
    out = values.copy()
    fill_value = not target
    n = len(values)
    i = 0
    while i < n:
        if bool(values[i]) != target:
            i += 1
            continue
        j = i
        while j + 1 < n and bool(values[j + 1]) == target:
            j += 1
        run_len = j - i + 1
        if run_len <= max_run:
            left = i > 0 and bool(values[i - 1]) != target
            right = j < (n - 1) and bool(values[j + 1]) != target
            if left and right:
                out[i : j + 1] = fill_value
        i = j + 1
    return out


def detect_line_and_gap_bands(
    text_pixels_mask: np.ndarray,
    text_region: tuple[int, int, int, int],
    row_ink_threshold: float = 0.05,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]], list[dict[str, object]]]:
    left, top, right, bottom = text_region
    roi = text_pixels_mask[top : bottom + 1, left : right + 1]
    if roi.size == 0:
        return [], [], []

    # Vertical smoothing prevents tiny single-pixel holes from splitting lines.
    # A row is treated as a linebreak row if it is >=95% black (ink ratio <=5%).
    roi_smooth = cv2.dilate(roi, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)
    row_ratio = np.mean(roi_smooth > 0, axis=1)
    non_empty = row_ratio > row_ink_threshold
    non_empty = _close_small_runs(non_empty, target=False, max_run=1)
    non_empty = _close_small_runs(non_empty, target=True, max_run=1)

    bands: list[dict[str, object]] = []
    n = len(non_empty)
    i = 0
    while i < n:
        state = bool(non_empty[i])
        j = i
        while j + 1 < n and bool(non_empty[j + 1]) == state:
            j += 1
        y1 = top + i
        y2 = top + j
        h = y2 - y1 + 1
        if state:
            seg = roi[i : j + 1, :]
            if h < 2 or int(np.count_nonzero(seg)) < 12:
                i = j + 1
                continue
            kind = "line"
        else:
            kind = "gap"
        bands.append(
            {
                "kind": kind,
                "y1": int(y1),
                "y2": int(y2),
                "box": (int(left), int(y1), int(right), int(y2)),
            }
        )
        i = j + 1

    # Keep only the core span from first line-band to last line-band.
    line_idxs = [idx for idx, b in enumerate(bands) if b["kind"] == "line"]
    if not line_idxs:
        return [], [], []
    lo, hi = min(line_idxs), max(line_idxs)
    bands = bands[lo : hi + 1]

    line_boxes = [b["box"] for b in bands if b["kind"] == "line"]
    gap_boxes = [b["box"] for b in bands if b["kind"] == "gap"]
    return line_boxes, gap_boxes, bands


def detect_indentation_boxes(
    text_pixels_mask: np.ndarray,
    line_boxes: list[tuple[int, int, int, int]],
    text_region: tuple[int, int, int, int],
    box_line_chars: int,
    redaction_boxes: list[tuple[int, int, int, int]],
    indent_chars: int = 5,
) -> tuple[list[bool], list[tuple[int, int, int, int]], int]:
    left, _, right, _ = text_region
    line_width = max(1, right - left + 1)
    probe_w = max(6, int(round((indent_chars / max(1, box_line_chars)) * line_width)))
    probe_x2 = min(right, left + probe_w)
    threshold_px = max(4, int(round(0.8 * probe_w)))

    flags: list[bool] = []
    boxes: list[tuple[int, int, int, int]] = []

    for lx1, ly1, lx2, ly2 in line_boxes:
        seg = text_pixels_mask[ly1 : ly2 + 1, left : right + 1]
        ys, xs = np.where(seg > 0)
        if xs.size == 0:
            flags.append(False)
            continue
        first_x = int(left + np.min(xs))
        indented = (first_x - left) >= threshold_px
        if indented:
            paragraph_box = (left, ly1, probe_x2, ly2)
            para_area = max(1, box_area(paragraph_box))
            inter_area_sum = 0
            for red in redaction_boxes:
                inter_area_sum += intersection_area(paragraph_box, red)
            overlap_ratio = min(1.0, inter_area_sum / para_area)
            if overlap_ratio > LINE_COVERAGE_MIN:
                # Fake paragraph trigger caused by redaction intersection.
                indented = False
            else:
                boxes.append(paragraph_box)

        flags.append(bool(indented))

    return flags, boxes, probe_w


def box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)


def intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 < ix1 or iy2 < iy1:
        return 0
    return (ix2 - ix1 + 1) * (iy2 - iy1 + 1)


def build_output_line_model(
    bands: list[dict[str, object]],
    text_lines: list[str],
    line_boxes: list[tuple[int, int, int, int]],
    line_indent_flags: list[bool],
    redaction_boxes: list[tuple[int, int, int, int]],
) -> tuple[list[str], list[tuple[int, int, int, int]], list[str], list[dict[str, object]]]:
    if not bands:
        return text_lines, line_boxes, ["text_line"] * len(text_lines), []

    heights = [max(1, y2 - y1 + 1) for (_, y1, _, y2) in line_boxes] if line_boxes else [10]
    avg_line_h = float(np.mean(heights))
    linebreak_threshold = max(1, int(round(0.50 * avg_line_h)))
    interline_gap_heights = []
    for b_idx, band in enumerate(bands):
        if str(band["kind"]) != "gap":
            continue
        prev_is_line = b_idx > 0 and str(bands[b_idx - 1]["kind"]) == "line"
        next_is_line = b_idx + 1 < len(bands) and str(bands[b_idx + 1]["kind"]) == "line"
        if not (prev_is_line and next_is_line):
            continue
        bx1, by1, bx2, by2 = [int(v) for v in band["box"]]
        h_gap = max(1, by2 - by1 + 1)
        if h_gap <= linebreak_threshold:
            interline_gap_heights.append(h_gap)
    avg_linebreak_h = float(np.mean(interline_gap_heights)) if interline_gap_heights else max(1.0, round(0.30 * avg_line_h, 4))
    avg_full_line_h = float(avg_line_h + avg_linebreak_h)

    out_lines: list[str] = []
    out_boxes: list[tuple[int, int, int, int]] = []
    out_kinds: list[str] = []
    gap_markers: list[dict[str, object]] = []
    line_idx = 0
    band_line_idx = 0

    def split_vertical_region_into_slots(
        x1: int,
        x2: int,
        y1: int,
        y2: int,
        n_slots: int,
    ) -> list[tuple[int, int, int, int]]:
        n = max(1, int(n_slots))
        total_h = max(1, y2 - y1 + 1)
        slots: list[tuple[int, int, int, int]] = []
        for i in range(n):
            sy1 = y1 + int(round((i * total_h) / n))
            sy2 = y1 + int(round(((i + 1) * total_h) / n)) - 1
            sy1 = max(y1, min(y2, sy1))
            sy2 = max(sy1, min(y2, sy2))
            slots.append((x1, sy1, x2, sy2))
        return slots

    for b_idx, band in enumerate(bands):
        kind = str(band["kind"])
        box = tuple(int(v) for v in band["box"])
        if kind == "line":
            text = text_lines[line_idx] if line_idx < len(text_lines) else ""
            out_lines.append(text)
            out_boxes.append(box)
            out_kinds.append("text_line")
            line_idx += 1
            band_line_idx += 1
            continue

        prev_is_line = b_idx > 0 and str(bands[b_idx - 1]["kind"]) == "line"
        next_is_line = b_idx + 1 < len(bands) and str(bands[b_idx + 1]["kind"]) == "line"
        if not (prev_is_line and next_is_line):
            continue

        y1, y2 = int(box[1]), int(box[3])
        gap_h = y2 - y1 + 1
        is_blankline_gap = gap_h > linebreak_threshold
        line_height_ratio = float(gap_h / max(1.0, avg_line_h))
        full_line_ratio = float(gap_h / max(1.0, avg_full_line_h))

        next_line_is_indented = False
        if band_line_idx < len(line_indent_flags):
            next_line_is_indented = bool(line_indent_flags[band_line_idx])

        # Use intersection over gap-box area (not just Y overlap) for paragraph suppression.
        gap_area = max(1, box_area(box))
        redaction_overlap_ratio = 0.0
        for red in redaction_boxes:
            redaction_overlap_ratio = max(redaction_overlap_ratio, intersection_area(box, red) / gap_area)

        # Paragraph marker follows indentation signal directly:
        # a red indentation box should always be preceded by PB.
        new_paragraph = bool(next_line_is_indented)

        if new_paragraph and is_blankline_gap:
            paragraphblank_height_px = int(max(1, round(avg_line_h)))
        elif new_paragraph:
            paragraphblank_height_px = int(gap_h)
        else:
            paragraphblank_height_px = 0
        paragraphblank_height_px = min(paragraphblank_height_px, gap_h)
        remaining_blank_height_px = max(0, gap_h - paragraphblank_height_px)
        blankline_slots = 0
        if is_blankline_gap and remaining_blank_height_px > 0:
            blankline_slots = int(max(0, round(remaining_blank_height_px / max(1.0, avg_full_line_h))))

        marker_kind = "blankline" if is_blankline_gap else "linebreak"
        gap_markers.append(
            {
                "box": [int(v) for v in box],
                "kind": marker_kind,
                "height_px": int(gap_h),
                "avg_text_line_height_px": round(float(avg_line_h), 4),
                "avg_linebreak_height_px": round(float(avg_linebreak_h), 4),
                "avg_full_line_height_px": round(float(avg_full_line_h), 4),
                "height_to_avg_line_ratio": round(line_height_ratio, 4),
                "height_to_avg_full_line_ratio": round(full_line_ratio, 4),
                "blankline_line_slots": int(blankline_slots),
                "paragraphblank_height_px": int(paragraphblank_height_px),
                "remaining_blank_height_px": int(remaining_blank_height_px),
                "next_line_indented": bool(next_line_is_indented),
                "redaction_overlap_ratio": round(float(redaction_overlap_ratio), 4),
                "new_paragraph": bool(new_paragraph),
            }
        )

        if is_blankline_gap:
            x1, _, x2, _ = box
            blank_y1 = y1
            blank_y2 = y2 - paragraphblank_height_px
            if blankline_slots > 0 and blank_y2 >= blank_y1:
                slot_boxes = split_vertical_region_into_slots(x1, x2, blank_y1, blank_y2, blankline_slots)
                for s_box in slot_boxes:
                    out_lines.append("")
                    out_boxes.append(s_box)
                    out_kinds.append("blankline")

            if new_paragraph and paragraphblank_height_px > 0:
                pb_y1 = y2 - paragraphblank_height_px + 1
                pb_y1 = max(y1, min(y2, pb_y1))
                pb_box = (x1, pb_y1, x2, y2)
                out_lines.append("")
                out_boxes.append(pb_box)
                out_kinds.append("paragraphblank")
        elif new_paragraph and paragraphblank_height_px > 0:
            x1, _, x2, _ = box
            pb_box = (x1, y1, x2, y2)
            out_lines.append("")
            out_boxes.append(pb_box)
            out_kinds.append("paragraphblank")

    return out_lines, out_boxes, out_kinds, gap_markers


def build_uniform_line_boxes_for_fallback(
    line_count: int,
    text_region: tuple[int, int, int, int],
) -> tuple[list[tuple[int, int, int, int]], float]:
    x1, y1, x2, y2 = text_region
    n = max(1, int(line_count))
    total_h = max(1, (y2 - y1 + 1))
    avg_h = total_h / n
    out: list[tuple[int, int, int, int]] = []
    for i in range(n):
        ly1 = int(round(y1 + i * avg_h))
        ly2 = int(round(y1 + (i + 1) * avg_h)) - 1
        ly1 = max(y1, min(y2, ly1))
        ly2 = max(ly1, min(y2, ly2))
        out.append((x1, ly1, x2, ly2))
    return out, float(avg_h)


def build_paragraph_blocks(
    line_boxes: list[tuple[int, int, int, int]],
    line_texts: list[str],
    line_kinds: list[str],
    text_region: tuple[int, int, int, int],
) -> list[dict[str, object]]:
    if not line_boxes:
        return []

    n = min(len(line_boxes), len(line_texts), len(line_kinds))
    if n <= 0:
        return []
    boxes = line_boxes[:n]
    texts = [_normalize_line_text(t) for t in line_texts[:n]]
    kinds = line_kinds[:n]

    left, _, right, _ = text_region
    start_idx = 0
    current_top_y = int(boxes[0][1])
    spans: list[tuple[int, int, int, int]] = []
    for i in range(n):
        if kinds[i] != "paragraphblank":
            continue
        mid_y = int(round((boxes[i][1] + boxes[i][3]) / 2.0))
        if i - 1 >= start_idx:
            spans.append((start_idx, i - 1, current_top_y, max(current_top_y, mid_y - 1)))
        start_idx = i + 1
        current_top_y = min(int(boxes[i][3]) + 1, int(mid_y + 1))
    if start_idx <= n - 1:
        spans.append((start_idx, n - 1, current_top_y, int(boxes[n - 1][3])))

    paragraphs: list[dict[str, object]] = []
    for p_idx, (s, e, py1_raw, py2_raw) in enumerate(spans, start=1):
        py1 = int(max(py1_raw, boxes[s][1]))
        py2 = int(min(py2_raw, boxes[e][3]))
        if py2 < py1:
            py1 = int(boxes[s][1])
            py2 = int(boxes[e][3])
        p_box = [int(left), int(py1), int(right), int(py2)]
        p_text = "\n".join(texts[s : e + 1]).rstrip()
        paragraphs.append(
            {
                "paragraph_1based": p_idx,
                "line_start_1based": int(s + 1),
                "line_end_1based": int(e + 1),
                "line_count": int(e - s + 1),
                "starts_with_paragraph_marker": bool(s > 0 and kinds[s - 1] == "paragraphblank"),
                "paragraph_box_xyxy": p_box,
                "text": p_text,
            }
        )
    return paragraphs


def save_paragraph_artifacts(
    gray: np.ndarray,
    out_dir: Path,
    stem: str,
    paragraphs: list[dict[str, object]],
) -> tuple[Path, Path, Path, list[dict[str, object]]]:
    paragraphs_dir = out_dir / f"{stem}.paragraph_crops"
    paragraphs_dir.mkdir(parents=True, exist_ok=True)
    for old_png in paragraphs_dir.glob("paragraph_*.png"):
        old_png.unlink(missing_ok=True)

    out_paragraph_txt = out_dir / f"{stem}.paragraph_ocr.txt"
    out_paragraph_json = out_dir / f"{stem}.paragraphs.json"

    h, w = gray.shape
    paragraph_text_blocks: list[str] = []
    saved_rows: list[dict[str, object]] = []
    for p in paragraphs:
        idx = int(p["paragraph_1based"])
        x1, y1, x2, y2 = [int(v) for v in p["paragraph_box_xyxy"]]
        box_w = max(1, x2 - x1 + 1)
        box_h = max(1, y2 - y1 + 1)
        # Keep a minimum context margin so tiny title/one-line crops still OCR reliably.
        pad_x = max(6, int(round(0.10 * box_w)))
        pad_y = max(8, int(round(0.10 * box_h)))
        xa = max(0, x1 - pad_x)
        ya = max(0, y1 - pad_y)
        xb = min(w - 1, x2 + pad_x)
        yb = min(h - 1, y2 + pad_y)
        crop = gray[ya : yb + 1, xa : xb + 1]

        crop_path = paragraphs_dir / f"paragraph_{idx:03d}.png"
        cv2.imwrite(str(crop_path), crop)

        text = str(p.get("text", "")).rstrip()
        paragraph_text_blocks.append(text)
        row = dict(p)
        row["expanded_crop_box_xyxy"] = [int(xa), int(ya), int(xb), int(yb)]
        row["crop_path"] = str(crop_path)
        saved_rows.append(row)

    out_paragraph_txt.write_text("\n\n".join(paragraph_text_blocks).rstrip() + "\n", encoding="utf-8")
    out_paragraph_json.write_text(json.dumps(saved_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return paragraphs_dir, out_paragraph_txt, out_paragraph_json, saved_rows


def ocr_paragraph_blocks(
    llm: Any,
    paragraphs: list[dict[str, object]],
    max_tokens: int,
    temperature: float,
    ngram_size: int,
    window_size: int,
) -> tuple[list[dict[str, object]], str]:
    out_rows: list[dict[str, object]] = []
    paragraph_texts: list[str] = []

    for p in paragraphs:
        row = dict(p)
        crop_path_raw = row.get("crop_path")
        crop_img: Image.Image | None = None
        width_px = 0
        if isinstance(crop_path_raw, str) and crop_path_raw:
            crop_path = Path(crop_path_raw)
            if crop_path.exists():
                crop_img = Image.open(crop_path).convert("RGB")
                width_px = crop_img.size[0]
        if crop_img is None:
            row["ocr_text"] = ""
            out_rows.append(row)
            paragraph_texts.append("")
            continue

        line_count = max(1, int(row.get("line_count", 1)))
        est_chars_per_line = max(18, int(round(max(1, width_px) / 7.0)))
        est_chars = max(24, line_count * est_chars_per_line)
        max_toks_para = max(96, min(max_tokens, int(round(est_chars * 1.8)) + 32))
        max_plausible_chars = max(120, int(round(est_chars * 2.5)))

        raw_1 = ocr_with_pil(
            llm=llm,
            image=crop_img,
            prompt=PARAGRAPH_OCR_PROMPT_STRICT,
            max_tokens=max_toks_para,
            temperature=temperature,
            ngram_size=ngram_size,
            window_size=window_size,
        )
        # Binarized retry for weak scans/noisy crops.
        roi_np = np.array(crop_img.convert("L"))
        roi_bin = cv2.adaptiveThreshold(roi_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
        roi_bin_img = Image.fromarray(roi_bin).convert("RGB")
        raw_2 = ocr_with_pil(
            llm=llm,
            image=roi_bin_img,
            prompt=PARAGRAPH_OCR_PROMPT,
            max_tokens=max_toks_para,
            temperature=temperature,
            ngram_size=ngram_size,
            window_size=window_size,
        )
        row["ocr_text"] = _choose_ocr_candidate([raw_1, raw_2], max_plausible_chars=max_plausible_chars)
        out_rows.append(row)
        paragraph_texts.append(row["ocr_text"])

    joined = "\n\n".join([t.rstrip() for t in paragraph_texts]).rstrip()
    return out_rows, joined


def save_paragraph_ocr_files(
    paragraphs_dir: Path,
    paragraph_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    out_rows: list[dict[str, object]] = []
    for row in paragraph_rows:
        r = dict(row)
        idx = int(r.get("paragraph_1based", 0))
        out_path = paragraphs_dir / f"paragraph_{idx:03d}.ocr.txt"
        text = str(r.get("ocr_text", "")).rstrip()
        out_path.write_text(text + "\n", encoding="utf-8")
        r["ocr_text_path"] = str(out_path)
        out_rows.append(r)
    return out_rows


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
    line_kinds: list[str] | None = None,
) -> list[dict[str, object]]:
    if line_kinds is None:
        line_kinds = ["text_line"] * len(ocr_lines)
    out = []
    total = min(len(ocr_lines), len(line_boxes), len(line_kinds))
    for idx in range(total):
        text = ocr_lines[idx]
        box = line_boxes[idx]
        line_kind = line_kinds[idx]
        serial_num = idx + 1
        if line_kind == "paragraphblank":
            serial_label = f"PB{serial_num}"
        elif line_kind == "blankline":
            serial_label = f"BL{serial_num}"
        else:
            serial_label = f"L{serial_num}"
        out.append(
            {
                "line_1based": serial_num,
                "serial_label": serial_label,
                "text": text,
                "char_len": len(text),
                "line_kind": line_kind,
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
    merged_region_mask: np.ndarray,
    structural_lines_mask: np.ndarray,
    text_pixels_mask: np.ndarray,
    text_region: tuple[int, int, int, int],
    line_boxes: list[tuple[int, int, int, int]],
    linebreak_boxes: list[tuple[int, int, int, int]],
    blankline_boxes: list[tuple[int, int, int, int]],
    redaction_boxes: list[tuple[int, int, int, int]],
    indentation_boxes: list[tuple[int, int, int, int]],
    serialized_line_boxes: list[tuple[int, int, int, int]],
    serialized_line_kinds: list[str],
    debug_dir: Path,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)

    x1, y1, x2, y2 = text_region

    vis_mask = cv2.cvtColor(merged_region_mask, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis_mask, (x1, y1), (x2, y2), (255, 0, 255), 1)
    cv2.imwrite(str(debug_dir / "12_structural_lines_mask.png"), structural_lines_mask)
    cv2.imwrite(str(debug_dir / "14_percentile_region.png"), vis_mask)
    cv2.imwrite(str(debug_dir / "17_text_pixels_mask.png"), text_pixels_mask)

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 1)

    for gx1, gy1, gx2, gy2 in linebreak_boxes:
        cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), LINEBREAK_COLOR, 1)
    for bx1, by1, bx2, by2 in blankline_boxes:
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), BLANKLINE_COLOR, 1)

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

    # Serialized sequence labels (L#/BL#/PB#) in running order.
    total_serial = min(len(serialized_line_boxes), len(serialized_line_kinds))
    for s_idx in range(total_serial):
        sx1, sy1, sx2, sy2 = serialized_line_boxes[s_idx]
        sk = serialized_line_kinds[s_idx]
        serial_num = s_idx + 1
        if sk == "paragraphblank":
            scolor = (0, 0, 255)
            slabel = f"PB{serial_num}"
        elif sk == "blankline":
            scolor = BLANKLINE_COLOR
            slabel = f"BL{serial_num}"
        else:
            scolor = (255, 255, 255)
            slabel = f"L{serial_num}"
        cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), scolor, 1)
        cv2.putText(
            vis,
            slabel,
            (sx1 + 2, min(sy2, sy1 + 11)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            scolor,
            1,
            cv2.LINE_AA,
        )

    for ix1, iy1, ix2, iy2 in indentation_boxes:
        cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), (0, 0, 255), 2)

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

    cv2.imwrite(str(debug_dir / "15_line_model_overlay.png"), vis)

    linebreak_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color_layer = linebreak_vis.copy()
    for gx1, gy1, gx2, gy2 in linebreak_boxes:
        cv2.rectangle(color_layer, (gx1, gy1), (gx2, gy2), LINEBREAK_COLOR, thickness=-1)
    for bx1, by1, bx2, by2 in blankline_boxes:
        cv2.rectangle(color_layer, (bx1, by1), (bx2, by2), BLANKLINE_COLOR, thickness=-1)
    linebreak_vis = cv2.addWeighted(color_layer, 0.28, linebreak_vis, 0.72, 0.0)
    cv2.rectangle(linebreak_vis, (x1, y1), (x2, y2), (255, 0, 255), 1)
    cv2.imwrite(str(debug_dir / "16_linebreak_overlay.png"), linebreak_vis)

    serialized_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(serialized_vis, (x1, y1), (x2, y2), (255, 0, 255), 1)
    for gx1, gy1, gx2, gy2 in linebreak_boxes:
        cv2.rectangle(serialized_vis, (gx1, gy1), (gx2, gy2), LINEBREAK_COLOR, 1)
    for rid, (rx1, ry1, rx2, ry2) in enumerate(redaction_boxes, start=1):
        cv2.rectangle(serialized_vis, (rx1, ry1), (rx2, ry2), (0, 255, 0), 1)
        cv2.putText(
            serialized_vis,
            f"R{rid}",
            (rx1 + 2, max(10, ry1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.33,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    total = min(len(serialized_line_boxes), len(serialized_line_kinds))
    serial_color = (255, 220, 0)  # same cyan used in 19_paragraph_blocks_overlay
    for idx in range(total):
        bx1, by1, bx2, by2 = serialized_line_boxes[idx]
        kind = serialized_line_kinds[idx]
        if kind == "paragraphblank":
            label = f"PB{idx + 1}"
        elif kind == "blankline":
            label = f"BL{idx + 1}"
        else:
            label = f"L{idx + 1}"
        cv2.rectangle(serialized_vis, (bx1, by1), (bx2, by2), serial_color, 1)
        cv2.putText(
            serialized_vis,
            label,
            (bx1 + 2, min(by2, by1 + 11)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            serial_color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(debug_dir / "18_serialized_blanklines_overlay.png"), serialized_vis)


def save_debug_paragraph_blocks(
    gray: np.ndarray,
    text_region: tuple[int, int, int, int],
    redaction_boxes: list[tuple[int, int, int, int]],
    indentation_boxes: list[tuple[int, int, int, int]],
    paragraphs: list[dict[str, object]],
    debug_dir: Path,
) -> None:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    x1, y1, x2, y2 = text_region
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 1)

    for rid, (rx1, ry1, rx2, ry2) in enumerate(redaction_boxes, start=1):
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 0), 1)
        cv2.putText(
            vis,
            f"R{rid}",
            (rx1 + 2, max(10, ry1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    for ix1, iy1, ix2, iy2 in indentation_boxes:
        cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), (0, 0, 255), 2)

    for p in paragraphs:
        pid = int(p.get("paragraph_1based", 0))
        px1, py1, px2, py2 = [int(v) for v in p.get("paragraph_box_xyxy", [0, 0, 0, 0])]
        cv2.rectangle(vis, (px1, py1), (px2, py2), (255, 220, 0), 2)
        cv2.putText(
            vis,
            f"P{pid}",
            (px1 + 3, min(py2, py1 + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 220, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(debug_dir / "19_paragraph_blocks_overlay.png"), vis)


def process_image(
    llm: Any,
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
    full_ocr_text = ""
    raw_ocr_source = "line_model"
    line_indent_flags: list[bool] = []

    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    # Housekeeping for deprecated line-split artifacts.
    stale_line_txt = out_dir / f"{img_path.stem}.line_ocr.txt"
    stale_line_jsonl = out_dir / f"{img_path.stem}.line_ocr.jsonl"
    stale_line_crops = out_dir / f"{img_path.stem}.line_crops"
    stale_line_txt.unlink(missing_ok=True)
    stale_line_jsonl.unlink(missing_ok=True)
    if stale_line_crops.exists() and stale_line_crops.is_dir():
        shutil.rmtree(stale_line_crops, ignore_errors=True)

    debug_dir = (out_dir / "debug" / img_path.stem) if debug else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        for old_png in debug_dir.glob("*.png"):
            old_png.unlink(missing_ok=True)
    redaction_boxes = detect_redaction_boxes(gray, debug_dir=debug_dir)

    dark_mask = binarize_dark(gray)
    lines_mask_02 = build_lines_mask(gray)
    structural_lines = build_structural_lines_mask(gray.shape, redaction_boxes)
    text_pixels_mask = build_text_pixels_mask(dark_mask, lines_mask_02, structural_lines)

    merged = build_text_merged_mask(gray, redaction_boxes)
    lines_exclusion = cv2.dilate(
        structural_lines,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    merged_region_mask = cv2.bitwise_and(merged, cv2.bitwise_not(lines_exclusion))
    if np.count_nonzero(merged_region_mask) == 0:
        merged_region_mask = merged

    if np.count_nonzero(text_pixels_mask) > 0:
        text_region = percentile_region(text_pixels_mask, q=percentile_q)
    else:
        text_region = percentile_region(merged_region_mask, q=percentile_q)

    line_band_boxes, gap_band_boxes, bands = detect_line_and_gap_bands(
        text_pixels_mask=text_pixels_mask,
        text_region=text_region,
    )

    ocr_line_source = "row_scan_bands"
    linebreak_boxes_for_overlay: list[tuple[int, int, int, int]] = []
    blankline_boxes_for_overlay: list[tuple[int, int, int, int]] = []
    indentation_boxes: list[tuple[int, int, int, int]] = []
    gap_markers: list[dict[str, object]] = []
    line_kinds: list[str] = []
    raw_line_bands_count = len(line_band_boxes)
    raw_gap_band_count = len(gap_band_boxes)
    raw_linebreak_band_count = 0
    raw_blankline_band_count = 0
    raw_new_paragraph_count = 0

    full_ocr_text = ocr_with_model(
        llm=llm,
        image_path=img_path,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        ngram_size=ngram_size,
        window_size=window_size,
    )
    ocr_base_lines = [_normalize_line_text(s) for s in normalize_ocr_lines(full_ocr_text)]

    if line_band_boxes:
        line_texts = fit_lines_to_count(ocr_base_lines, len(line_band_boxes))

        line_indent_flags, indentation_boxes, _ = detect_indentation_boxes(
            text_pixels_mask=text_pixels_mask,
            line_boxes=line_band_boxes,
            text_region=text_region,
            box_line_chars=box_line_chars,
            redaction_boxes=redaction_boxes,
            indent_chars=5,
        )
        ocr_lines, line_boxes, line_kinds, gap_markers = build_output_line_model(
            bands=bands,
            text_lines=line_texts,
            line_boxes=line_band_boxes,
            line_indent_flags=line_indent_flags,
            redaction_boxes=redaction_boxes,
        )
        linebreak_boxes_for_overlay = [tuple(m["box"]) for m in gap_markers if str(m["kind"]) == "linebreak"]
        blankline_boxes_for_overlay = [tuple(m["box"]) for m in gap_markers if str(m["kind"]) == "blankline"]
        raw_linebreak_band_count = len(linebreak_boxes_for_overlay)
        raw_blankline_band_count = len(blankline_boxes_for_overlay)
        raw_new_paragraph_count = sum(1 for m in gap_markers if bool(m.get("new_paragraph")))
    else:
        ocr_lines = list(ocr_base_lines)
        if not ocr_lines:
            ocr_lines = [""]
        line_boxes, _ = build_uniform_line_boxes_for_fallback(len(ocr_lines), text_region)
        line_kinds = ["text_line"] * len(line_boxes)
        line_indent_flags = [False] * len(line_boxes)
        ocr_line_source = "fallback_full_ocr_uniform_lines"

    if not ocr_lines:
        ocr_lines = [""]
        line_boxes, _ = build_uniform_line_boxes_for_fallback(1, text_region)
        line_kinds = ["text_line"]
        line_indent_flags = [False]

    if not line_kinds:
        line_kinds = ["text_line"] * len(line_boxes)
    if len(line_kinds) < len(line_boxes):
        line_kinds.extend(["text_line"] * (len(line_boxes) - len(line_kinds)))
    elif len(line_kinds) > len(line_boxes):
        line_kinds = line_kinds[: len(line_boxes)]

    paragraph_blocks = build_paragraph_blocks(
        line_boxes=line_boxes,
        line_texts=ocr_lines,
        line_kinds=line_kinds,
        text_region=text_region,
    )
    paragraph_crops_dir, out_paragraph_ocr, out_paragraph_json, paragraph_rows = save_paragraph_artifacts(
        gray=gray,
        out_dir=out_dir,
        stem=img_path.stem,
        paragraphs=paragraph_blocks,
    )

    paragraph_rows, paragraph_joined_ocr = ocr_paragraph_blocks(
        llm=llm,
        paragraphs=paragraph_rows,
        max_tokens=max_tokens,
        temperature=temperature,
        ngram_size=ngram_size,
        window_size=window_size,
    )
    paragraph_rows = save_paragraph_ocr_files(paragraph_crops_dir, paragraph_rows)
    # Canonical raw OCR follows the structured line model. Paragraph OCR is experimental,
    # saved separately, and not used as canonical text because it can be unstable.
    ocr_text = "\n".join(ocr_lines).rstrip()

    # Persist paragraph OCR outputs based on paragraph-level OCR pass.
    paragraph_ocr_text_blocks = [str(p.get("ocr_text", "")).rstrip() for p in paragraph_rows]
    out_paragraph_ocr.write_text("\n\n".join(paragraph_ocr_text_blocks).rstrip() + "\n", encoding="utf-8")
    out_paragraph_json.write_text(json.dumps(paragraph_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    avg_line_height = float(np.mean([max(1, y2 - y1 + 1) for (_, y1, _, y2) in line_boxes]))

    if print_raw_ocr:
        print("\n=== RAW OCR (LINE PRESERVED) ===")
        print(ocr_text)

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

    overlay_line_boxes = line_band_boxes if line_band_boxes else line_boxes
    if debug_dir is not None:
        save_debug_line_model(
            gray=gray,
            merged_region_mask=merged_region_mask,
            structural_lines_mask=structural_lines,
            text_pixels_mask=text_pixels_mask,
            text_region=text_region,
            line_boxes=overlay_line_boxes,
            linebreak_boxes=linebreak_boxes_for_overlay,
            blankline_boxes=blankline_boxes_for_overlay,
            redaction_boxes=redaction_boxes,
            indentation_boxes=indentation_boxes,
            serialized_line_boxes=line_boxes,
            serialized_line_kinds=line_kinds,
            debug_dir=debug_dir,
        )
        save_debug_paragraph_blocks(
            gray=gray,
            text_region=text_region,
            redaction_boxes=redaction_boxes,
            indentation_boxes=indentation_boxes,
            paragraphs=paragraph_rows,
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
                "paragraph_crops_dir": str(paragraph_crops_dir),
                "paragraph_ocr_txt_path": str(out_paragraph_ocr),
                "paragraph_json_path": str(out_paragraph_json),
                "paragraph_count": len(paragraph_rows),
                "paragraph_blocks": paragraph_rows,
                "avg_line_height_px": round(avg_line_height, 4),
                "ocr_line_source": ocr_line_source,
                "raw_ocr_source": raw_ocr_source,
                "raw_ocr_full_image_line_count": len(normalize_ocr_lines(full_ocr_text)) if full_ocr_text else 0,
                "row_scan_line_band_count": raw_line_bands_count,
                "row_scan_gap_band_count": raw_gap_band_count,
                "row_scan_linebreak_band_count": raw_linebreak_band_count,
                "row_scan_blankline_band_count": raw_blankline_band_count,
                "row_scan_new_paragraph_count": raw_new_paragraph_count,
                "indentation_band_count": len(indentation_boxes),
                "gap_markers": gap_markers,
                "line_kind_counts": {
                    "text_line": int(sum(1 for k in line_kinds if k == "text_line")),
                    "blankline": int(sum(1 for k in line_kinds if k == "blankline")),
                    "paragraphblank": int(sum(1 for k in line_kinds if k == "paragraphblank")),
                },
                "lines": line_model_summary(ocr_lines, line_boxes, line_kinds=line_kinds),
                "redaction_line_spans": redaction_spans,
                "params": {
                    "line_coverage_min": LINE_COVERAGE_MIN,
                    "first_line_indent_spaces": first_line_indent_spaces,
                    "box_label_mode": box_label_mode,
                    "box_line_chars": box_line_chars,
                },
                "note": "line model uses row-scan bands over text_pixels (01-02-12), maps full-image OCR lines onto bands, serializes linebreak/blankline/paragraphblank slots, and splits paragraphs at PB midpoints",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n[INFO] Detected {len(redaction_boxes)} redaction box(es)")
    print(f"[INFO] OCR line count: {len(ocr_lines)}")
    print(f"[INFO] Avg line height (px): {avg_line_height:.2f}")
    print(f"[INFO] Saved: {out_paragraph_ocr}")
    print(f"[INFO] Saved: {out_paragraph_json}")
    print(f"[INFO] Saved paragraph crops dir: {paragraph_crops_dir}")
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
    parser.add_argument(
        "--ocr_backend",
        choices=["auto", "vllm", "hf"],
        default="auto",
        help="OCR runtime backend. auto => OCR-2 uses HF, OCR v1 uses vLLM",
    )
    parser.add_argument(
        "--hf_attn_implementation",
        default="eager",
        help="HF attention implementation for OCR-2 (use eager for compatibility)",
    )
    parser.add_argument(
        "--hf_dtype",
        default="bfloat16",
        help="HF dtype for OCR-2 backend (bfloat16|float16|float32)",
    )

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

    images = collect_images(in_path)
    llm = build_llm(
        args.model,
        args.max_model_len,
        ocr_backend=args.ocr_backend,
        hf_attn_implementation=args.hf_attn_implementation,
        hf_dtype=args.hf_dtype,
    )

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
