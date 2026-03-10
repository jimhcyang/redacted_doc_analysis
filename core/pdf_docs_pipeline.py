from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from core.diagnostics_compare import (
    annotate_text_with_redaction_mask,
    build_chunks_from_mask,
    clean_ocr_text,
    tokenize_words,
    align_monotonic,
)
from core.ocr_shared import (
    MODEL_ID,
    PROMPT_FREE_OCR,
    build_llm,
    ocr_with_model,
)

NOISE_HINTS = [
    "preserve line breaks",
    "do not return punctuation",
    "do not remove the line breaks",
    "line breaks are not preserved",
    "press the enter key",
]
MOJIBAKE_HINTS = ["â€¢", "ï¿", "Â", "â€”", "â€“", "Ã"]


def _clean_filename_component(s: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "x"


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


@dataclass
class PDFPair:
    pair_key: str
    unredacted_numeric_id: str
    redacted_doc_id: str
    row: list[str]
    row_index_1based: int
    redacted_pdf: Path
    unredacted_pdf: Path


@dataclass
class PairCollectionStats:
    rows_total: int = 0
    rows_missing_required_fields: int = 0
    rows_invalid_unredacted_id_format: int = 0
    rows_missing_redacted_pdf: int = 0
    rows_missing_unredacted_pdf: int = 0
    rows_duplicate_unredacted_id: int = 0
    rows_duplicate_redacted_id: int = 0
    rows_valid: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "rows_total": int(self.rows_total),
            "rows_missing_required_fields": int(self.rows_missing_required_fields),
            "rows_invalid_unredacted_id_format": int(self.rows_invalid_unredacted_id_format),
            "rows_missing_redacted_pdf": int(self.rows_missing_redacted_pdf),
            "rows_missing_unredacted_pdf": int(self.rows_missing_unredacted_pdf),
            "rows_duplicate_unredacted_id": int(self.rows_duplicate_unredacted_id),
            "rows_duplicate_redacted_id": int(self.rows_duplicate_redacted_id),
            "rows_valid": int(self.rows_valid),
        }


def _normalize_csv_row(row: list[str], width: int = 11) -> list[str]:
    out = list(row)
    if len(out) < width:
        out.extend([""] * (width - len(out)))
    return out


def _parse_unredacted_id(raw: str) -> int | None:
    s = str(raw).strip()
    if not s or (not s.isdigit()):
        return None
    # User rule: keep digit IDs that are at most 8 digits; left-pad to 8 for filename.
    if len(s) > 8:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _canonical_unredacted_pdf_path(unredacted_id_int: int, unredacted_dir: Path) -> Path:
    return unredacted_dir / f"cib_{unredacted_id_int:08d}.pdf"


def _progress_iter(
    iterable: Iterable[Any],
    *,
    total: int | None,
    desc: str,
    leave: bool = False,
) -> Iterable[Any]:
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=leave,
        dynamic_ncols=True,
        mininterval=0.2,
    )


def collect_pdf_pairs_with_stats(
    csv_path: Path,
    redacted_dir: Path,
    unredacted_dir: Path,
    max_pairs: int | None = None,
) -> tuple[list[PDFPair], PairCollectionStats]:
    pairs: list[PDFPair] = []
    stats = PairCollectionStats()
    seen_unred_ids: set[str] = set()
    seen_red_ids: set[str] = set()

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, raw_row in enumerate(reader, start=1):
            stats.rows_total += 1
            row = _normalize_csv_row(raw_row)
            unred_id_raw = str(row[1]).strip()
            red_doc_id = str(row[5]).strip()

            if not unred_id_raw or not red_doc_id:
                stats.rows_missing_required_fields += 1
                continue

            unred_id_int = _parse_unredacted_id(unred_id_raw)
            if unred_id_int is None:
                stats.rows_invalid_unredacted_id_format += 1
                continue

            unred_id_norm = str(unred_id_int)
            if unred_id_norm in seen_unred_ids:
                stats.rows_duplicate_unredacted_id += 1
                continue
            if red_doc_id in seen_red_ids:
                stats.rows_duplicate_redacted_id += 1
                continue

            red_pdf = redacted_dir / f"{red_doc_id}.pdf"
            if not red_pdf.exists():
                stats.rows_missing_redacted_pdf += 1
                continue

            # Strict mapping rule: only cib_<8_digit_number>.pdf is valid.
            unred_pdf = _canonical_unredacted_pdf_path(unred_id_int, unredacted_dir)
            if not unred_pdf.exists():
                stats.rows_missing_unredacted_pdf += 1
                continue

            seen_unred_ids.add(unred_id_norm)
            seen_red_ids.add(red_doc_id)

            pair_key = _clean_filename_component(f"{unred_id_norm}__{red_doc_id}")
            pairs.append(
                PDFPair(
                    pair_key=pair_key,
                    unredacted_numeric_id=unred_id_norm,
                    redacted_doc_id=red_doc_id,
                    row=row,
                    row_index_1based=i,
                    redacted_pdf=red_pdf,
                    unredacted_pdf=unred_pdf,
                )
            )
            stats.rows_valid += 1

            if max_pairs is not None and len(pairs) >= max_pairs:
                break

    return pairs, stats


def collect_pdf_pairs(
    csv_path: Path,
    redacted_dir: Path,
    unredacted_dir: Path,
    max_pairs: int | None = None,
) -> list[PDFPair]:
    pairs, _ = collect_pdf_pairs_with_stats(
        csv_path=csv_path,
        redacted_dir=redacted_dir,
        unredacted_dir=unredacted_dir,
        max_pairs=max_pairs,
    )
    return pairs


def render_pdf_to_images(
    pdf_path: Path,
    out_dir: Path,
    prefix: str,
    dpi: int,
    max_pages: int | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer pypdfium2 because it has self-contained wheels and avoids poppler.
    try:
        import pypdfium2 as pdfium  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PDF rendering backend missing. Install pypdfium2 in your runtime: pip install pypdfium2"
        ) from e

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        page_count = len(doc)
        if max_pages is not None:
            page_count = min(page_count, max(0, int(max_pages)))

        images: list[Path] = []
        scale = max(0.1, float(dpi) / 72.0)
        for page_index in range(page_count):
            page = doc.get_page(page_index)
            try:
                bmp = page.render(scale=scale)
                pil = bmp.to_pil()
                out_path = out_dir / f"{prefix}_p{page_index + 1:04d}.png"
                pil.save(out_path)
                images.append(out_path)
            finally:
                page.close()
        return images
    finally:
        doc.close()


def extract_pdf_text_by_page(pdf_path: Path, max_pages: int | None = None) -> list[str]:
    try:
        import pypdfium2 as pdfium  # type: ignore
    except Exception:
        return []

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        page_count = len(doc)
        if max_pages is not None:
            page_count = min(page_count, max(0, int(max_pages)))

        out: list[str] = []
        for page_index in range(page_count):
            page = doc.get_page(page_index)
            try:
                text_page = page.get_textpage()
                try:
                    text = text_page.get_text_range()
                finally:
                    text_page.close()
            except Exception:
                text = ""
            finally:
                page.close()
            raw_text = str(text or "").strip()
            out.append(_repair_common_mojibake(raw_text))
        return out
    finally:
        doc.close()


def _repair_common_mojibake(text: str) -> str:
    s = str(text or "")
    if not s:
        return s
    if not any(h in s for h in MOJIBAKE_HINTS):
        return s
    try:
        fixed = s.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return s
    if _count_mojibake_hints(fixed) < _count_mojibake_hints(s):
        return fixed
    return s


def _count_mojibake_hints(text: str) -> int:
    s = str(text or "")
    return sum(s.count(h) for h in MOJIBAKE_HINTS)


def preprocess_page_for_ocr(
    src_path: Path,
    dst_path: Path,
    *,
    trim_border_ratio: float = 0.01,
    content_pad_ratio: float = 0.015,
    upscale: float = 1.35,
    autocontrast_cutoff: int = 1,
) -> dict[str, object]:
    img = Image.open(src_path).convert("L")
    w, h = img.size
    if h <= 1 or w <= 1:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(dst_path)
        return {
            "crop_xyxy": [0, 0, max(0, w - 1), max(0, h - 1)],
            "preprocessed_size_wh": [int(w), int(h)],
            "upscale": float(upscale),
            "autocontrast_cutoff": int(autocontrast_cutoff),
            "used_trim": False,
        }

    trim_border_ratio = max(0.0, min(0.15, float(trim_border_ratio)))
    content_pad_ratio = max(0.0, min(0.10, float(content_pad_ratio)))
    upscale = max(1.0, min(3.0, float(upscale)))

    tx = int(round(w * trim_border_ratio))
    ty = int(round(h * trim_border_ratio))
    x0 = min(max(0, tx), max(0, w - 1))
    y0 = min(max(0, ty), max(0, h - 1))
    x1 = max(x0 + 1, w - tx)
    y1 = max(y0 + 1, h - ty)
    core = img.crop((x0, y0, x1, y1))
    mask = core.point(lambda p: 255 if p < 245 else 0, mode="L")

    used_trim = False
    cx1, cy1, cx2, cy2 = 0, 0, w - 1, h - 1
    pix = mask.load()
    mw, mh = mask.size
    row_counts = [0] * mh
    col_counts = [0] * mw
    for yy in range(mh):
        c = 0
        for xx in range(mw):
            if pix[xx, yy] > 0:
                c += 1
                col_counts[xx] += 1
        row_counts[yy] = c

    row_thresh = max(1, int(round(0.002 * mw)))
    col_thresh = max(1, int(round(0.002 * mh)))
    rows = [i for i, c in enumerate(row_counts) if c >= row_thresh]
    cols = [i for i, c in enumerate(col_counts) if c >= col_thresh]

    if rows and cols:
        bx1_local = min(cols)
        bx2_local = max(cols) + 1
        by1_local = min(rows)
        by2_local = max(rows) + 1
        bx1 = x0 + int(bx1_local)
        by1 = y0 + int(by1_local)
        bx2 = x0 + int(max(bx1_local, bx2_local - 1))
        by2 = y0 + int(max(by1_local, by2_local - 1))
        pad_x = int(round((bx2 - bx1 + 1) * content_pad_ratio))
        pad_y = int(round((by2 - by1 + 1) * content_pad_ratio))
        cx1 = max(0, bx1 - pad_x)
        cy1 = max(0, by1 - pad_y)
        cx2 = min(w - 1, bx2 + pad_x)
        cy2 = min(h - 1, by2 + pad_y)
        # Use trim only if it materially reduces blank margins.
        used_trim = (cx2 - cx1 + 1) <= int(round(0.98 * w)) or (cy2 - cy1 + 1) <= int(round(0.98 * h))

    crop = img.crop((cx1, cy1, cx2 + 1, cy2 + 1)) if used_trim else img
    if autocontrast_cutoff > 0:
        crop = ImageOps.autocontrast(crop, cutoff=int(autocontrast_cutoff))
    if upscale > 1.0:
        nw = max(1, int(round(crop.width * upscale)))
        nh = max(1, int(round(crop.height * upscale)))
        crop = crop.resize((nw, nh), Image.Resampling.LANCZOS)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    crop.convert("RGB").save(dst_path)

    return {
        "crop_xyxy": [int(cx1), int(cy1), int(cx2), int(cy2)],
        "preprocessed_size_wh": [int(crop.width), int(crop.height)],
        "upscale": float(upscale),
        "autocontrast_cutoff": int(autocontrast_cutoff),
        "used_trim": bool(used_trim),
    }


def crop_page_fixed_margins(
    src_path: Path,
    dst_path: Path,
    *,
    left_ratio: float,
    right_ratio: float,
    top_ratio: float,
    bottom_ratio: float,
) -> dict[str, object]:
    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    if w <= 1 or h <= 1:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path)
        return {
            "src_size_wh": [int(w), int(h)],
            "crop_size_wh": [int(w), int(h)],
            "crop_xyxy": [0, 0, max(0, w - 1), max(0, h - 1)],
            "ratios": {
                "left": float(left_ratio),
                "right": float(right_ratio),
                "top": float(top_ratio),
                "bottom": float(bottom_ratio),
            },
        }

    left_ratio = max(0.0, min(0.49, float(left_ratio)))
    right_ratio = max(0.0, min(0.49, float(right_ratio)))
    top_ratio = max(0.0, min(0.49, float(top_ratio)))
    bottom_ratio = max(0.0, min(0.49, float(bottom_ratio)))

    x1 = int(round(w * left_ratio))
    x2 = int(round(w * (1.0 - right_ratio)))
    y1 = int(round(h * top_ratio))
    y2 = int(round(h * (1.0 - bottom_ratio)))

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    crop = img.crop((x1, y1, x2, y2))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(dst_path)
    return {
        "src_size_wh": [int(w), int(h)],
        "crop_size_wh": [int(crop.width), int(crop.height)],
        "crop_xyxy": [int(x1), int(y1), int(x2 - 1), int(y2 - 1)],
        "ratios": {
            "left": float(left_ratio),
            "right": float(right_ratio),
            "top": float(top_ratio),
            "bottom": float(bottom_ratio),
        },
    }


def _ocr_text_score(text: str) -> tuple[float, dict[str, object]]:
    cleaned = clean_ocr_text(text, strip_noise_lines=True)
    token_count = len(tokenize_words(cleaned.text))
    lines = [x for x in cleaned.text.splitlines() if x.strip()]
    nonempty_line_count = len(lines)
    lower_raw = str(text).lower()
    noise_hits = sum(1 for p in NOISE_HINTS if p in lower_raw)
    mojibake_hits = _count_mojibake_hints(text)
    symbol_only_line_count = 0
    for line in lines:
        if len(tokenize_words(line)) == 0 and any(not ch.isspace() for ch in line):
            symbol_only_line_count += 1
    total_chars = max(1, len(cleaned.text))
    alpha_chars = sum(1 for ch in cleaned.text if ch.isalpha())
    weird_chars = sum(1 for ch in cleaned.text if ord(ch) > 126 and ch not in "\n\r\t")
    alpha_ratio = alpha_chars / total_chars
    weird_ratio = weird_chars / total_chars

    score = 0.0
    score += 2.0 * token_count
    score += 0.12 * nonempty_line_count
    score += 20.0 * alpha_ratio
    score -= 8.0 * float(cleaned.dropped_line_count)
    score -= 30.0 * float(noise_hits)
    score -= 12.0 * float(mojibake_hits)
    score -= 2.0 * float(symbol_only_line_count)
    score -= 25.0 * weird_ratio

    return score, {
        "score": round(float(score), 6),
        "token_count": int(token_count),
        "nonempty_line_count": int(nonempty_line_count),
        "dropped_noise_line_count": int(cleaned.dropped_line_count),
        "noise_phrase_hits": int(noise_hits),
        "mojibake_hint_hits": int(mojibake_hits),
        "symbol_only_line_count": int(symbol_only_line_count),
        "alpha_ratio": round(float(alpha_ratio), 6),
        "weird_char_ratio": round(float(weird_ratio), 6),
    }


def _pick_best_text_candidate(candidates: list[tuple[str, str]]) -> tuple[str, str, list[dict[str, object]]]:
    if not candidates:
        return "none", "", []

    ranked: list[tuple[float, str, str, dict[str, object]]] = []
    details: list[dict[str, object]] = []
    for source, text in candidates:
        score, metrics = _ocr_text_score(text)
        row = {"source": source, **metrics}
        details.append(row)
        ranked.append((score, source, text, row))

    ranked.sort(key=lambda x: x[0], reverse=True)
    _, best_source, best_text, _ = ranked[0]
    details_sorted = sorted(details, key=lambda d: float(d["score"]), reverse=True)
    return best_source, best_text, details_sorted


def _is_usable_pdf_text_layer(text: str, min_tokens: int) -> bool:
    _, metrics = _ocr_text_score(text)
    token_count = int(metrics.get("token_count", 0))
    line_count = max(1, int(metrics.get("nonempty_line_count", 0)))
    symbol_only_count = int(metrics.get("symbol_only_line_count", 0))
    weird_ratio = float(metrics.get("weird_char_ratio", 0.0))
    symbol_line_ratio = symbol_only_count / line_count
    if token_count < max(1, int(min_tokens)):
        return False
    if symbol_line_ratio > 0.35:
        return False
    if weird_ratio > 0.08:
        return False
    return True


def _slice_tokens(tokens: list[str], start: int, end: int) -> str:
    if not tokens:
        return ""
    s = max(0, start)
    e = min(len(tokens) - 1, end)
    if e < s:
        return ""
    return " ".join(tokens[s : e + 1])


def _nearest_mapped_left(mapping: list[int], idx: int) -> tuple[int | None, int | None]:
    j = idx
    while j >= 0:
        mj = mapping[j]
        if mj >= 0:
            return j, int(mj)
        j -= 1
    return None, None


def _nearest_mapped_right(mapping: list[int], idx: int) -> tuple[int | None, int | None]:
    j = idx
    n = len(mapping)
    while j < n:
        mj = mapping[j]
        if mj >= 0:
            return j, int(mj)
        j += 1
    return None, None


def build_chunk_report_text(
    *,
    page_key: str,
    redacted_raw_file: Path,
    unredacted_raw_file: Path,
    unred_tokens: list[str],
    red_tokens: list[str],
    mapping_unred_to_red: list[int],
    chunks: list[dict[str, object]],
) -> str:
    out: list[str] = []
    out.append(f"Page Key: {page_key}")
    out.append(f"Redacted OCR file: {redacted_raw_file}")
    out.append(f"Unredacted OCR file: {unredacted_raw_file}")
    out.append("")
    out.append("Legend:")
    out.append("- [UNRED_ONLY: ...] = tokens present in unredacted but unmatched in redacted mapping")
    out.append("- [L@i:tok] and [R@j:tok] = nearest mapped anchors in redacted tokens")
    out.append("- [RED_GAP: ...] = redacted-token segment between anchors")
    out.append("")
    out.append(f"Predicted redaction chunks: {len(chunks)}")
    out.append("")

    for pi, chunk in enumerate(chunks, start=1):
        cs = int(chunk.get("start_token_idx_0based", -1))
        ce = int(chunk.get("end_token_idx_0based", -1))
        ctext = str(chunk.get("text", ""))
        left_ctx = _slice_tokens(unred_tokens, cs - 3, cs - 1)
        right_ctx = _slice_tokens(unred_tokens, ce + 1, ce + 3)
        unred_ctx = " ".join(x for x in [left_ctx, f"[UNRED_ONLY: {ctext}]", right_ctx] if x).strip()

        _, l_anchor = _nearest_mapped_left(mapping_unred_to_red, cs - 1)
        _, r_anchor = _nearest_mapped_right(mapping_unred_to_red, ce + 1)
        if l_anchor is None and r_anchor is None:
            red_ctx = "(no mapped redacted anchors available)"
            red_gap = ""
        elif l_anchor is None:
            red_ctx = f"[R_ANCHOR_ONLY @ {r_anchor}] {_slice_tokens(red_tokens, r_anchor - 3, r_anchor + 3)}"
            red_gap = ""
        elif r_anchor is None:
            red_ctx = f"[L_ANCHOR_ONLY @ {l_anchor}] {_slice_tokens(red_tokens, l_anchor - 3, l_anchor + 3)}"
            red_gap = ""
        else:
            lo = min(l_anchor, r_anchor)
            hi = max(l_anchor, r_anchor)
            gap = _slice_tokens(red_tokens, lo + 1, hi - 1)
            red_gap = gap
            red_ctx = " ".join(
                x
                for x in [
                    _slice_tokens(red_tokens, lo - 3, lo - 1),
                    f"[L@{lo}:{red_tokens[lo]}]",
                    f"[RED_GAP:{gap or '<empty>'}]",
                    f"[R@{hi}:{red_tokens[hi]}]",
                    _slice_tokens(red_tokens, hi + 1, hi + 3),
                ]
                if x
            ).strip()

        out.append("=" * 72)
        out.append(f"Chunk #{pi} (token_span={cs}..{ce}, len={int(chunk.get('token_count', 0))})")
        out.append(f"Predicted chunk text: {ctext}")
        out.append(f"Unredacted context: {unred_ctx}")
        out.append(f"Redacted anchor context: {red_ctx}")
        out.append(f"Diff interpretation: [UNRED_ONLY] vs [RED_GAP:{red_gap or '<empty>'}]")
        out.append("")

    return "\n".join(out).strip() + "\n"


def write_mapping_ground_truth_txt(pair: PDFPair, page_index_1based: int, out_path: Path) -> None:
    row = pair.row
    lines = [
        f"Pair key: {pair.pair_key}",
        f"CSV row index (1-based): {pair.row_index_1based}",
        f"Page index (1-based): {page_index_1based}",
        "",
        "=== PDF MAPPING ===",
        f"Unredacted numeric id: {pair.unredacted_numeric_id}",
        f"Redacted doc id: {pair.redacted_doc_id}",
        f"Unredacted PDF: {pair.unredacted_pdf}",
        f"Redacted PDF: {pair.redacted_pdf}",
        "",
        "=== CSV FIELDS ===",
        f"Collection: {row[0]}",
        f"Unredacted id: {row[1]}",
        f"Date label: {row[2]}",
        f"Unredacted PDF URL: {row[3]}",
        f"Date numeric: {row[4]}",
        f"Redacted doc id: {row[5]}",
        f"Redacted doc URL: {row[6]}",
        f"Meta: {row[7]}",
        f"Title: {row[8]}",
        f"Year: {row[9]}",
        f"Flag: {row[10]}",
        "",
        "Note: textual redaction ground-truth snippets are not provided in this CSV.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pdf_docs_pipeline(
    *,
    docs_root: Path,
    csv_name: str,
    redacted_dir_name: str,
    unredacted_dir_name: str,
    out_dir: Path,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    ngram_size: int,
    window_size: int,
    max_model_len: int | None,
    ocr_backend: str,
    hf_attn_implementation: str,
    hf_dtype: str,
    dpi: int,
    max_pairs: int | None,
    max_pages: int | None,
    skip_ocr: bool,
    use_pdf_text_layer: bool,
    min_pdf_text_tokens: int,
    page_preprocess: bool,
    trim_border_ratio: float,
    content_pad_ratio: float,
    ocr_upscale: float,
    autocontrast_cutoff: int,
    crop_left_ratio: float,
    crop_right_ratio: float,
    crop_top_ratio: float,
    crop_bottom_ratio: float,
) -> None:
    docs_root = docs_root.resolve()
    csv_path = docs_root / csv_name
    redacted_dir = docs_root / redacted_dir_name
    unredacted_dir = docs_root / unredacted_dir_name

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    if not redacted_dir.exists():
        raise SystemExit(f"Redacted PDF folder not found: {redacted_dir}")
    if not unredacted_dir.exists():
        raise SystemExit(f"Unredacted PDF folder not found: {unredacted_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    demo_dir = out_dir / "demo"
    rendered_dir = out_dir / "_rendered_pages"
    demo_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir.mkdir(parents=True, exist_ok=True)

    all_pairs, pair_stats = collect_pdf_pairs_with_stats(
        csv_path=csv_path,
        redacted_dir=redacted_dir,
        unredacted_dir=unredacted_dir,
        max_pairs=None,
    )
    if max_pairs is None:
        pairs = all_pairs
    else:
        pairs = all_pairs[: max(0, int(max_pairs))]

    dropped_rows = max(0, pair_stats.rows_total - pair_stats.rows_valid)
    print(
        "[PAIRING] "
        f"rows_total={pair_stats.rows_total} valid_pairs={pair_stats.rows_valid} "
        f"dropped_rows={dropped_rows} selected_pairs={len(pairs)}"
    )
    print(
        "[PAIRING] dropped_detail "
        f"missing_required={pair_stats.rows_missing_required_fields} "
        f"invalid_unredacted_id={pair_stats.rows_invalid_unredacted_id_format} "
        f"missing_redacted_pdf={pair_stats.rows_missing_redacted_pdf} "
        f"missing_unredacted_pdf={pair_stats.rows_missing_unredacted_pdf} "
        f"duplicate_unredacted_id={pair_stats.rows_duplicate_unredacted_id} "
        f"duplicate_redacted_id={pair_stats.rows_duplicate_redacted_id}"
    )
    if not pairs:
        raise SystemExit("No PDF pairs found from CSV + local files.")

    llm = None
    if not skip_ocr:
        llm = build_llm(
            model_id=model,
            max_model_len=max_model_len,
            ocr_backend=ocr_backend,
            hf_attn_implementation=hf_attn_implementation,
            hf_dtype=hf_dtype,
        )

    manifest_rows: list[dict[str, object]] = []
    total_pages = 0

    pair_iter = _progress_iter(
        pairs,
        total=len(pairs),
        desc="PDF pairs",
        leave=False,
    )
    for pair_idx, pair in enumerate(pair_iter, start=1):
        pair_render_dir = rendered_dir / pair.pair_key
        red_img_dir = pair_render_dir / "redacted"
        unred_img_dir = pair_render_dir / "unredacted"
        crop_dir = pair_render_dir / "cropped"
        if pair_render_dir.exists():
            shutil.rmtree(pair_render_dir)
        pair_render_dir.mkdir(parents=True, exist_ok=True)

        red_pages = render_pdf_to_images(
            pdf_path=pair.redacted_pdf,
            out_dir=red_img_dir,
            prefix=f"{pair.pair_key}_redacted",
            dpi=dpi,
            max_pages=max_pages,
        )
        unred_pages = render_pdf_to_images(
            pdf_path=pair.unredacted_pdf,
            out_dir=unred_img_dir,
            prefix=f"{pair.pair_key}_unredacted",
            dpi=dpi,
            max_pages=max_pages,
        )

        page_count = min(len(red_pages), len(unred_pages))
        if page_count <= 0:
            continue

        page_iter = _progress_iter(
            range(page_count),
            total=page_count,
            desc=f"{pair.pair_key} pages",
            leave=False,
        )
        for page_i in page_iter:
            page_no = page_i + 1
            total_pages += 1
            page_key = _clean_filename_component(f"{pair.pair_key}__p{page_no:04d}")
            doc_demo_dir = demo_dir / page_key
            if doc_demo_dir.exists():
                shutil.rmtree(doc_demo_dir)
            doc_demo_dir.mkdir(parents=True, exist_ok=True)

            red_img_full = red_pages[page_i]
            unred_img_full = unred_pages[page_i]
            red_img_src = crop_dir / f"{page_key}_redacted.crop.png"
            unred_img_src = crop_dir / f"{page_key}_unredacted.crop.png"
            red_crop_meta = crop_page_fixed_margins(
                src_path=red_img_full,
                dst_path=red_img_src,
                left_ratio=crop_left_ratio,
                right_ratio=crop_right_ratio,
                top_ratio=crop_top_ratio,
                bottom_ratio=crop_bottom_ratio,
            )
            unred_crop_meta = crop_page_fixed_margins(
                src_path=unred_img_full,
                dst_path=unred_img_src,
                left_ratio=crop_left_ratio,
                right_ratio=crop_right_ratio,
                top_ratio=crop_top_ratio,
                bottom_ratio=crop_bottom_ratio,
            )
            red_img_dst = doc_demo_dir / f"{page_key}_redacted.png"
            unred_img_dst = doc_demo_dir / f"{page_key}_unredacted.png"
            shutil.copy2(red_img_src, red_img_dst)
            shutil.copy2(unred_img_src, unred_img_dst)

            red_raw_path = doc_demo_dir / f"{page_key}_redacted.raw_ocr.txt"
            unred_raw_path = doc_demo_dir / f"{page_key}_unredacted.raw_ocr.txt"
            red_ocr_meta: dict[str, object] = {"selected_source": "skipped"}
            unred_ocr_meta: dict[str, object] = {"selected_source": "skipped"}

            if skip_ocr:
                red_raw = "[OCR_SKIPPED]"
                unred_raw = "[OCR_SKIPPED]"
            else:
                assert llm is not None
                red_raw = ocr_with_model(
                    llm=llm,
                    image_path=red_img_src,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    ngram_size=ngram_size,
                    window_size=window_size,
                )
                unred_raw = ocr_with_model(
                    llm=llm,
                    image_path=unred_img_src,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    ngram_size=ngram_size,
                    window_size=window_size,
                )
                red_ocr_meta = {"selected_source": "ocr_cropped_fixed_margin"}
                unred_ocr_meta = {"selected_source": "ocr_cropped_fixed_margin"}

            red_raw_path.write_text(red_raw.rstrip() + "\n", encoding="utf-8")
            unred_raw_path.write_text(unred_raw.rstrip() + "\n", encoding="utf-8")

            red_clean = clean_ocr_text(red_raw, strip_noise_lines=True)
            unred_clean = clean_ocr_text(unred_raw, strip_noise_lines=True)
            red_tokens = tokenize_words(red_clean.text)
            unred_tokens = tokenize_words(unred_clean.text)
            unred_to_red = align_monotonic(unred_tokens, red_tokens)
            unred_mask = [1 if m < 0 else 0 for m in unred_to_red]
            chunks = build_chunks_from_mask(unred_tokens, unred_mask)

            bracketed_text = annotate_text_with_redaction_mask(
                unred_raw,
                token_mask=unred_mask,
                label_prefix="PRED_REDACTION",
            )
            bracketed_path = doc_demo_dir / f"{page_key}.unredacted_bracketed.txt"
            bracketed_path.write_text(bracketed_text, encoding="utf-8")

            chunk_report = build_chunk_report_text(
                page_key=page_key,
                redacted_raw_file=red_raw_path,
                unredacted_raw_file=unred_raw_path,
                unred_tokens=unred_tokens,
                red_tokens=red_tokens,
                mapping_unred_to_red=unred_to_red,
                chunks=chunks,
            )
            chunk_report_path = doc_demo_dir / f"{page_key}.redaction_chunks.txt"
            chunk_report_path.write_text(chunk_report, encoding="utf-8")

            gt_path = doc_demo_dir / "_ground_truth.txt"
            write_mapping_ground_truth_txt(pair=pair, page_index_1based=page_no, out_path=gt_path)

            manifest = {
                "pair_index_1based": pair_idx,
                "page_no_1based": page_no,
                "page_key": page_key,
                "pair_key": pair.pair_key,
                "redacted_pdf": str(pair.redacted_pdf),
                "unredacted_pdf": str(pair.unredacted_pdf),
                "demo_folder": str(doc_demo_dir),
                "files": {
                    "redacted_png": str(red_img_dst),
                    "unredacted_png": str(unred_img_dst),
                    "redacted_raw_ocr": str(red_raw_path),
                    "unredacted_raw_ocr": str(unred_raw_path),
                    "redaction_chunks_txt": str(chunk_report_path),
                    "unredacted_bracketed_txt": str(bracketed_path),
                    "mapping_ground_truth_txt": str(gt_path),
                },
                "predicted_chunk_count": len(chunks),
                "skip_ocr": bool(skip_ocr),
                "ocr_selection": {
                    "redacted": red_ocr_meta,
                    "unredacted": unred_ocr_meta,
                },
                "preprocess": {
                    "enabled": False,
                    "mode": "fixed_margin_crop_only",
                    "crop_left_ratio": float(crop_left_ratio),
                    "crop_right_ratio": float(crop_right_ratio),
                    "crop_top_ratio": float(crop_top_ratio),
                    "crop_bottom_ratio": float(crop_bottom_ratio),
                    "redacted_crop_meta": red_crop_meta,
                    "unredacted_crop_meta": unred_crop_meta,
                },
            }
            manifest_rows.append(manifest)
            # Keep console output clean: tqdm progress bars + final summaries only.

    manifest_jsonl = out_dir / "docs_manifest.jsonl"
    manifest_json = out_dir / "docs_manifest.json"
    with manifest_jsonl.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_json.write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] PDF pairs processed: {len(pairs)}")
    print(f"[INFO] Page bundles created: {len(manifest_rows)} (total pages seen: {total_pages})")
    print(f"[INFO] Saved demo bundles under: {demo_dir}")
    print(f"[INFO] Saved manifests: {manifest_jsonl} and {manifest_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_root", default="docs_example", help="Root folder containing csv + redacted_pdfs + unredacted_pdfs")
    parser.add_argument("--csv_name", default="cibcia.csv", help="CSV file name under docs_root")
    parser.add_argument("--redacted_dir_name", default="redacted_pdfs", help="Redacted PDF folder name under docs_root")
    parser.add_argument("--unredacted_dir_name", default="unredacted_pdfs", help="Unredacted PDF folder name under docs_root")
    parser.add_argument("--out", default="docs_diagnostics", help="Output folder")
    parser.add_argument("--dpi", type=int, default=300, help="PDF render DPI")
    parser.add_argument("--max_pairs", type=int, default=None, help="Optional cap on PDF pairs")
    parser.add_argument("--max_pages", type=int, default=None, help="Optional cap on pages per PDF")
    parser.add_argument("--skip_ocr", action="store_true", help="Skip OCR calls and write placeholder text")
    parser.add_argument("--crop_left_ratio", type=float, default=0.15, help="Fixed left crop ratio per page")
    parser.add_argument("--crop_right_ratio", type=float, default=0.10, help="Fixed right crop ratio per page")
    parser.add_argument("--crop_top_ratio", type=float, default=0.10, help="Fixed top crop ratio per page")
    parser.add_argument("--crop_bottom_ratio", type=float, default=0.10, help="Fixed bottom crop ratio per page")

    parser.add_argument("--model", default=MODEL_ID, help="Model id or local path")
    parser.add_argument("--prompt", default=PROMPT_FREE_OCR, help="OCR prompt")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Generation max tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--ngram_size", type=int, default=30, help="N-gram no-repeat size")
    parser.add_argument("--window_size", type=int, default=90, help="N-gram no-repeat window")
    parser.add_argument("--max_model_len", type=int, default=None, help="Override vLLM max_model_len")
    parser.add_argument("--ocr_backend", choices=["auto", "vllm", "hf"], default="auto", help="OCR runtime backend")
    parser.add_argument("--hf_attn_implementation", default="eager", help="HF attention impl for --ocr_backend hf")
    parser.add_argument("--hf_dtype", default="bfloat16", help="HF dtype for --ocr_backend hf")

    args = parser.parse_args()

    run_pdf_docs_pipeline(
        docs_root=Path(args.docs_root),
        csv_name=args.csv_name,
        redacted_dir_name=args.redacted_dir_name,
        unredacted_dir_name=args.unredacted_dir_name,
        out_dir=Path(args.out),
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        ngram_size=args.ngram_size,
        window_size=args.window_size,
        max_model_len=args.max_model_len,
        ocr_backend=args.ocr_backend,
        hf_attn_implementation=args.hf_attn_implementation,
        hf_dtype=args.hf_dtype,
        dpi=args.dpi,
        max_pairs=args.max_pairs,
        max_pages=args.max_pages,
        skip_ocr=bool(args.skip_ocr),
        use_pdf_text_layer=False,
        min_pdf_text_tokens=0,
        page_preprocess=False,
        trim_border_ratio=0.0,
        content_pad_ratio=0.0,
        ocr_upscale=1.0,
        autocontrast_cutoff=0,
        crop_left_ratio=args.crop_left_ratio,
        crop_right_ratio=args.crop_right_ratio,
        crop_top_ratio=args.crop_top_ratio,
        crop_bottom_ratio=args.crop_bottom_ratio,
    )


if __name__ == "__main__":
    main()
