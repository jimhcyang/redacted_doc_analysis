from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pdf_docs_pipeline import PDFPair, collect_pdf_pairs
from core.pdf_text_chunk_pipeline import run_pdf_text_chunk_pipeline


def _canonical_unredacted_name(unredacted_id: str, fallback_name: str) -> str:
    try:
        return f"cib_{int(unredacted_id):08d}.pdf"
    except Exception:
        return fallback_name


def sample_pairs(
    *,
    source_docs_root: Path,
    csv_name: str,
    redacted_dir_name: str,
    unredacted_dir_name: str,
    sample_size: int,
    seed: int,
) -> tuple[list[PDFPair], int]:
    pairs = collect_pdf_pairs(
        csv_path=source_docs_root / csv_name,
        redacted_dir=source_docs_root / redacted_dir_name,
        unredacted_dir=source_docs_root / unredacted_dir_name,
        max_pairs=None,
    )
    if not pairs:
        raise SystemExit(f"No valid pairs found in source docs root: {source_docs_root}")

    if sample_size <= 0 or sample_size >= len(pairs):
        selected = list(pairs)
    else:
        rng = random.Random(seed)
        selected = rng.sample(pairs, sample_size)

    selected.sort(key=lambda p: p.row_index_1based)
    return selected, len(pairs)


def populate_target_docs_root(
    *,
    selected_pairs: list[PDFPair],
    target_docs_root: Path,
    csv_name: str,
    redacted_dir_name: str,
    unredacted_dir_name: str,
    clean_target: bool,
) -> None:
    red_dir = target_docs_root / redacted_dir_name
    unred_dir = target_docs_root / unredacted_dir_name
    csv_path = target_docs_root / csv_name

    if clean_target and target_docs_root.exists():
        if red_dir.exists():
            shutil.rmtree(red_dir)
        if unred_dir.exists():
            shutil.rmtree(unred_dir)
        if csv_path.exists():
            csv_path.unlink()

    target_docs_root.mkdir(parents=True, exist_ok=True)
    red_dir.mkdir(parents=True, exist_ok=True)
    unred_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for pair in selected_pairs:
            red_dst = red_dir / f"{pair.redacted_doc_id}.pdf"
            unred_dst = unred_dir / _canonical_unredacted_name(
                pair.unredacted_numeric_id,
                pair.unredacted_pdf.name,
            )
            shutil.copy2(pair.redacted_pdf, red_dst)
            shutil.copy2(pair.unredacted_pdf, unred_dst)
            writer.writerow(pair.row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample rows from docs/cibcia.csv, materialize a fake docs root, then run page text-chunk extraction.",
    )
    parser.add_argument("--source_docs_root", default="docs", help="Full source docs root")
    parser.add_argument("--target_docs_root", default="docs_example", help="Target sampled docs root")
    parser.add_argument("--csv_name", default="cibcia.csv", help="CSV filename")
    parser.add_argument("--redacted_dir_name", default="redacted_pdfs", help="Redacted PDF folder name")
    parser.add_argument("--unredacted_dir_name", default="unredacted_pdfs", help="Unredacted PDF folder name")
    parser.add_argument("--sample_size", type=int, default=10, help="How many valid rows/pairs to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_clean_target", action="store_true", help="Do not clean target_docs_root first")

    parser.add_argument("--out", default="docs_text_chunks", help="Chunk output root")
    parser.add_argument("--dpi", type=int, default=300, help="PDF render DPI")
    parser.add_argument("--max_pages", type=int, default=None, help="Optional pages per pdf cap")

    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--window_step", type=int, default=16)
    parser.add_argument("--window_ink_ratio", type=float, default=0.25)
    parser.add_argument("--border_percentile_q", type=float, default=1.0)
    parser.add_argument("--initial_trim_search_ratio", type=float, default=0.05)
    parser.add_argument("--initial_trim_anchor_white_ratio", type=float, default=0.90)
    parser.add_argument("--initial_trim_expand_white_ratio", type=float, default=0.997)
    parser.add_argument("--initial_trim_min_empty_area_ratio", type=float, default=0.05)
    parser.add_argument("--white_cell_size", type=int, default=32)
    parser.add_argument("--white_ratio", type=float, default=0.997)
    parser.add_argument("--min_zone_side", type=int, default=32)
    parser.add_argument("--min_zone_area_ratio", type=float, default=0.01)
    parser.add_argument("--min_zone_area_page_ratio", type=float, default=0.05)
    parser.add_argument("--max_zone_area_ratio", type=float, default=0.35)
    parser.add_argument("--min_text_side", type=int, default=32)
    parser.add_argument("--min_text_ink_ratio", type=float, default=0.01)
    args = parser.parse_args()

    source_root = Path(args.source_docs_root)
    target_root = Path(args.target_docs_root)

    selected, total = sample_pairs(
        source_docs_root=source_root,
        csv_name=args.csv_name,
        redacted_dir_name=args.redacted_dir_name,
        unredacted_dir_name=args.unredacted_dir_name,
        sample_size=int(args.sample_size),
        seed=int(args.seed),
    )
    print(f"[SAMPLE] available_valid_pairs={total} selected={len(selected)} seed={args.seed}")

    populate_target_docs_root(
        selected_pairs=selected,
        target_docs_root=target_root,
        csv_name=args.csv_name,
        redacted_dir_name=args.redacted_dir_name,
        unredacted_dir_name=args.unredacted_dir_name,
        clean_target=not bool(args.no_clean_target),
    )
    print(f"[POPULATE] wrote sampled docs root: {target_root}")

    run_pdf_text_chunk_pipeline(
        docs_root=target_root,
        csv_name=args.csv_name,
        redacted_dir_name=args.redacted_dir_name,
        unredacted_dir_name=args.unredacted_dir_name,
        out_dir=Path(args.out),
        dpi=int(args.dpi),
        max_pairs=None,
        max_pages=args.max_pages,
        window_size=int(args.window_size),
        window_step=int(args.window_step),
        window_ink_ratio=float(args.window_ink_ratio),
        border_percentile_q=float(args.border_percentile_q),
        initial_trim_search_ratio=float(args.initial_trim_search_ratio),
        initial_trim_anchor_white_ratio=float(args.initial_trim_anchor_white_ratio),
        initial_trim_expand_white_ratio=float(args.initial_trim_expand_white_ratio),
        initial_trim_min_empty_area_ratio=float(args.initial_trim_min_empty_area_ratio),
        white_cell_size=int(args.white_cell_size),
        white_ratio=float(args.white_ratio),
        min_zone_side=int(args.min_zone_side),
        min_zone_area_ratio=float(args.min_zone_area_ratio),
        min_zone_area_page_ratio=float(args.min_zone_area_page_ratio),
        max_zone_area_ratio=float(args.max_zone_area_ratio),
        min_text_side=int(args.min_text_side),
        min_text_ink_ratio=float(args.min_text_ink_ratio),
    )


if __name__ == "__main__":
    main()
