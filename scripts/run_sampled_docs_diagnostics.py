from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pdf_docs_pipeline import PDFPair, collect_pdf_pairs, run_pdf_docs_pipeline


def _canonical_unredacted_name(unredacted_id: str, fallback_name: str) -> str:
    try:
        return f"cib_{int(unredacted_id):08d}.pdf"
    except Exception:
        return fallback_name


def sample_pairs(
    *,
    source_root: Path,
    csv_name: str,
    redacted_dir_name: str,
    unredacted_dir_name: str,
    sample_size: int,
    seed: int,
) -> tuple[list[PDFPair], int]:
    csv_path = source_root / csv_name
    red_dir = source_root / redacted_dir_name
    unred_dir = source_root / unredacted_dir_name

    all_pairs = collect_pdf_pairs(
        csv_path=csv_path,
        redacted_dir=red_dir,
        unredacted_dir=unred_dir,
        max_pairs=None,
    )
    if not all_pairs:
        raise SystemExit(f"No valid PDF pairs found in source root: {source_root}")

    if sample_size <= 0 or sample_size >= len(all_pairs):
        selected = list(all_pairs)
    else:
        rng = random.Random(seed)
        selected = rng.sample(all_pairs, sample_size)

    # Stable output ordering for easier inspection.
    selected.sort(key=lambda p: p.row_index_1based)
    return selected, len(all_pairs)


def populate_docs_example(
    *,
    selected_pairs: list[PDFPair],
    target_root: Path,
    csv_name: str,
    redacted_dir_name: str,
    unredacted_dir_name: str,
    clean_target: bool,
) -> dict[str, object]:
    red_dir = target_root / redacted_dir_name
    unred_dir = target_root / unredacted_dir_name
    csv_path = target_root / csv_name

    if clean_target and target_root.exists():
        for p in [red_dir, unred_dir]:
            if p.exists():
                shutil.rmtree(p)
        if csv_path.exists():
            csv_path.unlink()

    target_root.mkdir(parents=True, exist_ok=True)
    red_dir.mkdir(parents=True, exist_ok=True)
    unred_dir.mkdir(parents=True, exist_ok=True)

    copied_red = 0
    copied_unred = 0

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for pair in selected_pairs:
            # Redacted PDF copied under canonical doc-id name.
            red_dst = red_dir / f"{pair.redacted_doc_id}.pdf"
            shutil.copy2(pair.redacted_pdf, red_dst)
            copied_red += 1

            # Unredacted PDF copied under canonical cib_<id>.pdf where possible.
            unred_name = _canonical_unredacted_name(
                pair.unredacted_numeric_id,
                fallback_name=pair.unredacted_pdf.name,
            )
            unred_dst = unred_dir / unred_name
            shutil.copy2(pair.unredacted_pdf, unred_dst)
            copied_unred += 1

            writer.writerow(pair.row)

    return {
        "target_root": str(target_root),
        "csv_path": str(csv_path),
        "copied_redacted_pdfs": copied_red,
        "copied_unredacted_pdfs": copied_unred,
    }


def _resolve_manifest_path(path_value: str) -> Path:
    p = Path(path_value)
    if p.exists():
        return p
    if not p.is_absolute():
        q = Path.cwd() / p
        if q.exists():
            return q
    return p


def aggregate_by_document(out_dir: Path) -> dict[str, object]:
    manifest_path = out_dir / "docs_manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found after pipeline run: {manifest_path}")

    rows: list[dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        pair_key = str(row.get("pair_key", "unknown_pair"))
        grouped[pair_key].append(row)

    agg_root = out_dir / "by_document"
    agg_root.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, object]] = []
    for pair_key, group in grouped.items():
        group_sorted = sorted(group, key=lambda r: int(r.get("page_no_1based", 0)))
        pair_dir = agg_root / pair_key
        if pair_dir.exists():
            shutil.rmtree(pair_dir)
        pair_dir.mkdir(parents=True, exist_ok=True)

        bracket_out = pair_dir / f"{pair_key}.unredacted_bracketed.full.txt"
        chunks_out = pair_dir / f"{pair_key}.redaction_chunks.full.txt"

        bracket_parts: list[str] = []
        chunk_parts: list[str] = []
        for row in group_sorted:
            page_no = int(row.get("page_no_1based", 0))
            page_key = str(row.get("page_key", "unknown_page"))
            files = row.get("files", {})
            if not isinstance(files, dict):
                continue

            bracket_path = _resolve_manifest_path(str(files.get("unredacted_bracketed_txt", "")))
            chunks_path = _resolve_manifest_path(str(files.get("redaction_chunks_txt", "")))

            bracket_text = bracket_path.read_text(encoding="utf-8") if bracket_path.exists() else ""
            chunks_text = chunks_path.read_text(encoding="utf-8") if chunks_path.exists() else ""

            bracket_parts.append(f"===== PAGE {page_no:04d} | {page_key} =====\n{bracket_text.strip()}\n")
            chunk_parts.append(f"===== PAGE {page_no:04d} | {page_key} =====\n{chunks_text.strip()}\n")

        bracket_out.write_text("\n".join(bracket_parts).rstrip() + "\n", encoding="utf-8")
        chunks_out.write_text("\n".join(chunk_parts).rstrip() + "\n", encoding="utf-8")

        index_rows.append(
            {
                "pair_key": pair_key,
                "page_count": len(group_sorted),
                "unredacted_bracketed_full": str(bracket_out),
                "redaction_chunks_full": str(chunks_out),
            }
        )

    index_rows.sort(key=lambda r: str(r["pair_key"]))
    index_json = agg_root / "by_document.index.json"
    index_jsonl = agg_root / "by_document.index.jsonl"
    index_json.write_text(json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with index_jsonl.open("w", encoding="utf-8") as f:
        for row in index_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "documents_aggregated": len(index_rows),
        "aggregate_root": str(agg_root),
        "index_json": str(index_json),
        "index_jsonl": str(index_jsonl),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample rows from cibcia.csv, populate docs_example, run PDF diagnostics, and aggregate per document.",
    )
    parser.add_argument("--source_docs_root", default="docs", help="Source folder containing full cibcia.csv and PDF folders")
    parser.add_argument("--target_docs_root", default="docs_example", help="Target folder to populate with sampled files")
    parser.add_argument("--csv_name", default="cibcia.csv", help="CSV file name in source/target docs roots")
    parser.add_argument("--redacted_dir_name", default="redacted_pdfs", help="Redacted PDF folder name")
    parser.add_argument("--unredacted_dir_name", default="unredacted_pdfs", help="Unredacted PDF folder name")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of rows/pairs to sample")
    parser.add_argument("--seed", type=int, default=42, help="Sampling random seed")
    parser.add_argument("--no_clean_target", action="store_true", help="Do not clear target docs root before copying sampled files")

    parser.add_argument("--out", default="docs_diagnostics", help="Diagnostics output folder")
    parser.add_argument("--dpi", type=int, default=300, help="PDF render DPI")
    parser.add_argument("--max_pages", type=int, default=None, help="Optional cap on pages per PDF in diagnostics run")
    parser.add_argument("--skip_ocr", action="store_true", help="Skip OCR calls and only build page/image structure")

    parser.add_argument("--model", default="deepseek-ai/DeepSeek-OCR-2", help="Model id or local path")
    parser.add_argument("--prompt", default="<image>\nFree OCR. Preserve original line breaks.", help="OCR prompt")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Generation max tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--ngram_size", type=int, default=30, help="N-gram no-repeat size")
    parser.add_argument("--window_size", type=int, default=90, help="N-gram no-repeat window")
    parser.add_argument("--max_model_len", type=int, default=None, help="Override vLLM max model length")
    parser.add_argument("--ocr_backend", choices=["auto", "vllm", "hf"], default="hf", help="OCR runtime backend")
    parser.add_argument("--hf_attn_implementation", default="eager", help="HF attention implementation")
    parser.add_argument("--hf_dtype", default="bfloat16", help="HF dtype for OCR backend hf")

    parser.add_argument("--no_pdf_text_layer", action="store_true", help="Disable PDF text-layer candidate extraction")
    parser.add_argument("--min_pdf_text_tokens", type=int, default=20, help="Minimum tokens for PDF text-layer candidate")
    parser.add_argument("--no_page_preprocess", action="store_true", help="Disable preprocessed OCR candidate")
    parser.add_argument("--trim_border_ratio", type=float, default=0.01, help="Outer border ratio to ignore before content crop")
    parser.add_argument("--content_pad_ratio", type=float, default=0.015, help="Padding ratio around detected content")
    parser.add_argument("--ocr_upscale", type=float, default=1.35, help="Upscale factor for preprocessed OCR candidate")
    parser.add_argument("--autocontrast_cutoff", type=int, default=1, help="Autocontrast cutoff percent")

    args = parser.parse_args()

    source_root = Path(args.source_docs_root)
    target_root = Path(args.target_docs_root)
    out_dir = Path(args.out)

    selected_pairs, total_available = sample_pairs(
        source_root=source_root,
        csv_name=args.csv_name,
        redacted_dir_name=args.redacted_dir_name,
        unredacted_dir_name=args.unredacted_dir_name,
        sample_size=int(args.sample_size),
        seed=int(args.seed),
    )
    print(f"[SAMPLE] valid pairs available={total_available} selected={len(selected_pairs)} seed={args.seed}")

    populate_stats = populate_docs_example(
        selected_pairs=selected_pairs,
        target_root=target_root,
        csv_name=args.csv_name,
        redacted_dir_name=args.redacted_dir_name,
        unredacted_dir_name=args.unredacted_dir_name,
        clean_target=not bool(args.no_clean_target),
    )
    print(
        f"[POPULATE] target={populate_stats['target_root']} redacted={populate_stats['copied_redacted_pdfs']} "
        f"unredacted={populate_stats['copied_unredacted_pdfs']}"
    )

    run_pdf_docs_pipeline(
        docs_root=target_root,
        csv_name=args.csv_name,
        redacted_dir_name=args.redacted_dir_name,
        unredacted_dir_name=args.unredacted_dir_name,
        out_dir=out_dir,
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
        max_pairs=None,
        max_pages=args.max_pages,
        skip_ocr=bool(args.skip_ocr),
        use_pdf_text_layer=not bool(args.no_pdf_text_layer),
        min_pdf_text_tokens=int(args.min_pdf_text_tokens),
        page_preprocess=not bool(args.no_page_preprocess),
        trim_border_ratio=float(args.trim_border_ratio),
        content_pad_ratio=float(args.content_pad_ratio),
        ocr_upscale=float(args.ocr_upscale),
        autocontrast_cutoff=int(args.autocontrast_cutoff),
    )

    agg_stats = aggregate_by_document(out_dir)
    print(
        f"[AGGREGATE] docs={agg_stats['documents_aggregated']} root={agg_stats['aggregate_root']} "
        f"index={agg_stats['index_json']}"
    )


if __name__ == "__main__":
    main()

