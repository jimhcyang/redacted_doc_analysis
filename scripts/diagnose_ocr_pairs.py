from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.diagnostics_compare import (
    apply_redactions_to_context,
    clean_ocr_text,
    compact_ws,
    line_structure_metrics,
    match_chunks_to_truth,
    summarize_mapping,
    token_level_metrics,
    tokenize_words,
    align_monotonic,
    build_chunks_from_mask,
    build_token_candidates,
)


RED_SUFFIX = "_redacted.raw_ocr.txt"
UNRED_SUFFIX = "_unredacted.raw_ocr.txt"


def _extract_example_id(name: str) -> int:
    m = re.match(r"^(\d+)", name)
    return int(m.group(1)) if m else 10**9


def _load_examples(examples_jsonl: Path) -> dict[int, dict[str, Any]]:
    if not examples_jsonl.exists():
        return {}
    out: dict[int, dict[str, Any]] = {}
    with examples_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ex_id = int(item["example_id"])
            out[ex_id] = item
    return out


def _collect_pairs(outputs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for red_path in sorted(outputs_dir.glob(f"*{RED_SUFFIX}")):
        stem = red_path.name[: -len(RED_SUFFIX)]
        ex_id = _extract_example_id(stem)
        unred_path = outputs_dir / f"{stem}{UNRED_SUFFIX}"
        if not unred_path.exists():
            continue
        rows.append(
            {
                "example_id": ex_id,
                "stem": stem,
                "redacted_file": red_path,
                "unredacted_file": unred_path,
            }
        )
    rows.sort(key=lambda r: (int(r["example_id"]), str(r["stem"])))
    return rows


def _resolve_reference_path(
    raw_path: str | None,
    *,
    examples_jsonl: Path,
) -> Path | None:
    if not raw_path:
        return None
    p = Path(str(raw_path))
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend(
            [
                p,
                examples_jsonl.parent / p,
                ROOT / p,
                ROOT / "inputs" / p,
                ROOT / "inputs" / "examples" / p,
                ROOT / "example_batch" / p.name,
            ]
        )
    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists():
            return c
    return None


def _copy_if_exists(src: Path | None, dst: Path) -> bool:
    if src is None or (not src.exists()) or (not src.is_file()):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _build_demo_bundle(
    *,
    demo_root: Path,
    example_id: int,
    stem: str,
    redacted_raw_file: Path,
    unredacted_raw_file: Path,
    chunk_report_path: Path,
    gt_item: dict[str, Any] | None,
    examples_jsonl: Path,
) -> tuple[Path, dict[str, str], list[str]]:
    doc_dir = demo_root / f"{example_id:03d}"
    if doc_dir.exists():
        shutil.rmtree(doc_dir)
    doc_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    missing: list[str] = []

    red_name = redacted_raw_file.name.replace(".raw_ocr.txt", "")
    unred_name = unredacted_raw_file.name.replace(".raw_ocr.txt", "")

    # Core inspection artifacts.
    if _copy_if_exists(redacted_raw_file, doc_dir / redacted_raw_file.name):
        copied["redacted_raw_ocr"] = str(doc_dir / redacted_raw_file.name)
    else:
        missing.append("redacted_raw_ocr")

    if _copy_if_exists(unredacted_raw_file, doc_dir / unredacted_raw_file.name):
        copied["unredacted_raw_ocr"] = str(doc_dir / unredacted_raw_file.name)
    else:
        missing.append("unredacted_raw_ocr")

    if _copy_if_exists(chunk_report_path, doc_dir / chunk_report_path.name):
        copied["chunk_report_txt"] = str(doc_dir / chunk_report_path.name)
    else:
        missing.append("chunk_report_txt")

    # Source images from dataset metadata.
    red_img_src = _resolve_reference_path(
        str(gt_item.get("redacted_path")) if gt_item else None,
        examples_jsonl=examples_jsonl,
    )
    unred_img_src = _resolve_reference_path(
        str(gt_item.get("unredacted_path")) if gt_item else None,
        examples_jsonl=examples_jsonl,
    )

    red_img_dst = doc_dir / f"{stem}_redacted.png"
    unred_img_dst = doc_dir / f"{stem}_unredacted.png"
    if _copy_if_exists(red_img_src, red_img_dst):
        copied["redacted_image"] = str(red_img_dst)
    else:
        missing.append("redacted_image")
    if _copy_if_exists(unred_img_src, unred_img_dst):
        copied["unredacted_image"] = str(unred_img_dst)
    else:
        missing.append("unredacted_image")

    # Human-readable GT text file for quick inspection.
    gt_txt_path = doc_dir / "_ground_truth.txt"
    if gt_item is not None:
        context = str(gt_item.get("context", "")).strip()
        redactions = [str(x).strip() for x in gt_item.get("redactions", [])]
        context_filled = apply_redactions_to_context(context, redactions)

        lines: list[str] = []
        lines.append(f"Example ID: {example_id}")
        lines.append(f"Doc stem: {stem}")
        lines.append(f"Date: {gt_item.get('date', '')}")
        lines.append(f"doc_id_r: {gt_item.get('doc_id_r', '')}")
        lines.append(f"doc_id_u: {gt_item.get('doc_id_u', '')}")
        lines.append("")
        lines.append("=== CONTEXT (WITH <redaction n> TAGS) ===")
        lines.append(context)
        lines.append("")
        lines.append("=== REDACTIONS ===")
        if redactions:
            for i, r in enumerate(redactions, start=1):
                lines.append(f"[redaction {i}] {r}")
        else:
            lines.append("(none)")
        lines.append("")
        lines.append("=== CONTEXT (REDACTIONS FILLED) ===")
        lines.append(context_filled)
        lines.append("")
        gt_txt_path.write_text("\n".join(lines), encoding="utf-8")
        copied["ground_truth_txt"] = str(gt_txt_path)
    else:
        gt_txt_path.write_text(
            "\n".join(
                [
                    f"Example ID: {example_id}",
                    f"Doc stem: {stem}",
                    "",
                    "Ground-truth row not found in examples.jsonl for this example_id.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        copied["ground_truth_txt"] = str(gt_txt_path)
        missing.append("ground_truth_row")
    return doc_dir, copied, missing


def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except Exception:
        return 0


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    header = [
        "example_id",
        "structure_preserved",
        "line_similarity",
        "unred_tokens",
        "red_tokens",
        "unred_only_words",
        "unred_only_chunks",
        "red_only_words",
        "truth_word_recall",
        "truth_word_precision",
        "truth_word_f1",
        "truth_redactions",
        "pred_redaction_chunks",
        "correct_redactions",
        "hallucinated_redactions",
        "missed_redactions",
        "perfect_redaction_match_rate",
        "strict_avg_pair_f1_on_correct",
        "relaxed_gt_found_count",
        "relaxed_gt_found_rate",
        "relaxed_avg_best_f1_over_gt",
        "demo_folder",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write(
                "\t".join(
                    [
                        str(row.get("example_id", "")),
                        str(row.get("structure", {}).get("structure_preserved", "")),
                        str(row.get("structure", {}).get("line_similarity", "")),
                        str(row.get("token_counts", {}).get("unredacted", "")),
                        str(row.get("token_counts", {}).get("redacted", "")),
                        str(row.get("unredacted_only", {}).get("word_count", "")),
                        str(row.get("unredacted_only", {}).get("chunk_count", "")),
                        str(row.get("redacted_only", {}).get("word_count", "")),
                        str(row.get("full_context_vs_truth", {}).get("recall", "")),
                        str(row.get("full_context_vs_truth", {}).get("precision", "")),
                        str(row.get("full_context_vs_truth", {}).get("f1", "")),
                        str(row.get("redaction_eval", {}).get("truth_redaction_count", "")),
                        str(row.get("redaction_eval", {}).get("predicted_redaction_count", "")),
                        str(row.get("redaction_eval", {}).get("correctly_found_redactions", "")),
                        str(row.get("redaction_eval", {}).get("hallucinated_redactions", "")),
                        str(row.get("redaction_eval", {}).get("missed_redactions", "")),
                        str(row.get("redaction_eval", {}).get("perfect_text_match_rate_on_correct", "")),
                        str(row.get("redaction_eval", {}).get("strict_avg_pair_f1_on_correct", "")),
                        str(row.get("redaction_eval", {}).get("relaxed_gt_found_count", "")),
                        str(row.get("redaction_eval", {}).get("relaxed_gt_found_rate", "")),
                        str(row.get("redaction_eval", {}).get("relaxed_avg_best_f1_over_gt", "")),
                        str(row.get("demo_folder", "")),
                    ]
                )
                + "\n"
            )


def _slice_tokens(tokens: list[str], start: int, end: int) -> str:
    if not tokens:
        return ""
    s = max(0, start)
    e = min(len(tokens) - 1, end)
    if e < s:
        return ""
    return " ".join(tokens[s : e + 1])


def _render_unred_context(tokens: list[str], start: int, end: int, pad: int = 3) -> str:
    left = _slice_tokens(tokens, start - pad, start - 1)
    mid = _slice_tokens(tokens, start, end)
    right = _slice_tokens(tokens, end + 1, end + pad)
    parts = []
    if left:
        parts.append(left)
    parts.append(f"[UNRED_ONLY: {mid}]")
    if right:
        parts.append(right)
    return " ".join(parts).strip()


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


def _render_red_context(
    red_tokens: list[str],
    mapping_unred_to_red: list[int],
    chunk_start: int,
    chunk_end: int,
    pad: int = 3,
) -> tuple[str, str]:
    _, left_anchor = _nearest_mapped_left(mapping_unred_to_red, chunk_start - 1)
    _, right_anchor = _nearest_mapped_right(mapping_unred_to_red, chunk_end + 1)

    if left_anchor is None and right_anchor is None:
        return "(no mapped redacted anchors available)", ""

    if left_anchor is None:
        win_l = max(0, right_anchor - pad)
        win_r = min(len(red_tokens) - 1, right_anchor + pad)
        ctx = _slice_tokens(red_tokens, win_l, win_r)
        return f"[R_ANCHOR_ONLY @ {right_anchor}] {ctx}", ""

    if right_anchor is None:
        win_l = max(0, left_anchor - pad)
        win_r = min(len(red_tokens) - 1, left_anchor + pad)
        ctx = _slice_tokens(red_tokens, win_l, win_r)
        return f"[L_ANCHOR_ONLY @ {left_anchor}] {ctx}", ""

    lo = min(left_anchor, right_anchor)
    hi = max(left_anchor, right_anchor)
    left_ctx = _slice_tokens(red_tokens, lo - pad, lo - 1)
    left_tok = red_tokens[lo] if 0 <= lo < len(red_tokens) else ""
    right_tok = red_tokens[hi] if 0 <= hi < len(red_tokens) else ""
    right_ctx = _slice_tokens(red_tokens, hi + 1, hi + pad)
    gap = _slice_tokens(red_tokens, lo + 1, hi - 1)

    rendered = []
    if left_ctx:
        rendered.append(left_ctx)
    rendered.append(f"[L@{lo}:{left_tok}]")
    rendered.append(f"[RED_GAP:{gap or '<empty>'}]")
    rendered.append(f"[R@{hi}:{right_tok}]")
    if right_ctx:
        rendered.append(right_ctx)
    return " ".join(rendered), gap


def _best_relaxed_for_pred_chunk(
    pred_index_0based: int,
    best_match_per_gt: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score = -1.0
    for item in best_match_per_gt:
        s = int(item.get("best_chunk_start_0based", -1))
        e = int(item.get("best_chunk_end_0based", -1))
        if s <= pred_index_0based <= e:
            score = float(item.get("best_score_f1", 0.0) or 0.0)
            if score > best_score:
                best_score = score
                best = item
    return best


def _build_chunk_report_text(
    *,
    example_id: int,
    stem: str,
    redacted_file: Path,
    unredacted_file: Path,
    unred_tokens: list[str],
    red_tokens: list[str],
    mapping_unred_to_red: list[int],
    unred_only_chunks: list[dict[str, Any]],
    redaction_eval: dict[str, Any],
    gt_redactions: list[str],
) -> str:
    strict_by_pred: dict[int, dict[str, Any]] = {}
    for m in list(redaction_eval.get("matched_pairs", [])):
        strict_by_pred[int(m.get("pred_index_0based", -1))] = m
    relaxed_list = list(redaction_eval.get("best_match_per_gt", []))

    out: list[str] = []
    out.append(f"Example ID: {example_id}")
    out.append(f"Stem: {stem}")
    out.append(f"Redacted OCR file: {redacted_file}")
    out.append(f"Unredacted OCR file: {unredacted_file}")
    out.append("")
    out.append("Legend:")
    out.append("- [UNRED_ONLY: ...] = tokens present in unredacted but unmatched in redacted mapping")
    out.append("- [L@i:tok] and [R@j:tok] = nearest mapped anchors in redacted tokens")
    out.append("- [RED_GAP: ...] = redacted-token segment between anchors")
    out.append("")
    out.append(f"Predicted redaction chunks: {len(unred_only_chunks)}")
    out.append(f"Ground-truth redactions: {len(gt_redactions)}")
    out.append("")

    if not unred_only_chunks:
        out.append("No predicted chunks.")
        return "\n".join(out).strip() + "\n"

    for pi, chunk in enumerate(unred_only_chunks):
        cid = int(chunk.get("chunk_id", pi + 1))
        cs = int(chunk.get("start_token_idx_0based", -1))
        ce = int(chunk.get("end_token_idx_0based", -1))
        ctext = str(chunk.get("text", ""))

        out.append("=" * 72)
        out.append(f"Chunk #{cid} (pred_index={pi}, token_span={cs}..{ce}, len={int(chunk.get('token_count', 0))})")
        out.append(f"Predicted chunk text: {ctext}")
        out.append(f"Unredacted context: {_render_unred_context(unred_tokens, cs, ce, pad=3)}")
        red_ctx, red_gap = _render_red_context(
            red_tokens,
            mapping_unred_to_red,
            chunk_start=cs,
            chunk_end=ce,
            pad=3,
        )
        out.append(f"Redacted anchor context: {red_ctx}")
        out.append(f"Diff interpretation: unredacted contains [UNRED_ONLY], while mapped redacted anchors show [RED_GAP:{red_gap or '<empty>'}]")

        strict = strict_by_pred.get(pi)
        if strict is not None:
            wm = dict(strict.get("word_metrics", {}))
            out.append("Strict GT match: YES")
            out.append(
                f"  -> GT[{int(strict.get('gt_index_0based', -1))}] score_f1={strict.get('score_f1')} exact={strict.get('exact_text_match')}"
            )
            out.append(f"  -> GT text: {strict.get('gt_text', '')}")
            out.append(
                "  -> Word metrics: "
                f"recall={wm.get('recall')} precision={wm.get('precision')} f1={wm.get('f1')} matched={wm.get('matched_tokens')}"
            )
        else:
            out.append("Strict GT match: NO")

        relaxed = _best_relaxed_for_pred_chunk(pi, relaxed_list)
        if relaxed is not None:
            bwm = dict(relaxed.get("best_word_metrics", {}))
            out.append(
                "Relaxed best GT candidate: "
                f"GT[{int(relaxed.get('gt_index_0based', -1))}] "
                f"best_f1={relaxed.get('best_score_f1')} "
                f"joined_pred_chunks={int(relaxed.get('best_chunk_start_0based', -1))}..{int(relaxed.get('best_chunk_end_0based', -1))}"
            )
            out.append(f"  -> GT text: {relaxed.get('gt_text', '')}")
            out.append(f"  -> Relaxed best pred text: {relaxed.get('best_pred_text', '')}")
            out.append(
                "  -> Relaxed word metrics: "
                f"recall={bwm.get('recall')} precision={bwm.get('precision')} f1={bwm.get('f1')}"
            )
        else:
            out.append("Relaxed best GT candidate: none")
        out.append("")

    return "\n".join(out).strip() + "\n"


def _fmt_float(v: Any, digits: int = 3) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return ""


def _print_terminal_tables(rows: list[dict[str, Any]], summary: dict[str, Any], max_rows: int | None = None) -> None:
    headers = [
        ("id", 4),
        ("struct", 6),
        ("lineSim", 7),
        ("uTok", 6),
        ("rTok", 6),
        ("uOnly", 6),
        ("uChunks", 7),
        ("rOnly", 6),
        ("ctxF1", 6),
        ("rGT", 4),
        ("rPred", 5),
        ("rOK", 4),
        ("rQual", 6),
        ("rRelax", 6),
    ]

    def clip(s: str, w: int) -> str:
        if len(s) <= w:
            return s
        if w <= 1:
            return s[:w]
        return s[: w - 1] + "."

    def render_line(vals: list[str]) -> str:
        parts = []
        for (h, w), v in zip(headers, vals):
            if h in {"id", "uTok", "rTok", "uOnly", "uChunks", "rOnly", "rGT", "rPred", "rOK"}:
                parts.append(clip(v, w).rjust(w))
            else:
                parts.append(clip(v, w).ljust(w))
        return " | ".join(parts)

    head = render_line([h for h, _ in headers])
    sep = "-+-".join("-" * w for _, w in headers)

    print("\n=== Per-File Diagnostics (Concise) ===")
    print(head)
    print(sep)

    shown = rows if max_rows is None else rows[: max(0, max_rows)]
    for row in shown:
        vals = [
            str(row.get("example_id", "")),
            "Y" if bool(row.get("structure", {}).get("structure_preserved")) else "N",
            _fmt_float(row.get("structure", {}).get("line_similarity", ""), 3),
            str(row.get("token_counts", {}).get("unredacted", "")),
            str(row.get("token_counts", {}).get("redacted", "")),
            str(row.get("unredacted_only", {}).get("word_count", "")),
            str(row.get("unredacted_only", {}).get("chunk_count", "")),
            str(row.get("redacted_only", {}).get("word_count", "")),
            _fmt_float(row.get("full_context_vs_truth", {}).get("f1", ""), 3),
            str(row.get("redaction_eval", {}).get("truth_redaction_count", "")),
            str(row.get("redaction_eval", {}).get("predicted_redaction_count", "")),
            str(row.get("redaction_eval", {}).get("correctly_found_redactions", "")),
            _fmt_float(row.get("redaction_eval", {}).get("strict_avg_pair_f1_on_correct", ""), 3),
            _fmt_float(row.get("redaction_eval", {}).get("relaxed_gt_found_rate", ""), 3),
        ]
        print(render_line(vals))

    if max_rows is not None and len(rows) > max_rows:
        print(f"... ({len(rows) - max_rows} more rows not shown)")

    print("\n=== Aggregate Diagnostics ===")
    agg_headers = [
        ("pairs", 6),
        ("structOK", 8),
        ("structRate", 10),
        ("truthTok", 8),
        ("matched", 8),
        ("halluc", 8),
        ("recall", 7),
        ("prec", 7),
        ("f1", 7),
        ("redGT", 6),
        ("redPred", 7),
        ("redOK", 6),
        ("redMiss", 7),
        ("redHall", 7),
        ("redQual", 7),
        ("redRlx", 7),
    ]
    agg_head = " | ".join(h.ljust(w) for h, w in agg_headers)
    agg_sep = "-+-".join("-" * w for _, w in agg_headers)
    agg_vals = [
        str(summary.get("pair_count", "")),
        str(summary.get("structure_preserved_count", "")),
        _fmt_float(summary.get("structure_preserved_rate", ""), 3),
        str(summary.get("truth_token_total", "")),
        str(summary.get("truth_token_matched", "")),
        str(summary.get("truth_token_hallucinated", "")),
        _fmt_float(summary.get("overall_truth_recall", ""), 3),
        _fmt_float(summary.get("overall_truth_precision", ""), 3),
        _fmt_float(summary.get("overall_truth_f1", ""), 3),
        str(summary.get("redaction_truth_total", "")),
        str(summary.get("redaction_predicted_total", "")),
        str(summary.get("redaction_correct_total", "")),
        str(summary.get("redaction_missed_total", "")),
        str(summary.get("redaction_hallucinated_total", "")),
        _fmt_float(summary.get("strict_avg_pair_f1_on_correct_overall", ""), 3),
        _fmt_float(summary.get("relaxed_gt_found_rate_overall", ""), 3),
    ]
    agg_row = " | ".join(v.rjust(w) if v.isdigit() else v.ljust(w) for (k, w), v in zip(agg_headers, agg_vals))
    print(agg_head)
    print(agg_sep)
    print(agg_row)


def _aggregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"pair_count": 0}

    structure_true = sum(1 for r in rows if bool(r.get("structure", {}).get("structure_preserved")))
    sum_truth_total = 0
    sum_truth_matched = 0
    sum_truth_hallucinated = 0
    sum_unred_only = 0
    sum_red_only = 0
    red_gt = 0
    red_pred = 0
    red_correct = 0
    red_hall = 0
    red_missed = 0
    red_correct_weighted_pair_f1_sum = 0.0
    red_relaxed_found = 0
    red_relaxed_best_f1_sum = 0.0

    for r in rows:
        ft = r.get("full_context_vs_truth", {})
        sum_truth_total += _safe_int(ft.get("reference_token_total"))
        sum_truth_matched += _safe_int(ft.get("matched_tokens"))
        sum_truth_hallucinated += _safe_int(ft.get("hallucinated_tokens"))
        sum_unred_only += _safe_int(r.get("unredacted_only", {}).get("word_count"))
        sum_red_only += _safe_int(r.get("redacted_only", {}).get("word_count"))

        reval = r.get("redaction_eval", {})
        red_gt += _safe_int(reval.get("truth_redaction_count"))
        red_pred += _safe_int(reval.get("predicted_redaction_count"))
        red_correct += _safe_int(reval.get("correctly_found_redactions"))
        red_hall += _safe_int(reval.get("hallucinated_redactions"))
        red_missed += _safe_int(reval.get("missed_redactions"))
        red_correct_weighted_pair_f1_sum += (
            float(reval.get("strict_avg_pair_f1_on_correct", 0.0) or 0.0)
            * _safe_int(reval.get("correctly_found_redactions"))
        )
        red_relaxed_found += _safe_int(reval.get("relaxed_gt_found_count"))
        red_relaxed_best_f1_sum += (
            float(reval.get("relaxed_avg_best_f1_over_gt", 0.0) or 0.0)
            * _safe_int(reval.get("truth_redaction_count"))
        )

    overall_recall = (sum_truth_matched / sum_truth_total) if sum_truth_total else 1.0
    overall_precision = (
        sum_truth_matched / (sum_truth_matched + sum_truth_hallucinated)
        if (sum_truth_matched + sum_truth_hallucinated)
        else 1.0
    )
    overall_f1 = (
        (2.0 * overall_recall * overall_precision) / (overall_recall + overall_precision)
        if (overall_recall + overall_precision)
        else 1.0
    )
    strict_avg_pair_f1_overall = (
        red_correct_weighted_pair_f1_sum / red_correct
        if red_correct
        else 0.0
    )
    relaxed_gt_found_rate_overall = (
        red_relaxed_found / red_gt
        if red_gt
        else 1.0
    )
    relaxed_avg_best_f1_overall = (
        red_relaxed_best_f1_sum / red_gt
        if red_gt
        else 1.0
    )

    return {
        "pair_count": n,
        "structure_preserved_count": structure_true,
        "structure_preserved_rate": round(structure_true / n, 6),
        "truth_token_total": sum_truth_total,
        "truth_token_matched": sum_truth_matched,
        "truth_token_hallucinated": sum_truth_hallucinated,
        "overall_truth_recall": round(overall_recall, 6),
        "overall_truth_precision": round(overall_precision, 6),
        "overall_truth_f1": round(overall_f1, 6),
        "unredacted_only_word_total": sum_unred_only,
        "redacted_only_word_total": sum_red_only,
        "redaction_truth_total": red_gt,
        "redaction_predicted_total": red_pred,
        "redaction_correct_total": red_correct,
        "redaction_hallucinated_total": red_hall,
        "redaction_missed_total": red_missed,
        "strict_avg_pair_f1_on_correct_overall": round(strict_avg_pair_f1_overall, 6),
        "relaxed_gt_found_total": red_relaxed_found,
        "relaxed_gt_found_rate_overall": round(relaxed_gt_found_rate_overall, 6),
        "relaxed_avg_best_f1_overall": round(relaxed_avg_best_f1_overall, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", default="outputs/examples_ocr2", help="Folder with *_redacted.raw_ocr.txt pairs")
    parser.add_argument("--examples_jsonl", default="inputs/examples.jsonl", help="Ground-truth examples file")
    parser.add_argument("--out_dir", default="diagnostics", help="Diagnostics output folder")
    parser.add_argument("--max_examples", type=int, default=None, help="Optional cap for quick runs")
    parser.add_argument(
        "--keep_noise_lines",
        action="store_true",
        help="Keep instruction/noise lines instead of dropping them before token analysis",
    )
    parser.add_argument(
        "--table_max_rows",
        type=int,
        default=None,
        help="Optional limit on rows printed in terminal table (default: print all)",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    examples_jsonl = Path(args.examples_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_example_dir = out_dir / "per_example"
    per_example_dir.mkdir(parents=True, exist_ok=True)
    demo_dir = out_dir / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    pairs = _collect_pairs(outputs_dir)
    if args.max_examples is not None:
        pairs = pairs[: args.max_examples]
    gt_map = _load_examples(examples_jsonl)

    summary_rows: list[dict[str, Any]] = []
    chunk_report_count = 0
    demo_bundle_count = 0

    for row in pairs:
        ex_id = int(row["example_id"])
        stem = str(row["stem"])
        red_raw = Path(row["redacted_file"]).read_text(encoding="utf-8")
        unred_raw = Path(row["unredacted_file"]).read_text(encoding="utf-8")

        red_clean = clean_ocr_text(red_raw, strip_noise_lines=not args.keep_noise_lines)
        unred_clean = clean_ocr_text(unred_raw, strip_noise_lines=not args.keep_noise_lines)

        red_tokens = tokenize_words(red_clean.text)
        unred_tokens = tokenize_words(unred_clean.text)
        unred_to_red = align_monotonic(unred_tokens, red_tokens)
        red_to_unred = align_monotonic(red_tokens, unred_tokens)

        unred_mask = [1 if m < 0 else 0 for m in unred_to_red]
        red_mask = [1 if m < 0 else 0 for m in red_to_unred]
        unred_only_chunks = build_chunks_from_mask(unred_tokens, unred_mask)

        structure = line_structure_metrics(redacted_raw_text=red_raw, unredacted_raw_text=unred_raw)

        gt_item = gt_map.get(ex_id)
        full_context_metrics: dict[str, Any] = {}
        redaction_eval: dict[str, Any] = {}
        gt_full_context = None
        gt_redactions: list[str] = []
        if gt_item is not None:
            gt_redactions = [str(x) for x in gt_item.get("redactions", [])]
            gt_full_context = apply_redactions_to_context(
                str(gt_item.get("context", "")),
                gt_redactions,
            )
            gt_clean = clean_ocr_text(gt_full_context, strip_noise_lines=True)
            gt_tokens = tokenize_words(gt_clean.text)
            full_context_metrics = token_level_metrics(gt_tokens, unred_tokens)
            full_context_metrics.pop("mapping_ref_to_hyp", None)
            redaction_eval = match_chunks_to_truth(
                predicted_chunks=unred_only_chunks,
                gt_redactions=gt_redactions,
            )

        red_candidates = build_token_candidates(red_tokens)
        mapping_rows = summarize_mapping(
            unredacted_tokens=unred_tokens,
            mapping_unred_to_red=unred_to_red,
            redacted_candidates=red_candidates,
        )

        detail = {
            "example_id": ex_id,
            "stem": stem,
            "files": {
                "redacted_raw_ocr": str(row["redacted_file"]),
                "unredacted_raw_ocr": str(row["unredacted_file"]),
            },
            "noise_filtering": {
                "strip_noise_lines": (not args.keep_noise_lines),
                "redacted_dropped_line_count": red_clean.dropped_line_count,
                "unredacted_dropped_line_count": unred_clean.dropped_line_count,
                "redacted_dropped_lines": red_clean.dropped_lines[:64],
                "unredacted_dropped_lines": unred_clean.dropped_lines[:64],
            },
            "structure": structure,
            "token_counts": {
                "redacted": len(red_tokens),
                "unredacted": len(unred_tokens),
            },
            "unredacted_only": {
                "mask": unred_mask,
                "word_count": sum(unred_mask),
                "chunk_count": len(unred_only_chunks),
                "chunks": unred_only_chunks,
                "word_pct_vs_unredacted": round((sum(unred_mask) / len(unred_tokens)) if unred_tokens else 0.0, 6),
            },
            "redacted_only": {
                "mask": red_mask,
                "word_count": sum(red_mask),
                "word_pct_vs_unredacted": round((sum(red_mask) / len(unred_tokens)) if unred_tokens else 0.0, 6),
            },
            "mapping_unredacted_to_redacted": mapping_rows,
            "ground_truth": {
                "has_ground_truth": gt_item is not None,
                "context_compact": compact_ws(gt_full_context or ""),
                "redactions": gt_redactions,
            },
            "full_context_vs_truth": full_context_metrics,
            "redaction_eval": redaction_eval,
        }

        detail_path = per_example_dir / f"{ex_id:03d}.diagnostics.json"
        detail_path.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")

        chunk_report_text = _build_chunk_report_text(
            example_id=ex_id,
            stem=stem,
            redacted_file=Path(row["redacted_file"]),
            unredacted_file=Path(row["unredacted_file"]),
            unred_tokens=unred_tokens,
            red_tokens=red_tokens,
            mapping_unred_to_red=unred_to_red,
            unred_only_chunks=unred_only_chunks,
            redaction_eval=redaction_eval,
            gt_redactions=gt_redactions,
        )
        chunk_report_path = per_example_dir / f"{ex_id:03d}.redaction_chunks.txt"
        chunk_report_path.write_text(chunk_report_text, encoding="utf-8")
        chunk_report_count += 1

        demo_folder, demo_copied, demo_missing = _build_demo_bundle(
            demo_root=demo_dir,
            example_id=ex_id,
            stem=stem,
            redacted_raw_file=Path(row["redacted_file"]),
            unredacted_raw_file=Path(row["unredacted_file"]),
            chunk_report_path=chunk_report_path,
            gt_item=gt_item,
            examples_jsonl=examples_jsonl,
        )
        demo_bundle_count += 1

        summary_rows.append(
            {
                "example_id": ex_id,
                "detail_file": str(detail_path),
                "chunk_report_file": str(chunk_report_path),
                "demo_folder": str(demo_folder),
                "demo_copied_artifacts": demo_copied,
                "demo_missing_artifacts": demo_missing,
                "structure": structure,
                "token_counts": detail["token_counts"],
                "unredacted_only": {
                    "word_count": detail["unredacted_only"]["word_count"],
                    "chunk_count": detail["unredacted_only"]["chunk_count"],
                    "word_pct_vs_unredacted": detail["unredacted_only"]["word_pct_vs_unredacted"],
                },
                "redacted_only": detail["redacted_only"],
                "full_context_vs_truth": full_context_metrics,
                "redaction_eval": redaction_eval,
            }
        )

    jsonl_path = out_dir / "pair_diagnostics.jsonl"
    tsv_path = out_dir / "pair_diagnostics.tsv"
    summary_path = out_dir / "pair_diagnostics.summary.json"
    aggregate_txt = out_dir / "pair_diagnostics.aggregate.txt"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in summary_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _write_tsv(tsv_path, summary_rows)

    summary = _aggregate_summary(summary_rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with aggregate_txt.open("w", encoding="utf-8") as f:
        for row in summary_rows:
            f.write(f"### example_id={row['example_id']}\n")
            f.write(f"structure_preserved={row['structure'].get('structure_preserved')}\n")
            f.write(f"chunk_report_file={row.get('chunk_report_file', '')}\n")
            f.write(f"demo_folder={row.get('demo_folder', '')}\n")
            f.write(f"line_similarity={row['structure'].get('line_similarity')}\n")
            f.write(f"unred_only_words={row['unredacted_only'].get('word_count')}\n")
            f.write(f"unred_only_chunks={row['unredacted_only'].get('chunk_count')}\n")
            f.write(f"red_only_words={row['redacted_only'].get('word_count')}\n")
            if row.get("full_context_vs_truth"):
                f.write(f"truth_recall={row['full_context_vs_truth'].get('recall')}\n")
                f.write(f"truth_precision={row['full_context_vs_truth'].get('precision')}\n")
                f.write(f"truth_f1={row['full_context_vs_truth'].get('f1')}\n")
            if row.get("redaction_eval"):
                f.write(f"truth_redaction_count={row['redaction_eval'].get('truth_redaction_count')}\n")
                f.write(f"predicted_redaction_count={row['redaction_eval'].get('predicted_redaction_count')}\n")
                f.write(f"correct_redactions={row['redaction_eval'].get('correctly_found_redactions')}\n")
                f.write(f"hallucinated_redactions={row['redaction_eval'].get('hallucinated_redactions')}\n")
                f.write(f"missed_redactions={row['redaction_eval'].get('missed_redactions')}\n")
                f.write(f"strict_avg_pair_f1_on_correct={row['redaction_eval'].get('strict_avg_pair_f1_on_correct')}\n")
                f.write(f"relaxed_gt_found_rate={row['redaction_eval'].get('relaxed_gt_found_rate')}\n")
                f.write(f"relaxed_avg_best_f1_over_gt={row['redaction_eval'].get('relaxed_avg_best_f1_over_gt')}\n")
            f.write("\n")

    print(f"[INFO] pairs: {len(summary_rows)}")
    print(f"[INFO] wrote: {jsonl_path}")
    print(f"[INFO] wrote: {tsv_path}")
    print(f"[INFO] wrote: {aggregate_txt}")
    print(f"[INFO] wrote: {summary_path}")
    print(f"[INFO] wrote per-example folder: {per_example_dir}")
    print(f"[INFO] wrote redaction chunk reports: {chunk_report_count}")
    print(f"[INFO] wrote demo bundles: {demo_bundle_count} -> {demo_dir}")
    _print_terminal_tables(summary_rows, summary, max_rows=args.table_max_rows)


if __name__ == "__main__":
    main()
