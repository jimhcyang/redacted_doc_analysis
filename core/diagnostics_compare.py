from __future__ import annotations

import difflib
import re
from collections import defaultdict
from dataclasses import dataclass


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
REDACTION_TAG_RE = re.compile(r"<\s*redaction\s*(\d+)\s*>", flags=re.IGNORECASE)

INSTRUCTION_LINE_PATTERNS = [
    re.compile(r"^\s*(preserve|ignore)\s+line\s*breaks?\b", flags=re.IGNORECASE),
    re.compile(r"^\s*do\s+not\s+.*line\s*breaks?\b", flags=re.IGNORECASE),
    re.compile(r"^\s*press\s+the\b", flags=re.IGNORECASE),
    re.compile(r"^\s*line\s*breaks?\s+(are|in)\b", flags=re.IGNORECASE),
    re.compile(r"^\s*approved\s+for\s+release\b", flags=re.IGNORECASE),
]
NOISE_LINE_PATTERNS = [
    re.compile(r"^\s*[#=\-]{3,}\s*$"),
]


@dataclass
class CleanTextResult:
    text: str
    dropped_lines: list[str]
    dropped_line_count: int


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    for rx in INSTRUCTION_LINE_PATTERNS:
        if rx.search(s):
            return True
    for rx in NOISE_LINE_PATTERNS:
        if rx.search(s):
            return True
    return False


def clean_ocr_text(raw_text: str, strip_noise_lines: bool = True) -> CleanTextResult:
    src = _normalize_newlines(raw_text)
    dropped: list[str] = []
    kept: list[str] = []
    for line in src.split("\n"):
        if strip_noise_lines and _is_noise_line(line):
            dropped.append(line)
            continue
        kept.append(line.rstrip())

    while kept and kept[0] == "":
        kept.pop(0)
    while kept and kept[-1] == "":
        kept.pop()

    return CleanTextResult(
        text="\n".join(kept),
        dropped_lines=dropped,
        dropped_line_count=len(dropped),
    )


def compact_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize_words(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def build_token_candidates(tokens: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for idx, tok in enumerate(tokens):
        out[tok].append(idx)
    return dict(out)


def align_monotonic(a_tokens: list[str], b_tokens: list[str]) -> list[int]:
    # SequenceMatcher gives a monotonic matching with a strong preference
    # for consecutive blocks, which matches the intended index-order rules.
    sm = difflib.SequenceMatcher(a=a_tokens, b=b_tokens, autojunk=False)
    mapping = [-1] * len(a_tokens)
    for block in sm.get_matching_blocks():
        if block.size <= 0:
            continue
        for k in range(block.size):
            mapping[block.a + k] = block.b + k
    return mapping


def build_chunks_from_mask(tokens: list[str], mask: list[int]) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    i = 0
    cid = 1
    n = len(tokens)
    while i < n:
        if mask[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1] == 1:
            j += 1
        chunk_tokens = tokens[i : j + 1]
        chunks.append(
            {
                "chunk_id": cid,
                "start_token_idx_0based": i,
                "end_token_idx_0based": j,
                "token_count": len(chunk_tokens),
                "text": " ".join(chunk_tokens),
                "tokens": chunk_tokens,
            }
        )
        cid += 1
        i = j + 1
    return chunks


def token_level_metrics(reference_tokens: list[str], hypothesis_tokens: list[str]) -> dict[str, object]:
    mapping = align_monotonic(reference_tokens, hypothesis_tokens)
    matched = sum(1 for v in mapping if v >= 0)
    ref_total = len(reference_tokens)
    hyp_total = len(hypothesis_tokens)
    missing = ref_total - matched
    hallucinated = hyp_total - matched

    recall = matched / ref_total if ref_total else 1.0
    precision = matched / hyp_total if hyp_total else 1.0
    f1 = (2.0 * matched / (ref_total + hyp_total)) if (ref_total + hyp_total) else 1.0
    net = (matched - hallucinated) / ref_total if ref_total else 1.0

    return {
        "reference_token_total": ref_total,
        "hypothesis_token_total": hyp_total,
        "matched_tokens": matched,
        "missing_tokens": missing,
        "hallucinated_tokens": hallucinated,
        "recall": round(recall, 6),
        "precision": round(precision, 6),
        "f1": round(f1, 6),
        "net_recovery_minus_hallucination": round(net, 6),
        "mapping_ref_to_hyp": mapping,
    }


def line_structure_metrics(redacted_raw_text: str, unredacted_raw_text: str) -> dict[str, object]:
    r_lines = _normalize_newlines(redacted_raw_text).split("\n")
    u_lines = _normalize_newlines(unredacted_raw_text).split("\n")

    r_nonempty = [x.strip() for x in r_lines if x.strip()]
    u_nonempty = [x.strip() for x in u_lines if x.strip()]

    ratio = (
        min(len(r_nonempty), len(u_nonempty)) / max(1, max(len(r_nonempty), len(u_nonempty)))
        if (r_nonempty or u_nonempty)
        else 1.0
    )
    line_similarity = difflib.SequenceMatcher(a=u_nonempty, b=r_nonempty, autojunk=False).ratio()
    preserved = ratio >= 0.75 and line_similarity >= 0.60

    return {
        "redacted_raw_line_count": len(r_lines),
        "unredacted_raw_line_count": len(u_lines),
        "redacted_nonempty_line_count": len(r_nonempty),
        "unredacted_nonempty_line_count": len(u_nonempty),
        "line_count_ratio": round(ratio, 6),
        "line_similarity": round(line_similarity, 6),
        "structure_preserved": bool(preserved),
    }


def _token_overlap_f1(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    m = token_level_metrics(a, b)
    return float(m["f1"])


def _norm_phrase(s: str) -> str:
    return compact_ws(s).lower()


def _build_joined_chunk_windows(
    pred_tokens: list[list[str]],
    max_join_chunks: int = 3,
) -> list[dict[str, object]]:
    if not pred_tokens:
        return []
    out: list[dict[str, object]] = []
    n = len(pred_tokens)
    for start in range(n):
        joined: list[str] = []
        for end in range(start, min(n, start + max_join_chunks)):
            joined.extend(pred_tokens[end])
            out.append(
                {
                    "start_chunk_index_0based": start,
                    "end_chunk_index_0based": end,
                    "joined_chunk_count": end - start + 1,
                    "tokens": list(joined),
                }
            )
    return out


def match_chunks_to_truth(
    predicted_chunks: list[dict[str, object]],
    gt_redactions: list[str],
    min_score: float = 0.35,
) -> dict[str, object]:
    gt_tokens = [tokenize_words(x) for x in gt_redactions]
    pred_tokens = [list(chunk.get("tokens", [])) for chunk in predicted_chunks]

    candidates: list[tuple[float, int, int]] = []
    for pi, pt in enumerate(pred_tokens):
        for gi, gt in enumerate(gt_tokens):
            score = _token_overlap_f1(pt, gt)
            if score >= min_score:
                candidates.append((score, pi, gi))

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    matched: list[dict[str, object]] = []

    for score, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        pred_text = str(predicted_chunks[pi].get("text", ""))
        gt_text = gt_redactions[gi]
        pair_metrics = token_level_metrics(pred_tokens[pi], gt_tokens[gi])
        matched.append(
            {
                "pred_chunk_id": int(predicted_chunks[pi].get("chunk_id", pi + 1)),
                "pred_index_0based": pi,
                "gt_index_0based": gi,
                "score_f1": round(float(score), 6),
                "exact_text_match": _norm_phrase(pred_text) == _norm_phrase(gt_text),
                "pred_text": pred_text,
                "gt_text": gt_text,
                "word_metrics": {
                    k: v
                    for k, v in pair_metrics.items()
                    if k != "mapping_ref_to_hyp"
                },
            }
        )

    correct = len(matched)
    pred_count = len(predicted_chunks)
    gt_count = len(gt_redactions)
    hallucinated = pred_count - correct
    missed = gt_count - correct
    perfect = sum(1 for x in matched if bool(x.get("exact_text_match")))

    matched_gt_words = sum(len(gt_tokens[x["gt_index_0based"]]) for x in matched)
    matched_pred_words = sum(len(pred_tokens[x["pred_index_0based"]]) for x in matched)
    matched_word_hits = sum(int(x["word_metrics"]["matched_tokens"]) for x in matched)

    matched_word_recall = matched_word_hits / matched_gt_words if matched_gt_words else 1.0
    matched_word_precision = matched_word_hits / matched_pred_words if matched_pred_words else 1.0

    # Relaxed matching quality: for each GT redaction, find the best-scoring
    # predicted span among single and short joined adjacent chunks.
    # This is robust to over-segmentation and numbering mismatches.
    windows = _build_joined_chunk_windows(pred_tokens, max_join_chunks=3)
    best_per_gt: list[dict[str, object]] = []
    for gi, gt in enumerate(gt_tokens):
        gt_text = gt_redactions[gi]
        best_score = -1.0
        best_window: dict[str, object] | None = None
        best_metrics: dict[str, object] | None = None
        for w in windows:
            w_tokens = list(w["tokens"])
            score = _token_overlap_f1(w_tokens, gt)
            if score > best_score:
                best_score = score
                best_window = w
                best_metrics = token_level_metrics(gt, w_tokens)
        if best_window is None or best_metrics is None:
            best_score = 0.0
            best_window = {
                "start_chunk_index_0based": -1,
                "end_chunk_index_0based": -1,
                "joined_chunk_count": 0,
                "tokens": [],
            }
            best_metrics = token_level_metrics(gt, [])

        pred_text = " ".join(list(best_window["tokens"]))
        best_per_gt.append(
            {
                "gt_index_0based": gi,
                "gt_text": gt_text,
                "best_score_f1": round(float(max(0.0, best_score)), 6),
                "best_chunk_start_0based": int(best_window["start_chunk_index_0based"]),
                "best_chunk_end_0based": int(best_window["end_chunk_index_0based"]),
                "best_joined_chunk_count": int(best_window["joined_chunk_count"]),
                "best_pred_text": pred_text,
                "best_exact_text_match": _norm_phrase(pred_text) == _norm_phrase(gt_text),
                "best_word_metrics": {
                    k: v
                    for k, v in best_metrics.items()
                    if k != "mapping_ref_to_hyp"
                },
            }
        )

    strict_avg_pair_f1 = (
        sum(float(x["score_f1"]) for x in matched) / len(matched)
        if matched
        else 0.0
    )
    relaxed_found_count = sum(1 for x in best_per_gt if float(x["best_score_f1"]) >= min_score)
    relaxed_avg_best_f1 = (
        sum(float(x["best_score_f1"]) for x in best_per_gt) / len(best_per_gt)
        if best_per_gt
        else 0.0
    )
    relaxed_exact_count = sum(1 for x in best_per_gt if bool(x["best_exact_text_match"]))

    return {
        "truth_redaction_count": gt_count,
        "predicted_redaction_count": pred_count,
        "correctly_found_redactions": correct,
        "missed_redactions": missed,
        "hallucinated_redactions": hallucinated,
        "perfect_text_match_count": perfect,
        "perfect_text_match_rate_on_correct": round((perfect / correct) if correct else 0.0, 6),
        "strict_avg_pair_f1_on_correct": round(strict_avg_pair_f1, 6),
        "matched_word_recall": round(matched_word_recall, 6),
        "matched_word_precision": round(matched_word_precision, 6),
        "relaxed_gt_found_count": relaxed_found_count,
        "relaxed_gt_found_rate": round((relaxed_found_count / gt_count) if gt_count else 1.0, 6),
        "relaxed_avg_best_f1_over_gt": round(relaxed_avg_best_f1, 6),
        "relaxed_exact_text_match_count": relaxed_exact_count,
        "relaxed_exact_text_match_rate": round((relaxed_exact_count / gt_count) if gt_count else 1.0, 6),
        "matched_pairs": matched,
        "best_match_per_gt": best_per_gt,
    }


def apply_redactions_to_context(context: str, redactions: list[str]) -> str:
    def repl(m: re.Match[str]) -> str:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(redactions):
            return redactions[idx]
        return m.group(0)

    return REDACTION_TAG_RE.sub(repl, context)


def summarize_mapping(
    unredacted_tokens: list[str],
    mapping_unred_to_red: list[int],
    redacted_candidates: dict[str, list[int]],
    candidate_cap: int = 32,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, tok in enumerate(unredacted_tokens):
        cands = redacted_candidates.get(tok, [])
        rows.append(
            {
                "unredacted_index_0based": idx,
                "token": tok,
                "mapped_redacted_index_0based": int(mapping_unred_to_red[idx]),
                "candidate_redacted_positions_0based": cands[:candidate_cap],
                "candidate_count": len(cands),
            }
        )
    return rows


def annotate_text_with_redaction_mask(
    raw_text: str,
    token_mask: list[int],
    label_prefix: str = "REDACTION",
) -> str:
    # Build markup directly on the original OCR text (preserve formatting).
    token_spans = list(WORD_RE.finditer(raw_text))
    if not token_spans or not token_mask:
        return raw_text

    n = min(len(token_spans), len(token_mask))
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if token_mask[i] != 1:
            i += 1
            continue
        j = i
        while j + 1 < n and token_mask[j + 1] == 1:
            j += 1
        runs.append((i, j))
        i = j + 1

    if not runs:
        return raw_text

    out: list[str] = []
    cursor = 0
    for rid, (s, e) in enumerate(runs, start=1):
        start_char = token_spans[s].start()
        end_char = token_spans[e].end()
        out.append(raw_text[cursor:start_char])
        out.append(f"[[{label_prefix}_{rid}]]")
        out.append(raw_text[start_char:end_char])
        out.append(f"[[/{label_prefix}_{rid}]]")
        cursor = end_char
    out.append(raw_text[cursor:])
    return "".join(out)
