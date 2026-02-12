from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _norm_context_tokens(text: str) -> str:
    # Map dataset markers to the same token family used by OCR outputs.
    s = re.sub(r"<\s*redaction\s*(\d+)\s*>", r"[REDACTED_\1]", text, flags=re.IGNORECASE)
    return s.strip()


def _compact_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _numeric_prefix(name: str) -> int:
    m = re.match(r"^(\d+)", name)
    return int(m.group(1)) if m else 10**9


def _load_allowed_images(image_folder: Path | None) -> set[str] | None:
    if image_folder is None:
        return None
    if not image_folder.exists():
        raise SystemExit(f"Image folder not found: {image_folder}")
    allowed = set()
    for p in image_folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != ".png":
            continue
        allowed.add(p.name)
    return allowed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples_jsonl", required=True, help="Path to examples.jsonl")
    parser.add_argument("--image_folder", default=None, help="Optional folder of target redacted PNGs")
    parser.add_argument("--out_dir", default="outputs/compare", help="Output folder")
    parser.add_argument("--max_examples", type=int, default=None, help="Optional cap after filtering")
    args = parser.parse_args()

    examples_jsonl = Path(args.examples_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed_images = _load_allowed_images(Path(args.image_folder)) if args.image_folder else None

    rows: list[dict[str, object]] = []
    with examples_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            image_name = Path(item["redacted_path"]).name
            if allowed_images is not None and image_name not in allowed_images:
                continue
            context = str(item["context"])
            row = {
                "example_id": int(item["example_id"]),
                "image_name": image_name,
                "doc_id_r": item.get("doc_id_r"),
                "date": item.get("date"),
                "context": context,
                "context_with_tokens": _norm_context_tokens(context),
                "context_compact": _compact_ws(_norm_context_tokens(context)),
            }
            rows.append(row)

    rows.sort(key=lambda r: int(r["example_id"]))
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    jsonl_path = out_dir / "expected_contexts.jsonl"
    agg_path = out_dir / "expected_contexts.aggregate.txt"
    compact_path = out_dir / "expected_contexts.compact.tsv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with agg_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"### example_id={row['example_id']} image={row['image_name']}\n")
            f.write(str(row["context_with_tokens"]) + "\n\n")

    with compact_path.open("w", encoding="utf-8") as f:
        f.write("example_id\timage_name\tcontext_compact\n")
        for row in rows:
            f.write(f"{row['example_id']}\t{row['image_name']}\t{row['context_compact']}\n")

    print(f"[INFO] rows: {len(rows)}")
    print(f"[INFO] wrote: {jsonl_path}")
    print(f"[INFO] wrote: {agg_path}")
    print(f"[INFO] wrote: {compact_path}")


if __name__ == "__main__":
    main()

