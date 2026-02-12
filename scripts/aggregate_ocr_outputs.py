from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _compact_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_example_id(stem_name: str) -> int:
    m = re.match(r"^(\d+)", stem_name)
    return int(m.group(1)) if m else 10**9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", required=True, help="Folder containing per-image OCR txt files")
    parser.add_argument(
        "--suffix",
        default=".readflow_with_placeholders.txt",
        help="Target suffix to aggregate (default: readflow outputs)",
    )
    parser.add_argument("--out_dir", default="outputs/compare", help="Output folder")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    pattern = f"*{args.suffix}"
    for p in outputs_dir.glob(pattern):
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        image_stem = p.name[: -len(args.suffix)]
        image_name = f"{image_stem}.png"
        rows.append(
            {
                "example_id": _extract_example_id(image_stem),
                "image_name": image_name,
                "output_file": str(p),
                "text": text.strip(),
                "text_compact": _compact_ws(text),
            }
        )

    rows.sort(key=lambda r: (int(r["example_id"]), str(r["image_name"])))

    jsonl_path = out_dir / "ocr_outputs.jsonl"
    agg_path = out_dir / "ocr_outputs.aggregate.txt"
    compact_path = out_dir / "ocr_outputs.compact.tsv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with agg_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"### example_id={row['example_id']} image={row['image_name']}\n")
            f.write(str(row["text"]) + "\n\n")

    with compact_path.open("w", encoding="utf-8") as f:
        f.write("example_id\timage_name\ttext_compact\n")
        for row in rows:
            compact = str(row["text_compact"]).replace("\t", " ")
            f.write(f"{row['example_id']}\t{row['image_name']}\t{compact}\n")

    print(f"[INFO] rows: {len(rows)}")
    print(f"[INFO] wrote: {jsonl_path}")
    print(f"[INFO] wrote: {agg_path}")
    print(f"[INFO] wrote: {compact_path}")


if __name__ == "__main__":
    main()

