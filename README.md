# DeepSeek OCR Readflow Pipeline

This repo is now organized around one production entrypoint:

- `scripts/run_readflow.py`

The old placeholder-only pipeline was removed to keep one clear workflow.

## What this does

Given one image or a folder of images, it:

1. Runs OCR and CV preprocessing.
2. Detects hollow redaction boxes.
3. Builds a percentile-based text region.
4. Detects row-scan line/gap bands and runs OCR stripe-by-stripe.
5. Projects redactions onto line/character spans.
6. Writes:
   - raw OCR text
   - OCR + `[REDACTED_n]` placement
   - layout projection (`[BOXn____]` or pixel form)
   - analysis JSON
   - ordered debug images (if `--debug`)

Input behavior:

- File input: processes one supported image.
- Folder input: processes all supported images in that folder (sorted, non-recursive).
- Supported extensions: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff`.

## Project layout

```text
core/
  __init__.py
  ocr_shared.py             # vLLM + OCR shared helpers
  redaction_detection.py    # redaction detector + debug 01-05 images
  readflow_pipeline.py      # main pipeline logic
scripts/
  run_readflow.py           # main runnable entrypoint
outputs/                    # generated files
0_redacted.png              # sample input
```

## Terminal commands

Run these from the repo root.

### 1) (Optional) Create and activate environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

Use your existing environment if already working. Typical install:

```bash
pip install vllm opencv-python pillow numpy
```

### 3) Single image run

```bash
python scripts/run_readflow.py --input 0_redacted.png --out outputs --debug
```

### 4) Batch folder run (e.g., 60 PNGs)

```bash
python scripts/run_readflow.py --input /path/to/image_folder --out outputs_batch --debug
```

### 5) Layout token style

Underscore mode (default):

```bash
python scripts/run_readflow.py --input 0_redacted.png --box_label_mode underscore
```

Pixel mode:

```bash
python scripts/run_readflow.py --input 0_redacted.png --box_label_mode pixel
```

### 6) Useful knobs

```bash
python scripts/run_readflow.py \
  --input 0_redacted.png \
  --percentile_q 0.5 \
  --first_line_indent_spaces 5 \
  --box_line_chars 64 \
  --debug
```

## Main CLI options

- `--input`: image file or folder
- `--out`: output folder
- `--model`: model id/path (default `deepseek-ai/DeepSeek-OCR`)
- `--prompt`: OCR prompt
- `--max_tokens`: generation length
- `--temperature`: sampling temperature
- `--ngram_size`, `--window_size`: anti-repeat settings
- `--max_model_len`: vLLM max model length
- `--percentile_q`: percentile for text-region border estimation
- `--first_line_indent_spaces`: x-mapping slack for line 1
- `--box_label_mode`: `underscore` or `pixel`
- `--box_line_chars`: normalized line budget for underscore sizing
- `--debug`: save debug images

## Outputs per input image

For image `foo.png`, output files are:

- `outputs/foo.raw_ocr.txt`
- `outputs/foo.readflow_with_placeholders.txt`
- `outputs/foo.layout_projection.txt`
- `outputs/foo.analysis.json`

Debug folder:

- `outputs/debug/foo/01_dark_mask.png`
- `outputs/debug/foo/02_lines_mask.png`
- `outputs/debug/foo/05_redaction_boxes.png`
- `outputs/debug/foo/12_structural_lines_mask.png`
- `outputs/debug/foo/14_percentile_region.png`
- `outputs/debug/foo/15_line_model_overlay.png`
- `outputs/debug/foo/16_linebreak_overlay.png`
- `outputs/debug/foo/17_text_pixels_mask.png`

Numbering is intentional so you can inspect in order.

Flow:

1. `01` + `02` + `12` -> `17` where `text_pixels = 01 - 02 - 12`.
2. `17` -> row-scan bands and percentile region (`14`).
3. Bands + redactions -> classified `linebreak` vs `blankline`.
4. Classified bands + line OCR + redactions -> overlay outputs (`15`, `16`) and final txt/json.

## Notes

- Redaction-to-line mapping uses a strict vertical coverage threshold (`>50%`).
- If a redaction crosses a line's right border, token placement is forced to end-of-line.
- If a redaction crosses left border, placement is forced to start-of-line.
- Width/underscore sizing uses the clipped in-line span only (outside-region overflow is ignored).
- Line structure is recovered from row-scan line/gap bands and OCR is run stripe-by-stripe.
- Row-scan uses `text pixels = dark mask (01) minus lines mask (02) minus structural lines mask (12)` and marks linebreak rows when a strip is at least 95% black.
- A gap is `blankline` when its height is more than 50% of average text-line height; otherwise it is `linebreak`.
- New-paragraph insertion is suppressed when the paragraph probe box intersects redaction boxes by more than 50% of probe-box area.
- Overlay colors: `blue = linebreak (small gap)`, `yellow = blankline-sized gap (>50% avg line height)`.

## Week 1 Summary: Document-Difference Approach

This week we built and tested a second workflow focused on paired documents (redacted vs unredacted) instead of single-image reconstruction. The objective is to recover newly unredacted text by comparing the two OCR outputs at token level, then producing inspectable artifacts for each page and each full document. The pipeline is now end-to-end in this repo: sample pairs, materialize local test sets, run OCR + diff, and write both page-level and document-level diagnostics.

The core method is monotonic alignment between unredacted and redacted token streams. Tokens present only in the unredacted stream become candidate redaction chunks. We then attach local context on both sides (anchors before/after the chunk in each version) so each candidate can be manually verified. This is the basis for `*.redaction_chunks.txt` and `*.unredacted_bracketed.txt`, and now also for per-document full aggregates in `docs_diagnostics/by_document/`.

A major practical finding is that the diff logic itself is relatively stable, while OCR fidelity is the dominant source of downstream error. When both page transcriptions are clean, the extracted chunks are usually correct and contextually located. When OCR drops words, alters spelling, or collapses structure, the aligner can overproduce chunk candidates because it treats transcription drift as semantic difference. In short: difference extraction quality is conditionally high, but strongly depends on OCR quality of both inputs.

To improve this, the PDF workflow now runs multi-candidate page OCR and chooses the best transcript per page: rendered-page OCR, preprocessed-page OCR (crop + autocontrast + upscale), and PDF text-layer fallback when it passes quality gates. This reduced obvious noise (including engine artifacts and unusable text-layer outputs) and improved consistency on sparse pages, but not enough to remove all OCR brittleness. Short text regions and redaction-adjacent lines remain the most error-prone.

From earlier pixel-math experiments, geometry remains a strong signal. Box/line/paragraph detection and debug overlays are highly inspectable and generally robust on clean, straight scans. However, geometry does not solve lexical transcription failures. The current direction combines both strengths: geometry-aware preprocessing and diagnostics, with token-level difference extraction as the recovery mechanism.

Evaluation status this week: detailed manual inspection was done on representative examples, and full pipeline runs were completed on larger batches (including all 62 examples in the earlier image workflow and sampled PDF-pair runs in the new document workflow). Full manual audit of every output is still pending. Current interpretation is straightforward: structural math is strong; OCR remains the bottleneck; document-difference extraction is useful and interpretable when OCR quality is acceptable.

### Scripts for the document-difference workflow

- `scripts/run_docs_diagnostics_pdf.py`
  - Run paired-PDF diagnostics for an existing `docs_example/` folder.
- `scripts/run_sampled_docs_diagnostics.py`
  - Sample `N` rows from a full source corpus (default `docs/`), populate a target docs folder, run diagnostics, and build per-document aggregates.

Example full run (100 sampled pairs):

```bash
/home/jimyh/miniforge3/envs/deepseek-ocr2/bin/python scripts/run_sampled_docs_diagnostics.py \
  --source_docs_root docs \
  --target_docs_root docs_example \
  --out docs_diagnostics \
  --sample_size 100 \
  --seed 42 \
  --model deepseek-ai/DeepSeek-OCR-2 \
  --ocr_backend hf \
  --hf_attn_implementation eager \
  --hf_dtype bfloat16
```

### Suggested figures for writeup/slides

1. `outputs/test_ocr2/debug/0_redacted/05_redaction_boxes.png`
2. `outputs/test_ocr2/debug/3_redacted/15_line_model_overlay.png`
3. `outputs/test_ocr2/debug/8_redacted/16_linebreak_overlay.png`
4. `outputs/test_ocr2/debug/8_redacted/18_serialized_blanklines_overlay.png`
5. `outputs/test_ocr2/debug/3_redacted/19_paragraph_blocks_overlay.png`
6. Side-by-side: `outputs/test_ocr2/8_redacted.layout_projection.txt` vs `outputs/test_ocr2/8_redacted.paragraph_ocr.txt`
