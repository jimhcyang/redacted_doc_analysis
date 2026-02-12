from pathlib import Path
import re

from PIL import Image
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

MODEL_ID = "deepseek-ai/DeepSeek-OCR"
PROMPT_FREE_OCR = "<image>\nFree OCR. Preserve original line breaks."
EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def clean_ds_output(raw) -> str:
    if raw is None:
        return ""
    s = str(raw)
    s = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", s, flags=re.DOTALL)
    s = re.sub(r"<\|det\|>.*?<\|/det\|>", "", s, flags=re.DOTALL)
    return s.strip()


def build_llm(model_id: str, max_model_len: int | None = None) -> LLM:
    kwargs = dict(
        model=model_id,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
    )
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    return LLM(**kwargs)


def ocr_with_model(
    llm: LLM,
    image_path: Path,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    ngram_size: int = 30,
    window_size: int = 90,
) -> str:
    img = Image.open(image_path).convert("RGB")
    model_input = [{
        "prompt": prompt,
        "multi_modal_data": {"image": img},
    }]
    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        extra_args=dict(
            ngram_size=ngram_size,
            window_size=window_size,
            whitelist_token_ids={128821, 128822},
        ),
        skip_special_tokens=False,
    )
    out = llm.generate(model_input, sampling)
    return clean_ds_output(out[0].outputs[0].text)


def collect_images(in_path: Path) -> list[Path]:
    if in_path.is_file() and in_path.suffix.lower() in EXTS:
        return [in_path]
    if in_path.is_dir():
        return [p for p in sorted(in_path.iterdir()) if p.is_file() and p.suffix.lower() in EXTS]
    raise SystemExit(f"Input not found or not an image/folder: {in_path}")

