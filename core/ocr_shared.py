from pathlib import Path
import re
import io
import contextlib
import tempfile
import warnings
from dataclasses import dataclass
from typing import Any

from PIL import Image
try:
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None
    SamplingParams = None
try:
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
except Exception:
    NGramPerReqLogitsProcessor = None

MODEL_ID = "deepseek-ai/DeepSeek-OCR"
PROMPT_FREE_OCR = "<image>\nFree OCR. Preserve original line breaks."
EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class HFOCREngine:
    model_id: str
    tokenizer: Any
    model: Any
    attn_implementation: str = "eager"
    base_size: int = 1024
    image_size: int = 768
    crop_mode: bool = True


def clean_ds_output(raw) -> str:
    if raw is None:
        return ""
    s = str(raw)
    s = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", s, flags=re.DOTALL)
    s = re.sub(r"<\|det\|>.*?<\|/det\|>", "", s, flags=re.DOTALL)
    return s.strip()


def _build_hf_engine(
    model_id: str,
    hf_attn_implementation: str = "eager",
    hf_dtype: str = "bfloat16",
) -> HFOCREngine:
    import torch
    from transformers import AutoModel, AutoTokenizer
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    dtype_name = str(hf_dtype).lower().strip()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype_name, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_safetensors=True,
        attn_implementation=hf_attn_implementation,
        torch_dtype=torch_dtype,
    ).eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return HFOCREngine(
        model_id=model_id,
        tokenizer=tokenizer,
        model=model,
        attn_implementation=hf_attn_implementation,
    )


def build_llm(
    model_id: str,
    max_model_len: int | None = None,
    ocr_backend: str = "auto",
    hf_attn_implementation: str = "eager",
    hf_dtype: str = "bfloat16",
):
    backend = str(ocr_backend).lower().strip()
    if backend not in {"auto", "vllm", "hf"}:
        raise ValueError(f"Unsupported ocr_backend: {ocr_backend}")

    if backend == "auto":
        backend = "hf" if "deepseek-ocr-2" in model_id.lower() else "vllm"

    if backend == "hf":
        return _build_hf_engine(
            model_id=model_id,
            hf_attn_implementation=hf_attn_implementation,
            hf_dtype=hf_dtype,
        )

    if LLM is None:
        raise RuntimeError("vLLM is not available in this environment. Use --ocr_backend hf.")

    kwargs = dict(
        model=model_id,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
    )
    # v1 OCR no-repeat processor is optional; OCR-2 may not expose this class.
    if NGramPerReqLogitsProcessor is not None and "deepseek-ocr-2" not in model_id.lower():
        kwargs["logits_processors"] = [NGramPerReqLogitsProcessor]
    # OCR-2 HF usage in upstream examples requires remote code; this is harmless for v1.
    if "deepseek-ocr-2" in model_id.lower():
        kwargs["trust_remote_code"] = True
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    return LLM(**kwargs)


def _extract_hf_infer_text(stdout_text: str) -> str:
    m = re.search(r"=+\s*\n(.*?)\n=+save results:", stdout_text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return clean_ds_output(m.group(1))

    cleaned: list[str] = []
    for raw_line in stdout_text.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned.append("")
            continue
        low = line.lower()
        if line.startswith("="):
            continue
        if low.startswith("base:") or low == "no patches":
            continue
        if low.startswith("image:") or low.startswith("other:"):
            continue
        if "save results" in low:
            continue
        cleaned.append(raw_line.rstrip())
    return clean_ds_output("\n".join(cleaned))


def _ocr_hf_path(engine: HFOCREngine, image_path: Path, prompt: str) -> str:
    if not hasattr(engine.model, "infer"):
        raise RuntimeError("HF OCR backend loaded model without infer(...).")

    buf = io.StringIO()
    with tempfile.TemporaryDirectory(prefix="ocr2_infer_") as td:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*do_sample.*temperature.*")
            warnings.filterwarnings("ignore", message=r".*attention mask and the pad token id.*")
            warnings.filterwarnings("ignore", message=r".*Setting `pad_token_id`.*")
            warnings.filterwarnings("ignore", message=r".*attention mask is not set.*")
            warnings.filterwarnings("ignore", message=r".*seen_tokens.*deprecated.*")
            warnings.filterwarnings("ignore", message=r".*get_max_cache\(\).*deprecated.*")
            with contextlib.redirect_stdout(buf):
                engine.model.infer(
                    engine.tokenizer,
                    prompt=prompt,
                    image_file=str(image_path),
                    output_path=td,
                    base_size=int(engine.base_size),
                    image_size=int(engine.image_size),
                    crop_mode=bool(engine.crop_mode),
                    save_results=False,
                )
    return _extract_hf_infer_text(buf.getvalue())


def ocr_with_model(
    llm,
    image_path: Path,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    ngram_size: int = 30,
    window_size: int = 90,
) -> str:
    if isinstance(llm, HFOCREngine):
        return _ocr_hf_path(llm, image_path, prompt=prompt)

    img = Image.open(image_path).convert("RGB")
    return ocr_with_pil(
        llm=llm,
        image=img,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        ngram_size=ngram_size,
        window_size=window_size,
    )


def ocr_with_pil(
    llm,
    image: Image.Image,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    ngram_size: int = 30,
    window_size: int = 90,
) -> str:
    if isinstance(llm, HFOCREngine):
        with tempfile.TemporaryDirectory(prefix="ocr2_pil_") as td:
            tmp = Path(td) / "input.png"
            image.convert("RGB").save(tmp)
            return _ocr_hf_path(llm, tmp, prompt=prompt)

    if SamplingParams is None:
        raise RuntimeError("vLLM SamplingParams is unavailable in this environment.")

    img = image.convert("RGB")
    model_input = [{
        "prompt": prompt,
        "multi_modal_data": {"image": img},
    }]
    sampling_kwargs = dict(
        temperature=temperature,
        max_tokens=max_tokens,
        skip_special_tokens=False,
    )
    # Keep v1 anti-repeat knobs when available. Retry without model-specific extras
    # for compatibility with OCR-2 and different vLLM builds.
    try:
        sampling = SamplingParams(
            **sampling_kwargs,
            extra_args=dict(
                ngram_size=ngram_size,
                window_size=window_size,
                whitelist_token_ids={128821, 128822},
            ),
        )
        out = llm.generate(model_input, sampling)
    except Exception:
        sampling = SamplingParams(**sampling_kwargs)
        out = llm.generate(model_input, sampling)
    return clean_ds_output(out[0].outputs[0].text)


def collect_images(in_path: Path) -> list[Path]:
    if in_path.is_file() and in_path.suffix.lower() in EXTS:
        return [in_path]
    if in_path.is_dir():
        return [p for p in sorted(in_path.iterdir()) if p.is_file() and p.suffix.lower() in EXTS]
    raise SystemExit(f"Input not found or not an image/folder: {in_path}")
