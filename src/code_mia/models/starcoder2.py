from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    device: torch.device
    dtype_name: str
    quantization: str


def _source(config: dict) -> str:
    local = config.get("local_path")
    if local:
        path = Path(local)
        if not path.exists() or not (path / "config.json").exists():
            raise FileNotFoundError(f"Model local_path lacks config.json: {path}")
        return str(path)
    return config["id"]


def _common_kwargs(config: dict, revision_key: str = "revision") -> dict:
    return {
        "revision": None if config.get("local_path") else config[revision_key],
        "local_files_only": bool(config.get("offline", False)),
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
    }


def load_tokenizer(config: dict):
    from transformers import AutoTokenizer
    source = config.get("local_path") or config.get("tokenizer_id", config["id"])
    kwargs = _common_kwargs(config, "tokenizer_revision")
    try:
        return AutoTokenizer.from_pretrained(source, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load tokenizer {source} at pinned revision {config.get('tokenizer_revision')}. "
            "No latest-revision fallback was used. Check internet/cache or model.local_path. "
            f"Original error: {exc}"
        ) from exc


def choose_dtype(requested: str = "auto") -> tuple[torch.dtype, str]:
    if requested not in {"auto", "float16", "bfloat16", "float32"}:
        raise ValueError(f"Unsupported dtype: {requested}")
    if requested == "float32":
        return torch.float32, "float32"
    if requested == "bfloat16":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 was requested but the GPU does not genuinely support it")
        return torch.bfloat16, "bfloat16"
    if requested == "float16":
        return torch.float16, "float16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bfloat16"
    if torch.cuda.is_available():
        return torch.float16, "float16"
    return torch.float32, "float32"


def load_model_and_tokenizer(config: dict) -> LoadedModel:
    from transformers import AutoModelForCausalLM
    dtype, dtype_name = choose_dtype(config.get("dtype", "auto"))
    quantization = config.get("quantization", "none")
    kwargs = _common_kwargs(config)
    kwargs["torch_dtype"] = dtype
    if quantization != "none":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=quantization == "8bit", load_in_4bit=quantization == "4bit"
        )
        kwargs["device_map"] = {"": 0}
    source = _source(config)
    try:
        model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model {source} at pinned revision {config['revision']}. "
            "No latest-revision fallback was used. Check internet/cache and available VRAM. "
            f"Original error: {exc}"
        ) from exc
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if quantization != "none" and device.type != "cuda":
        raise RuntimeError("bitsandbytes quantization requires CUDA in this baseline")
    if quantization == "none":
        model.to(device)
    model.eval()
    return LoadedModel(model, load_tokenizer(config), device, dtype_name, quantization)
