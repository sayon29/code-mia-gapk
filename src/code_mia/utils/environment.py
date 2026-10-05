from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import socket
import sys
from pathlib import Path

import psutil
import torch


def _package_versions() -> dict[str, str | None]:
    names = ["torch", "transformers", "datasets", "huggingface-hub", "nltk", "numpy", "scikit-learn", "matplotlib"]
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _internet_available() -> bool:
    try:
        socket.getaddrinfo("huggingface.co", 443)
        return True
    except OSError:
        return False


def collect_environment(model_id: str | None = None, dataset_id: str | None = None) -> dict:
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            try:
                free, total = torch.cuda.mem_get_info(index)
            except Exception:
                free, total = None, props.total_memory
            gpus.append({
                "index": index, "name": props.name,
                "vram_bytes": int(props.total_memory),
                "free_vram_bytes": int(free) if free is not None else None,
                "total_visible_vram_bytes": int(total),
            })
    disk_base = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
    disk = shutil.disk_usage(disk_base)
    cache_paths = {
        "HF_HOME": os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
        "HUGGINGFACE_HUB_CACHE": os.getenv("HUGGINGFACE_HUB_CACHE"),
        "HF_DATASETS_CACHE": os.getenv("HF_DATASETS_CACHE"),
    }
    return {
        "platform": platform.platform(), "operating_system": platform.system(),
        "python_version": sys.version, "package_versions": _package_versions(),
        "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": gpus,
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "system_ram_total_bytes": int(psutil.virtual_memory().total),
        "system_ram_available_bytes": int(psutil.virtual_memory().available),
        "disk_path": str(disk_base), "disk_free_bytes": int(disk.free),
        "huggingface_cache_paths": cache_paths,
        "internet_dns_available": _internet_available(),
        "model_snapshot_hint": _snapshot_hint(cache_paths, model_id, "models") if model_id else None,
        "dataset_snapshot_hint": _snapshot_hint(cache_paths, dataset_id, "datasets") if dataset_id else None,
    }


def _snapshot_hint(cache_paths: dict, repo_id: str | None, kind: str) -> dict | None:
    if not repo_id:
        return None
    root_value = cache_paths.get("HUGGINGFACE_HUB_CACHE") or str(Path(cache_paths["HF_HOME"]) / "hub")
    path = Path(root_value) / f"{kind}--{repo_id.replace('/', '--')}"
    return {"path": str(path), "exists": path.exists()}


def print_environment(report: dict) -> None:
    print(json.dumps(report, indent=2))


def estimate_memory(config: dict, token_count: int = 64, vocab_size: int = 49152) -> dict:
    dtype_name = config["model"].get("dtype", "auto")
    from ..models.starcoder2 import choose_dtype
    _, selected = choose_dtype(dtype_name)
    model_bytes = 3_000_000_000 * ({"float16": 2, "bfloat16": 2, "float32": 4}[selected])
    logits_bytes = token_count * vocab_size * 4
    free = None
    if torch.cuda.is_available():
        free = int(torch.cuda.mem_get_info(0)[0])
    return {
        "model_loading_dtype": selected,
        "available_gpu_vram_bytes": free,
        "expected_sequence_length": token_count,
        "assumed_vocab_size": vocab_size,
        "expected_one_float32_full_logit_tensor_bytes": logits_bytes,
        "approximate_model_weight_bytes": model_bytes,
        "likely_to_fit": None if free is None else bool(free > model_bytes + 4 * logits_bytes + 1_500_000_000),
        "note": "Fit estimate includes a conservative 1.5 GB runtime margin; actual peaks depend on architecture and allocator state.",
    }

