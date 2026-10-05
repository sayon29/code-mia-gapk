from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from .utils.hashing import fingerprint

ENV_PATHS = {
    ("experiment", "output_dir"): "CODE_MIA_OUTPUT_DIR",
    ("model", "local_path"): "CODE_MIA_MODEL_PATH",
    ("dataset", "local_path"): "CODE_MIA_DATASET_PATH",
}


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    config = copy.deepcopy(config)
    for keys, env_name in ENV_PATHS.items():
        if os.getenv(env_name) and not config.get(keys[0], {}).get(keys[1]):
            config.setdefault(keys[0], {})[keys[1]] = os.environ[env_name]
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        cursor = config
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    validate_config(config)
    config["implementation"] = {"name": "code-mia-gapk", "version": "0.1.0"}
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = ["experiment", "dataset", "canonicalization", "model", "attack", "runtime"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    if config["dataset"]["config"].lower() != "python":
        raise ValueError("This baseline supports only the Python dataset configuration")
    if config["dataset"]["split"] != "test":
        raise ValueError("This baseline supports only the public test split")
    if config["attack"]["batch_size"] != 1:
        raise ValueError("Exact memory-safe baseline requires batch_size=1")
    if not 0 < float(config["attack"]["gap_fraction"]) <= 1:
        raise ValueError("gap_fraction must lie in (0, 1]")
    if int(config["attack"]["smoothing_window"]) < 1:
        raise ValueError("smoothing_window must be positive")
    if config["model"].get("quantization", "none") not in {"none", "8bit", "4bit"}:
        raise ValueError("quantization must be one of: none, 8bit, 4bit")


def config_fingerprint(config: dict[str, Any]) -> str:
    material = copy.deepcopy(config)
    material.get("experiment", {}).pop("output_dir", None)
    return fingerprint(material)


def dump_yaml(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

