from __future__ import annotations

from typing import Any


def load_public_test(config: dict[str, Any]):
    try:
        from datasets import DownloadConfig, load_dataset, load_from_disk
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package before loading Poisoned Chalice") from exc

    local_path = config.get("local_path")
    offline = bool(config.get("offline", False))
    try:
        if local_path:
            loaded = load_from_disk(str(local_path))
            return loaded[config["split"]] if hasattr(loaded, "keys") and config["split"] in loaded else loaded
        return load_dataset(
            config["id"],
            config["config"],
            split=config["split"],
            revision=config["revision"],
            download_mode="reuse_dataset_if_exists",
            download_config=DownloadConfig(local_files_only=offline),
        )
    except Exception as exc:
        mode = "offline local snapshot" if offline else "pinned Hugging Face revision"
        raise RuntimeError(
            f"Could not load {mode} for dataset {config['id']} config={config['config']} "
            f"split={config['split']} revision={config['revision']}. No latest-revision fallback was used. "
            "Enable internet or set dataset.local_path to a valid datasets.save_to_disk snapshot. "
            f"Original error: {exc}"
        ) from exc


from typing import Any


def membership_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)

    if value in {0, 1}:
        return int(value)

    if isinstance(value, str):
        normalized = value.strip().lower()

        mapping = {
            "0": 0,
            "1": 1,
            "member": 1,
            "non-member": 0,
        }

        if normalized in mapping:
            return mapping[normalized]

    allowed = (
        "0, 1, False, True, '0', '1', "
        "'member', or 'non-member'"
    )

    raise ValueError(
        f"membership must be one of {allowed}; got {value!r}"
    )