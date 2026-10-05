from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Iterable


def build_audit(records: Iterable[dict], exclusions: Iterable[dict]) -> dict:
    rows = list(records)
    excluded = list(exclusions)
    hashes = Counter(row["canonical_text_sha256"] for row in rows)
    raw_lengths = [row["raw_character_count"] for row in rows]
    token_lengths = [row["starcoder_token_count"] for row in rows]
    return {
        "selected_count": len(rows),
        "class_counts": dict(Counter(str(row["membership"]) for row in rows)),
        "exclusion_count": len(excluded),
        "exclusion_reasons": dict(Counter(row["exclusion_reason"] for row in excluded)),
        "raw_character_count": _summary(raw_lengths),
        "starcoder_token_count": _summary(token_lengths),
        "duplicate_canonical_hash_groups": sum(count > 1 for count in hashes.values()),
        "duplicate_canonical_hash_rows": sum(count for count in hashes.values() if count > 1),
    }


def _summary(values: list[int]) -> dict:
    return {"min": min(values), "max": max(values), "mean": mean(values)} if values else {}

