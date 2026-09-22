"""Dependency-light episode recorder for synchronized Quest/action/robot data."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


class EpisodeRecorder:
    def __init__(self, output_dir: str | Path, metadata: dict[str, Any]) -> None:
        self.output_dir = Path(output_dir)
        self.metadata = dict(metadata)
        self.active = False
        self._rows: list[dict[str, Any]] = []

    def start(self) -> None:
        self._rows.clear()
        self.active = True

    def append(self, **row: Any) -> None:
        if self.active:
            self._rows.append(row)

    def stop_and_save(self) -> Path | None:
        if not self.active:
            return None
        self.active = False
        if not self._rows:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        base = self.output_dir / f"episode_{stamp}"
        keys = self._rows[0].keys()
        arrays = {key: np.asarray([row[key] for row in self._rows]) for key in keys}
        np.savez_compressed(base.with_suffix(".npz"), **arrays)
        metadata = {
            **self.metadata,
            "created_at_utc": stamp,
            "samples": len(self._rows),
            "fields": {key: list(value.shape) for key, value in arrays.items()},
        }
        base.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._rows.clear()
        return base.with_suffix(".npz")

