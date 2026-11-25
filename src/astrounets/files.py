"""Safe writes: every result is written to a temporary file next to its target and renamed when
complete, so a job that is killed half-way never leaves a truncated file behind."""
from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def atomic(path: str | Path) -> Iterator[Path]:
    """Yield a temporary path with the same suffix; rename it to `path` if the block succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex[:8]}.tmp{path.suffix}")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_text(path: str | Path, text: str) -> Path:
    with atomic(path) as tmp:
        tmp.write_text(text, encoding="utf-8")
    return Path(path)


def write_json(path: str | Path, obj) -> Path:
    return write_text(path, json.dumps(obj, indent=2))


def write_parquet(table, path: str | Path) -> Path:
    with atomic(path) as tmp:
        table.to_parquet(tmp, index=False)
    return Path(path)


def write_csv(table, path: str | Path) -> Path:
    with atomic(path) as tmp:
        table.to_csv(tmp, index=False)
    return Path(path)
