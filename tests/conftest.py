"""Shared fixtures: configs pointing at temporary folders and one mock sky per test session."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from astrounets.config import load_config, replace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "smoke.yaml"
DEFAULT = ROOT / "configs" / "default.yaml"


def smoke_config(tmp: Path, **sections):
    cfg = load_config(SMOKE, {"paths": {"data": str(tmp / "data"), "runs": str(tmp / "runs")}})
    return replace(cfg, sections) if sections else cfg


@pytest.fixture
def cfg(tmp_path):
    return smoke_config(tmp_path)


@pytest.fixture(scope="session")
def mock_data(tmp_path_factory):
    """Mock MAST table + images + the 'mock' dataset, built once."""
    from astrounets.data.dataset import build_dataset
    from astrounets.data.mast import fetch

    tmp = tmp_path_factory.mktemp("mock")
    config = smoke_config(tmp, dataset={"workers": 1})
    fetch(config)
    build_dataset(config)
    return config
