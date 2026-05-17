"""Additional branch coverage for mast.main orchestration."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import mast


@pytest.mark.unit
def test_mast_main_resolve_urls_without_metadata_csv_returns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = {
        'mission': 'HST',
        'filters': {'exp': {'column': 'exp', 'conditions': {'greater': 1}}},
        'main_column': 'exp',
        'max_workers': 1,
        'prefer_token': 'sci',
        'chunk_size': 10,
        'save_dir': str(tmp_path / 'downloads'),
        'download': False,
        'max_requests': 2,
        'reset_after': 10,
        'resolve_max_retries': 1,
        'resolve_retry_delay': 0,
        'metadata_output': str(tmp_path / 'missing.csv'),
        'id_column': 'obsid',
        'url_column': 'url',
        'fetch_metadata': False,
        'resolve_urls': True,
    }

    monkeypatch.setattr(mast, 'parse_config_overrides', lambda *args, **kwargs: {})
    monkeypatch.setattr(mast, 'load_config', lambda **kwargs: {'mast': cfg})

    mast.main()
