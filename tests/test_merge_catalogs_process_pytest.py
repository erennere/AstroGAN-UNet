"""Additional coverage for merge_catalogs process orchestration."""
from __future__ import annotations

import sys
import runpy
import concurrent.futures
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import merge_catalogs as merge_mod


class _FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _FakeExecutor:
    def __init__(self, max_workers=1):
        self.futures = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        future = _FakeFuture(fn(*args, **kwargs))
        self.futures.append(future)
        return future


class _RaisingFuture:
    def result(self):
        raise RuntimeError('chunk-failed')


@pytest.mark.unit
def test_process_merges_chunks_and_writes_parquet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    noise_csv = tmp_path / 'noise.csv'
    org_csv = tmp_path / 'org.csv'
    rec_csv = tmp_path / 'rec.csv'
    out_parquet = tmp_path / 'merged.parquet'

    base = pd.DataFrame(
        {
            'image_id': ['img1', 'img2'],
            'new_exp_time': [10.0, 10.0],
            'x': [1.0, 2.0],
            'y': [1.0, 2.0],
            'flux': [10.0, 20.0],
        }
    )
    base.to_csv(noise_csv, index=False)
    base.to_csv(org_csv, index=False)
    base.to_csv(rec_csv, index=False)

    fake_executor = _FakeExecutor()
    monkeypatch.setattr(merge_mod, 'ProcessPoolExecutor', lambda max_workers=1: fake_executor)
    monkeypatch.setattr(merge_mod, 'as_completed', lambda futures: futures)

    merge_mod.process(
        noise_csv=str(noise_csv),
        org_csv=str(org_csv),
        rec_csv=str(rec_csv),
        workers=2,
        output_parquet=str(out_parquet),
        threshold=2.0,
    )

    assert out_parquet.exists()
    merged = pd.read_parquet(out_parquet)
    assert 'image_id' in merged.columns


@pytest.mark.unit
def test_run_from_config_builds_paths_and_calls_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls = []

    monkeypatch.setattr(merge_mod, 'process', lambda **kwargs: calls.append(kwargs))

    eval_cfg = {
        'uncropped_output_dir': str(tmp_path),
        'uncropped_rec_catalog_csv': 'rec.csv',
        'uncropped_noisy_catalog_csv': 'noise.csv',
        'uncropped_org_catalog_csv': 'org.csv',
        'photometrical_data_filename': 'photo.parquet',
        'merge_catalog_workers': 2,
        'workers': 4,
        'merge_catalog_threshold': 3.5,
    }

    merge_mod._run_from_config(eval_cfg)

    assert len(calls) == 1
    payload = calls[0]
    assert payload['workers'] == 2
    assert payload['threshold'] == 3.5
    assert payload['output_parquet'].endswith('photo.parquet')


@pytest.mark.unit
def test_process_handles_future_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    noise_csv = tmp_path / 'noise.csv'
    org_csv = tmp_path / 'org.csv'
    rec_csv = tmp_path / 'rec.csv'
    out_parquet = tmp_path / 'merged.parquet'

    base = pd.DataFrame(
        {
            'image_id': ['img1'],
            'new_exp_time': [10.0],
            'x': [1.0],
            'y': [1.0],
            'flux': [10.0],
        }
    )
    base.to_csv(noise_csv, index=False)
    base.to_csv(org_csv, index=False)
    base.to_csv(rec_csv, index=False)

    class _ExecutorWithRaisingFuture(_FakeExecutor):
        def submit(self, fn, *args, **kwargs):
            fut = _RaisingFuture()
            self.futures.append(fut)
            return fut

    monkeypatch.setattr(merge_mod, 'ProcessPoolExecutor', lambda max_workers=1: _ExecutorWithRaisingFuture())
    monkeypatch.setattr(merge_mod, 'as_completed', lambda futures: futures)

    merge_mod.process(
        noise_csv=str(noise_csv),
        org_csv=str(org_csv),
        rec_csv=str(rec_csv),
        workers=1,
        output_parquet=str(out_parquet),
        threshold=2.0,
    )
    assert not out_parquet.exists()


@pytest.mark.unit
def test_module_main_guard_executes_with_stubbed_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import starter

    data = pd.DataFrame(
        {
            'image_id': ['img1'],
            'new_exp_time': [10.0],
            'x': [1.0],
            'y': [1.0],
            'flux': [10.0],
        }
    )
    rec = tmp_path / 'rec.csv'
    org = tmp_path / 'org.csv'
    noise = tmp_path / 'noise.csv'
    out = tmp_path / 'photo.parquet'
    data.to_csv(rec, index=False)
    data.to_csv(org, index=False)
    data.to_csv(noise, index=False)

    cfg = {
        'evaluation': {
            'uncropped_output_dir': str(tmp_path),
            'uncropped_rec_catalog_csv': 'rec.csv',
            'uncropped_noisy_catalog_csv': 'noise.csv',
            'uncropped_org_catalog_csv': 'org.csv',
            'photometrical_data_filename': 'photo.parquet',
            'merge_catalog_workers': 1,
            'workers': 1,
            'merge_catalog_threshold': 3.5,
        }
    }

    monkeypatch.setattr(starter, 'parse_config_overrides', lambda *args, **kwargs: {})
    monkeypatch.setattr(starter, 'load_config', lambda **kwargs: cfg)

    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', _FakeExecutor)
    monkeypatch.setattr(concurrent.futures, 'as_completed', lambda futures: futures)

    runpy.run_module('src.evaluation.merge_catalogs', run_name='__main__')
    assert out.exists()
