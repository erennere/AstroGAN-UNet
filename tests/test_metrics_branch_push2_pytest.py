"""Additional deterministic branch tests for src/evaluation/metrics.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import metrics as metrics_mod


@pytest.mark.unit
def test_strip_pad_width_overflow_guard_returns_none():
    img = np.zeros((4, 4, 1), dtype=np.float32)
    out = metrics_mod.strip_pad(img, pad_h_top=0, pad_h_bottom=0, pad_w_left=3, pad_w_right=2, pad_eq_h=0, pad_eq_w=0)
    assert out is None


@pytest.mark.unit
def test_sliding_window_inference_input_guards():
    class DummyModel:
        def predict(self, patches):
            return patches

    img = np.zeros((8, 8, 1), dtype=np.float32)
    assert metrics_mod.sliding_window_inference(img, DummyModel(), patch_size=(16, 16, 1), stride=(8, 8, 1)) is None
    assert metrics_mod.sliding_window_inference(img, DummyModel(), patch_size=(8, 8, 1), stride=(16, 16, 1)) is None


@pytest.mark.unit
def test_sliding_window_inference_unknown_weighting_returns_none():
    class DummyModel:
        def predict(self, patches):
            return patches

    img = np.zeros((8, 8, 1), dtype=np.float32)
    assert metrics_mod.sliding_window_inference(img, DummyModel(), patch_size=(4, 4, 1), stride=(2, 2, 1), weighting='unknown') is None


@pytest.mark.unit
def test_scale_image_invalid_input_branches():
    assert metrics_mod.scale_image('bad') is None
    assert metrics_mod.scale_image(np.array([1.0], dtype=np.float32)) is None
    assert metrics_mod.scale_image(np.array([['x']], dtype=object)) is None


@pytest.mark.unit
def test_scale_image_exception_fallback(monkeypatch: pytest.MonkeyPatch):
    img = np.ones((4, 4), dtype=np.float32)

    class BoomNorm:
        def __call__(self, *_args, **_kwargs):
            raise RuntimeError('boom')

    monkeypatch.setattr(metrics_mod.colors, 'Normalize', lambda *args, **kwargs: BoomNorm())
    out_img, vmin, vmax = metrics_mod.scale_image(img)
    assert isinstance(out_img, np.ndarray)
    assert vmin is not None and vmax is not None


@pytest.mark.unit
def test_plot_source_comparison_invalid_image_and_scaling_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])
    matched = np.array([0])
    unmatched = np.array([1])

    metrics_mod.plot_source_comparison(
        original_image='bad',
        noisy_image=np.ones((4, 4), dtype=np.float32),
        reconstructed_image=np.ones((4, 4), dtype=np.float32),
        x_org=x,
        y_org=y,
        x_rec=x,
        y_rec=y,
        matched_indices_org=matched,
        unmatched_indices_org=unmatched,
        matched_indices_rec=matched,
        unmatched_indices_rec=unmatched,
        filepath=str(tmp_path / 'bad.png'),
    )

    monkeypatch.setattr(metrics_mod, 'scale_image', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('scale fail')))
    metrics_mod.plot_source_comparison(
        original_image=np.ones((4, 4), dtype=np.float32),
        noisy_image=np.ones((4, 4), dtype=np.float32),
        reconstructed_image=np.ones((4, 4), dtype=np.float32),
        x_org=x,
        y_org=y,
        x_rec=x,
        y_rec=y,
        matched_indices_org=matched,
        unmatched_indices_org=unmatched,
        matched_indices_rec=matched,
        unmatched_indices_rec=unmatched,
        filepath=str(tmp_path / 'err.png'),
    )


@pytest.mark.unit
def test_plot_source_comparison_happy_path_writes_file(tmp_path: Path):
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])
    matched = np.array([0])
    unmatched = np.array([1])
    out_fp = tmp_path / 'ok.png'

    metrics_mod.plot_source_comparison(
        original_image=np.ones((8, 8), dtype=np.float32),
        noisy_image=np.ones((8, 8), dtype=np.float32),
        reconstructed_image=np.ones((8, 8), dtype=np.float32),
        x_org=x,
        y_org=y,
        x_rec=x,
        y_rec=y,
        matched_indices_org=matched,
        unmatched_indices_org=unmatched,
        matched_indices_rec=matched,
        unmatched_indices_rec=unmatched,
        filepath=str(out_fp),
    )
    assert out_fp.exists()
