from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import starter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization import prepare_images as prep_images_mod


@pytest.mark.unit
def test_create_image_returns_empty_when_open_fits_none(monkeypatch: pytest.MonkeyPatch):
    row = {'location': 'x.fits'}
    kwargs = {
        'nan_value': 0.0,
        'posinf_value': 0.0,
        'neginf_value': 0.0,
        'type_of_image': 'SCI',
    }
    monkeypatch.setattr(prep_images_mod, 'open_fits', lambda *args, **kwargs: None)
    out = list(prep_images_mod.create_image(row, model=object(), kwargs_data=kwargs, ps=8) or [])
    assert out == []


@pytest.mark.unit
def test_create_image_returns_empty_when_no_candidates(monkeypatch: pytest.MonkeyPatch):
    class DummyModel:
        def predict(self, arr, verbose=0):
            return np.asarray(arr, dtype=np.float32)

    row = {'location': 'x.fits'}
    kwargs = {
        'nan_value': 0.0,
        'posinf_value': 0.0,
        'neginf_value': 0.0,
        'type_of_image': 'SCI',
    }
    monkeypatch.setattr(prep_images_mod, 'open_fits', lambda *args, **kwargs: np.ones((8, 8), dtype=np.float32))
    monkeypatch.setattr(prep_images_mod, 'candidates_based_on_ratio', lambda *args, **kwargs: [])
    out = list(prep_images_mod.create_image(row, model=DummyModel(), kwargs_data=kwargs, ps=8) or [])
    assert out == []


@pytest.mark.unit
def test_plot_source_comparison_sep_validation_and_coordinate_skip(monkeypatch: pytest.MonkeyPatch):
    bad = prep_images_mod.plot_source_comparison_sep(
        np.array([['x']], dtype=object),
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]),
        np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0]),
        np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]),
        np.array([0.0]), np.array([0.0]), np.array([0.0]),
    )
    assert bad is None

    called = {'n': 0}
    monkeypatch.setattr(prep_images_mod, 'compare_images', lambda *args, **kwargs: None)
    monkeypatch.setattr(prep_images_mod, 'create_composite_plot_detections', lambda *args, **kwargs: called.__setitem__('n', called['n'] + 1))
    prep_images_mod.coordinate_detect_source(
        np.ones((4, 4), dtype=np.float32),
        [np.ones((4, 4), dtype=np.float32)],
        [np.ones((4, 4), dtype=np.float32)],
        [2.0],
        'label',
        {'distance_threshold': 2.0, 'func': object()},
        'out.png',
    )
    assert called['n'] == 0


@pytest.mark.unit
def test_main_missing_metadata_and_none_metadata_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    missing_cfg = {
        'visualization': {
            'prepare_images': {
                'data_kwargs': {'nan_value': 0.0, 'posinf_value': 0.0, 'neginf_value': 0.0},
                'output_dir': str(tmp_path / 'out'),
                'low': 1,
                'metadata_filepath': str(tmp_path / 'missing.csv'),
                'dataset': 'dataset',
                'sample_n': 1,
                'ps': 8,
                'kwargs_source': {'distance_threshold': 2.0, 'func': object()},
                'exp_column': 'exp_time',
                'targ_col': 'target',
                'type_of_image': 'SCI',
                'ratio_initial': 2,
                'ratio_count': 1,
                'ratio_growth': 1,
                'model_dir': 'models',
                'model_prototype': '*.keras',
                'scaling': 'min_max',
            }
        }
    }

    monkeypatch.setattr(prep_images_mod, 'parse_config_overrides', lambda: {})
    monkeypatch.setattr(prep_images_mod, 'load_config', lambda **kwargs: missing_cfg)
    monkeypatch.setattr(prep_images_mod, 'find_best_performing_models', lambda *args, **kwargs: 'dummy.keras')
    monkeypatch.setattr(prep_images_mod, 'read_checkpoint_info', lambda *_: {'scaling': 'min_max'})
    monkeypatch.setattr(prep_images_mod, 'load_checkpoint_model', lambda *args, **kwargs: object())
    prep_images_mod.main()

    none_meta_cfg = {
        'visualization': {
            'prepare_images': {
                'data_kwargs': {'nan_value': 0.0, 'posinf_value': 0.0, 'neginf_value': 0.0},
                'output_dir': str(tmp_path / 'out2'),
                'low': 1,
                'metadata_filepath': None,
                'dataset': 'dataset',
                'sample_n': 1,
                'ps': 8,
                'kwargs_source': {'distance_threshold': 2.0, 'func': object()},
                'exp_column': 'exp_time',
                'targ_col': 'target',
                'type_of_image': 'SCI',
                'ratio_initial': 2,
                'ratio_count': 1,
                'ratio_growth': 1,
                'model_dir': 'models',
                'model_prototype': '*.keras',
                'scaling': 'min_max',
            }
        }
    }

    test_df = pd.DataFrame({
        'location': ['test/file.fits'],
        'exp_time': [100.0],
        'target': ['M51'],
        'dataset': ['id1'],
    })

    monkeypatch.setattr(prep_images_mod, 'load_config', lambda **kwargs: none_meta_cfg)
    monkeypatch.setattr(prep_images_mod, 'get_test_images', lambda *args, **kwargs: test_df.copy())
    monkeypatch.setattr(prep_images_mod, 'create_image', lambda *args, **kwargs: [(np.ones((4, 4), dtype=np.float32), [np.ones((4, 4), dtype=np.float32)], [np.ones((4, 4), dtype=np.float32)], [2.0])])
    monkeypatch.setattr(prep_images_mod, 'create_composite_plot', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('plot fail')))
    monkeypatch.setattr(prep_images_mod, 'coordinate_detect_source', lambda *args, **kwargs: None)

    prep_images_mod.main()


@pytest.mark.unit
def test_plot_source_comparison_sep_validation_for_non_array_and_low_dim():
    image = np.ones((2, 2), dtype=np.float32)

    out_non_array = prep_images_mod.plot_source_comparison_sep(
        'not-array',
        image,
        image,
        np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]),
        np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0]),
        np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]),
        np.array([0.0]), np.array([0.0]), np.array([0.0]),
    )
    assert out_non_array is None

    out_low_dim = prep_images_mod.plot_source_comparison_sep(
        np.array([1.0], dtype=np.float32),
        image,
        image,
        np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]),
        np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0]),
        np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]),
        np.array([0.0]), np.array([0.0]), np.array([0.0]),
    )
    assert out_low_dim is None


@pytest.mark.unit
def test_plot_source_comparison_sep_exception_branch(monkeypatch: pytest.MonkeyPatch):
    image = np.ones((4, 4), dtype=np.float32)
    monkeypatch.setattr(prep_images_mod, 'Ellipse', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('ellipse-fail')))

    out = prep_images_mod.plot_source_comparison_sep(
        image,
        image,
        image,
        np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]),
        np.array([0]), np.array([], dtype=int), np.array([0]), np.array([], dtype=int), np.array([0]), np.array([], dtype=int),
        np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]),
        np.array([0.0]), np.array([0.0]), np.array([0.0]),
    )
    assert out is None


@pytest.mark.unit
def test_compare_images_shape_mismatch_and_source_exception_and_tree_exception():
    image = np.ones((4, 4), dtype=np.float32)
    out_shape = prep_images_mod.compare_images(
        image,
        np.ones((5, 5), dtype=np.float32),
        image,
        {'func': lambda *_args, **_kwargs: None, 'distance_threshold': 2.0},
    )
    assert out_shape is None

    def raising_func(*_args, **_kwargs):
        raise RuntimeError('extract-fail')

    out_extract = prep_images_mod.compare_images(
        image,
        image,
        image,
        {'func': raising_func, 'distance_threshold': 2.0},
    )
    assert out_extract is None

    def fixed_func(_img, _flag, _kwargs):
        return (
            np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([0.1]), np.array([False]),
            np.array([1.0]), np.array([1.0]), np.array([0.0]), None,
        )

    out_tree = prep_images_mod.compare_images(
        image,
        image,
        image,
        {'func': fixed_func, 'distance_threshold': 'bad-threshold'},
    )
    assert out_tree is None


@pytest.mark.unit
def test_prepare_images_module_main_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = {
        'visualization': {
            'prepare_images': {
                'data_kwargs': {'nan_value': 0.0, 'posinf_value': 0.0, 'neginf_value': 0.0},
                'output_dir': str(tmp_path / 'out_main_guard'),
                'low': 1,
                'metadata_filepath': str(tmp_path / 'missing_from_main_guard.csv'),
                'dataset': 'dataset',
                'sample_n': 1,
                'ps': 8,
                'kwargs_source': {'distance_threshold': 2.0, 'func': object()},
                'exp_column': 'exp_time',
                'targ_col': 'target',
                'type_of_image': 'SCI',
                'ratio_initial': 2,
                'ratio_count': 1,
                'ratio_growth': 1,
                'model_dir': 'models',
                'model_prototype': '*.keras',
                'scaling': 'min_max',
            }
        }
    }

    monkeypatch.setattr(starter, 'parse_config_overrides', lambda *args, **kwargs: {})
    monkeypatch.setattr(starter, 'load_config', lambda **kwargs: cfg)
    runpy.run_module('src.visualization.prepare_images', run_name='__main__')
