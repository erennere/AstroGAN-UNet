"""Additional branch-heavy tests for remaining uncovered paths."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import create_dataset as create_dataset_mod
from src.data import mast as mast_mod
from src.evaluation import metrics as metrics_mod
from src.visualization import prepare_images as prep_images_mod


@pytest.mark.unit
def test_create_dataset_plot_histogram_writes_file(tmp_path: Path):
    df = pd.DataFrame({'exp': [10, 20, 30, 40, 50]})
    out = tmp_path / 'hist.png'
    create_dataset_mod.plot_histogram(df, 'exp', str(out))
    assert out.exists()


@pytest.mark.unit
def test_process_image_cropping_handles_missing_and_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls = {'crop': 0}

    monkeypatch.setattr(create_dataset_mod, 'open_fits', lambda *args, **kwargs: None)
    assert create_dataset_mod.process_image_cropping('x.fits', str(tmp_path), 16, 'SCI', '_', '.fits') is None

    monkeypatch.setattr(create_dataset_mod, 'open_fits', lambda *args, **kwargs: np.ones((32, 32), dtype=np.float32))
    monkeypatch.setattr(create_dataset_mod, 'crop_image', lambda *args, **kwargs: calls.__setitem__('crop', calls['crop'] + 1))
    create_dataset_mod.process_image_cropping('x.fits', str(tmp_path), 16, 'SCI', '_', '.fits')

    assert calls['crop'] == 1


@pytest.mark.unit
def test_process_image_stats_full_path_with_mask_save(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(create_dataset_mod, 'open_fits', lambda *args, **kwargs: np.array([[1.0, np.nan], [np.inf, -np.inf]], dtype=np.float32))
    monkeypatch.setattr(
        create_dataset_mod,
        'calculate_image_stats',
        lambda *args, **kwargs: (
            (1.0, 1.0, 0.1, 2.0, 3.0, 3.0, 0.2, 4.0, 1.5, 1.4),
            np.array([[True, False], [False, True]]),
            {'10_mean': 0.1},
        ),
    )

    calls = {'saved': 0}
    monkeypatch.setattr(create_dataset_mod, 'save_fits', lambda *args, **kwargs: calls.__setitem__('saved', calls['saved'] + 1))

    stats_map = {
        'mean_bkg': 'mean_bkg',
        'median_bkg': 'median_bkg',
        'std_bkg': 'std_bkg',
        'max_bkg': 'max_bkg',
        'abs_mean': 'abs_mean',
        'abs_median': 'abs_median',
        'mean_src': 'mean_src',
        'median_src': 'median_src',
        'std_src': 'std_src',
        'max_src': 'max_src',
    }

    result = create_dataset_mod.process_image_stats(
        filepath='origA_0_0.fits',
        type_of_image='SCI',
        sigma=3,
        n_sigma=2,
        n_pixels=5,
        footprint_radius=3,
        maxiters=5,
        step=5,
        bkg_box_size=32,
        exclude_percentile=10.0,
        save=True,
        masked_images_dirname=str(tmp_path / 'masked'),
        masked_filename_prefix='masked_',
        filename_column='filename',
        location_col='location',
        stats_column_map=stats_map,
        nan_value=0.0,
        posinf_value=10.0,
        neginf_value=-10.0,
        original_filename_column='original_filename',
        crop_name_separator='_',
        crop_prefix_parts=1,
        original_filename_suffix='_drz.fits',
    )

    assert result is not None
    assert result['filename'] == 'origA_0_0.fits'
    assert result['original_filename'] == 'origA_drz.fits'
    assert calls['saved'] == 1


@pytest.mark.unit
def test_calculate_image_stats_returns_none_when_no_sources(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(create_dataset_mod, 'detect_sources', lambda *args, **kwargs: None)
    data = np.ones((16, 16), dtype=np.float32)
    assert create_dataset_mod.calculate_image_stats(data) is None


@pytest.mark.unit
def test_calculate_image_stats_returns_none_when_segmentation_missing_mask(monkeypatch: pytest.MonkeyPatch):
    class NoMaskSegmentation:
        pass

    monkeypatch.setattr(create_dataset_mod, 'detect_sources', lambda *args, **kwargs: NoMaskSegmentation())
    data = np.ones((16, 16), dtype=np.float32)
    assert create_dataset_mod.calculate_image_stats(data) is None


@pytest.mark.unit
def test_calculate_image_stats_fallback_threshold_path(monkeypatch: pytest.MonkeyPatch):
    class FakeSegmentation:
        def make_source_mask(self, footprint=None):
            mask = np.zeros((16, 16), dtype=bool)
            mask[4:8, 4:8] = True
            return mask

    monkeypatch.setattr(create_dataset_mod, 'Background2D', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('bkg fail')))
    monkeypatch.setattr(create_dataset_mod, 'detect_threshold', lambda *args, **kwargs: np.ones((16, 16), dtype=np.float32))
    monkeypatch.setattr(create_dataset_mod, 'detect_sources', lambda *args, **kwargs: FakeSegmentation())

    result = create_dataset_mod.calculate_image_stats(np.ones((16, 16), dtype=np.float32))
    assert result is not None


@pytest.mark.unit
def test_filter_out_metadata_missing_last_name_column_returns_none(tmp_path: Path):
    csv_path = tmp_path / 'meta.csv'
    pd.DataFrame({'survey': ['IR'], 'exp': [100.0]}).to_csv(csv_path, index=False)

    result = create_dataset_mod.filter_out_metadata(
        str(csv_path),
        col='survey',
        exp_column='exp',
        allowed_survey=['IR'],
        filter_by_last_name=True,
        last_name_col=None,
    )
    assert result is None


@pytest.mark.unit
def test_filter_out_metadata_invalid_path_returns_none():
    assert create_dataset_mod.filter_out_metadata('does_not_exist.csv', 'survey', 'exp', ['IR']) is None


@pytest.mark.unit
def test_find_best_models_filter_and_condition_fallback_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    models_root = tmp_path / 'models'
    parts = ['gan', 'attn', 'mse', 'alias', 'z_scale', 'drop0', 'relu', 'sigmoid', 'lrelu', 'sigmoid']
    cp = models_root
    for part in parts:
        cp = cp / part
    cp = cp / 'checkpoints'
    cp.mkdir(parents=True, exist_ok=True)
    for filename in ('model_010.keras', 'model_020.keras'):
        (cp / filename).write_text('x', encoding='utf-8')

    def one_arg_filter(file_list, prototype, modulo):
        return file_list[:1]

    def weird_condition(_, condition_kwargs):
        raise RuntimeError('force fallback')

    monkeypatch.setattr(metrics_mod, '_decode_models_dir', lambda *_: {'model_alias_hex': 'test_alias', 'data_alias_enriched_hex': 'test_hex'})

    with pytest.raises(RuntimeError, match='force fallback'):
        metrics_mod.find_best_performing_models(
            str(models_root),
            condition=weird_condition,
            filter_model=one_arg_filter,
            model_prototype='*.keras',
            modulo=1,
            index=0,
            concurrent_workers=1,
        )


@pytest.mark.unit
def test_reconstruct_patch_handles_scaling_failure():
    class DummyModel:
        def predict(self, arr):
            return np.asarray(arr, dtype=np.float32)

    patch = np.ones((8, 8), dtype=np.float32)

    assert metrics_mod._reconstruct_patch(
        patch,
        scales=(lambda img: None, lambda img, *args: img),
        model=DummyModel(),
        use_mosaic=False,
        patch_size=(4, 4, 1),
        stride=(2, 2, 1),
        weighting='average',
        batch_size=1,
    ) is None


@pytest.mark.unit
def test_prepare_images_compare_images_success_and_invalid(monkeypatch: pytest.MonkeyPatch):
    def fake_extract(image, flag, kwargs):
        x = np.array([2.0, 6.0], dtype=float)
        y = np.array([2.0, 6.0], dtype=float)
        flux = np.array([10.0, 20.0], dtype=float)
        flux_err = np.array([1.0, 2.0], dtype=float)
        mask = np.zeros((8, 8), dtype=bool)
        a = np.array([1.0, 1.2], dtype=float)
        b = np.array([0.8, 1.0], dtype=float)
        theta = np.array([0.0, 0.1], dtype=float)
        df = pd.DataFrame({'x': x, 'y': y})
        return x, y, flux, flux_err, mask, a, b, theta, df

    monkeypatch.setattr(prep_images_mod, 'plot_source_comparison_sep', lambda *args, **kwargs: ['ok'])

    kwargs = {'func': fake_extract, 'distance_threshold': 2.0}
    image = np.ones((8, 8), dtype=np.float32)

    result = prep_images_mod.compare_images(image, image, image, kwargs)
    assert result == ['ok']
    assert prep_images_mod.compare_images('bad', image, image, kwargs) is None


@pytest.mark.unit
def test_filter_out_mast_with_paged_mocked_mission(monkeypatch: pytest.MonkeyPatch):
    class FakeMission:
        def __bool__(self):
            return True

        def get_column_list(self):
            return pd.DataFrame({'name': ['instrument_name']})

        def query_criteria(self, select_cols=None, limit=None, offset=None, **kwargs):
            if offset == 0:
                return Table(rows=[('row1',)], names=('instrument_name',))
            return Table(rows=[], names=('instrument_name',))

    monkeypatch.setattr(mast_mod, 'MastMissions', lambda mission: FakeMission())

    result = mast_mod.filter_out_mast('HST', {'instrument_name': ['WFC3']})
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
