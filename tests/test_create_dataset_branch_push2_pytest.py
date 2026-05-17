"""Additional branch tests for src/data/create_dataset.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import create_dataset as cd


@pytest.mark.unit
def test_create_dir_exception_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cd, 'ensure_directory_exists', lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('x')))
    assert cd.create_dir('x') is False


@pytest.mark.unit
def test_iter_image_crops_invalid_inputs():
    assert list(cd._iter_image_crops('bad', ps=16)) == []
    assert list(cd._iter_image_crops(np.zeros((4, 4), dtype=np.float32), ps=8)) == []


@pytest.mark.unit
def test_classify_data_appends_100_percentile_when_needed():
    stats = cd.classify_data(np.array([1.0, 2.0, 3.0]), step=6)
    assert '100_mean' in stats


@pytest.mark.unit
def test_filter_out_metadata_error_and_guard_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    missing = cd.filter_out_metadata(str(tmp_path / 'missing.csv'), 'survey', 'exp', ['IR'])
    assert missing is None

    fp = tmp_path / 'meta.csv'
    pd.DataFrame({'a': [1], 'b': [2]}).to_csv(fp, index=False)
    assert cd.filter_out_metadata(str(fp), 'survey', 'exp', ['IR']) is None
    assert cd.filter_out_metadata(str(fp), 'a', 'exp', ['IR']) is None

    df = pd.DataFrame({'survey': ['IR'], 'exp': [100.0], 'last_name': ['SMITH']})
    df.to_csv(fp, index=False)
    assert cd.filter_out_metadata(
        str(fp),
        'survey',
        'exp',
        ['IR'],
        filter_surveys=False,
        filter_by_last_name=True,
        last_name_col=None,
    ) is None


@pytest.mark.unit
def test_filter_out_metadata_sampling_iteration_branch(tmp_path: Path):
    fp = tmp_path / 'meta2.csv'
    df = pd.DataFrame({'survey': ['IR'] * 10, 'exp': np.linspace(100, 109, 10)})
    df.to_csv(fp, index=False)

    out = cd.filter_out_metadata(
        str(fp),
        col='survey',
        exp_column='exp',
        allowed_survey=['IR'],
        size=15,
        low=0,
        high=1000,
        seed=1,
        max_iterations=2,
    )
    assert out is not None
    assert not out.empty


@pytest.mark.unit
def test_test_train_validation_split_missing_column_returns_none():
    out = cd.test_train_validation_split(pd.DataFrame({'x': [1, 2]}), url_column='url')
    assert out is None


@pytest.mark.unit
def test_download_dataset_missing_columns_returns_none(tmp_path: Path):
    df = pd.DataFrame({'id': [1], 'url': ['u']})
    assert cd.download_dataset(df[['url']], 'id', 'url', str(tmp_path)) is None
    assert cd.download_dataset(df[['id']], 'id', 'url', str(tmp_path)) is None


@pytest.mark.unit
def test_process_image_stats_none_open_and_none_stats(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cd, 'open_fits', lambda *_args, **_kwargs: None)
    res = cd.process_image_stats(
        'x.fits', 'SCI', 3, 2, 5, 2, 5, 5, 64, 10.0, False, None, None,
        'filename', 'location',
        {
            'mean_bkg': 'mean_bkg', 'median_bkg': 'median_bkg', 'std_bkg': 'std_bkg', 'max_bkg': 'max_bkg',
            'abs_mean': 'abs_mean', 'abs_median': 'abs_median',
            'mean_src': 'mean_src', 'median_src': 'median_src', 'std_src': 'std_src', 'max_src': 'max_src',
        },
        0.0, 0.0, 0.0,
    )
    assert res is None

    monkeypatch.setattr(cd, 'open_fits', lambda *_args, **_kwargs: np.ones((4, 4), dtype=np.float32))
    monkeypatch.setattr(cd, 'calculate_image_stats', lambda *_args, **_kwargs: None)
    res2 = cd.process_image_stats(
        'x.fits', 'SCI', 3, 2, 5, 2, 5, 5, 64, 10.0, False, None, None,
        'filename', 'location',
        {
            'mean_bkg': 'mean_bkg', 'median_bkg': 'median_bkg', 'std_bkg': 'std_bkg', 'max_bkg': 'max_bkg',
            'abs_mean': 'abs_mean', 'abs_median': 'abs_median',
            'mean_src': 'mean_src', 'median_src': 'median_src', 'std_src': 'std_src', 'max_src': 'max_src',
        },
        0.0, 0.0, 0.0,
    )
    assert res2 is None


@pytest.mark.unit
def test_control_flow_early_returns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(cd, 'filter_out_metadata', lambda *args, **kwargs: None)
    cd.control_flow(
        dataset_dir=str(tmp_path / 'ds'),
        metadata_filepath=str(tmp_path / 'missing.csv'),
        survey_column='survey',
        exp_column='exp',
        id_column='id',
        url_column='url',
        allowed_survey=['IR'],
        split_dirs=['training', 'test', 'eval'],
        originals_subdir='originals',
        masked_images_dirname='masked',
        file_extension='.fits',
        filtered_metadata_output_file=str(tmp_path / 'filtered.csv'),
        noisy_filtered_metadata_output_file=str(tmp_path / 'noisy.csv'),
        cropped_stats_output_file=str(tmp_path / 'crop.csv'),
        url_filename_split_token='/',
        crop_name_separator='_',
        crop_prefix_parts=1,
        original_filename_suffix='_drz.fits',
        stats_column_tokens=['mean', 'median', 'std'],
        temp_index_column='temp_index',
        filename_column='filename',
        original_filename_column='original_filename',
        location_col='location',
        masked_filename_prefix='masked_',
        stats_column_map={
            'mean_bkg': 'mean_bkg', 'median_bkg': 'median_bkg', 'std_bkg': 'std_bkg', 'max_bkg': 'max_bkg',
            'abs_mean': 'abs_mean', 'abs_median': 'abs_median',
            'mean_src': 'mean_src', 'median_src': 'median_src', 'std_src': 'std_src', 'max_src': 'max_src',
        },
        original_stats_prefix='orig_',
        max_iterations=1,
        nan_value=0.0,
        posinf_value=0.0,
        neginf_value=0.0,
        download=True,
        cropping=False,
        stats_on_crops=False,
    )
