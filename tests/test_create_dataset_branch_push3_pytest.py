from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import create_dataset as cd
import starter


def _stats_map() -> dict[str, str]:
    return {
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


@pytest.mark.unit
def test_calculate_image_stats_outer_exception_returns_none(monkeypatch: pytest.MonkeyPatch):
    class Seg:
        def make_source_mask(self, footprint=None):
            raise RuntimeError('mask fail')

    monkeypatch.setattr(cd, 'Background2D', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('bkg fail')))
    monkeypatch.setattr(cd, 'detect_threshold', lambda *args, **kwargs: np.ones((8, 8), dtype=np.float32))
    monkeypatch.setattr(cd, 'detect_sources', lambda *args, **kwargs: Seg())

    out = cd.calculate_image_stats(np.ones((8, 8), dtype=np.float32))
    assert out is None


@pytest.mark.unit
def test_filter_out_metadata_read_csv_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    p = tmp_path / 'meta.csv'
    p.write_text('x', encoding='utf-8')
    monkeypatch.setattr(cd.pd, 'read_csv', lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('bad csv')))
    assert cd.filter_out_metadata(str(p), 'survey', 'exp', ['IR']) is None


@pytest.mark.unit
def test_filter_out_metadata_last_name_and_sampling_loop(tmp_path: Path):
    p = tmp_path / 'meta2.csv'
    # Repeated exposure values ensure part1 keeps only a small unique seed set,
    # forcing the additional sampling loop (381-392) to execute.
    df = pd.DataFrame(
        {
            'survey': ['IR'] * 30,
            'exp': [100.0] * 15 + [101.0] * 15,
            'pi_last_name': ['DOE'] * 30,
        }
    )
    df.to_csv(p, index=False)

    out = cd.filter_out_metadata(
        str(p),
        col='survey',
        exp_column='exp',
        allowed_survey=['IR'],
        size=12,
        low=90,
        high=200,
        seed=1,
        max_iterations=2,
        filter_surveys=True,
        filter_by_last_name=True,
        last_name_filter_value=['DOE'],
        last_name_col='pi_last_name',
    )
    assert out is not None
    assert not out.empty
    assert len(out) >= 2


@pytest.mark.unit
def test_filter_out_metadata_additional_size_non_positive_break(tmp_path: Path):
    class WeirdSize:
        def __gt__(self, other):
            return True

        def __sub__(self, other):
            return 0

        def __int__(self):
            return 0

    p = tmp_path / 'meta_weird.csv'
    # Duplicate exposure values keep one row in part1 and leave at least one in subdf.
    pd.DataFrame({'survey': ['IR', 'IR'], 'exp': [100.0, 100.0]}).to_csv(p, index=False)

    out = cd.filter_out_metadata(
        str(p),
        col='survey',
        exp_column='exp',
        allowed_survey=['IR'],
        size=WeirdSize(),
        low=90,
        high=200,
        seed=1,
        max_iterations=2,
    )
    assert out is not None


@pytest.mark.unit
def test_control_flow_split_failure_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    meta = pd.DataFrame({'url': ['a.fits'], 'id': ['a'], 'survey': ['IR'], 'exp': [100.0]})
    monkeypatch.setattr(cd, 'filter_out_metadata', lambda *args, **kwargs: meta)
    monkeypatch.setattr(cd, 'test_train_validation_split', lambda *args, **kwargs: None)

    cd.control_flow(
        dataset_dir=str(tmp_path / 'ds'),
        metadata_filepath=str(tmp_path / 'm.csv'),
        survey_column='survey',
        exp_column='exp',
        id_column='id',
        url_column='url',
        allowed_survey=['IR'],
        split_dirs=['training', 'test', 'eval'],
        originals_subdir='originals',
        masked_images_dirname='masked',
        file_extension='.fits',
        filtered_metadata_output_file=str(tmp_path / 'out' / 'filtered.csv'),
        noisy_filtered_metadata_output_file=str(tmp_path / 'out' / 'noisy.csv'),
        cropped_stats_output_file=str(tmp_path / 'out' / 'crop.csv'),
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
        stats_column_map=_stats_map(),
        original_stats_prefix='org_',
        max_iterations=1,
        nan_value=0.0,
        posinf_value=0.0,
        neginf_value=0.0,
        download=True,
        cropping=False,
        stats_on_crops=False,
    )


@pytest.mark.unit
def test_control_flow_download_phase_create_dir_false_and_download_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    meta = pd.DataFrame({'url': ['a.fits', 'b.fits', 'c.fits'], 'id': ['a', 'b', 'c'], 'survey': ['IR'] * 3, 'exp': [100.0, 110.0, 120.0]})

    monkeypatch.setattr(cd, 'filter_out_metadata', lambda *args, **kwargs: meta)
    monkeypatch.setattr(cd, 'test_train_validation_split', lambda *args, **kwargs: (['a.fits'], ['b.fits'], ['c.fits']))

    def fake_create_dir(path):
        if 'training' in str(path) and str(path).endswith('originals'):
            return False
        Path(path).mkdir(parents=True, exist_ok=True)
        return True

    monkeypatch.setattr(cd, 'create_dir', fake_create_dir)
    monkeypatch.setattr(cd, 'download_dataset', lambda *args, **kwargs: None)

    cd.control_flow(
        dataset_dir=str(tmp_path / 'ds2'),
        metadata_filepath=str(tmp_path / 'm.csv'),
        survey_column='survey',
        exp_column='exp',
        id_column='id',
        url_column='url',
        allowed_survey=['IR'],
        split_dirs=['training', 'test', 'eval'],
        originals_subdir='originals',
        masked_images_dirname='masked',
        file_extension='.fits',
        filtered_metadata_output_file=str(tmp_path / 'out2' / 'filtered.csv'),
        noisy_filtered_metadata_output_file=str(tmp_path / 'out2' / 'noisy.csv'),
        cropped_stats_output_file=str(tmp_path / 'out2' / 'crop.csv'),
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
        stats_column_map=_stats_map(),
        original_stats_prefix='org_',
        max_iterations=1,
        nan_value=0.0,
        posinf_value=0.0,
        neginf_value=0.0,
        download=True,
        cropping=False,
        stats_on_crops=False,
    )


@pytest.mark.unit
def test_control_flow_cropping_missing_dirs_and_stats_on_crops_guard(tmp_path: Path):
    ds3 = tmp_path / 'ds3'
    existing_originals = ds3 / 'training' / 'originals'
    existing_originals.mkdir(parents=True, exist_ok=True)
    (existing_originals / 'a.fits').write_text('x', encoding='utf-8')

    out3 = tmp_path / 'out3'
    out3.mkdir(parents=True, exist_ok=True)
    filtered3 = out3 / 'filtered.csv'
    pd.DataFrame({'url': ['a.fits']}).to_csv(filtered3, index=False)

    original_process_image_stats = cd.process_image_stats
    cd.process_image_stats = lambda *args, **kwargs: {
        'filename': 'a.fits',
        'location': str(existing_originals / 'a.fits'),
        'mean_bkg': 1.0,
        'median_bkg': 1.0,
        'std_bkg': 0.1,
        'max_bkg': 1.0,
        'abs_mean': 1.0,
        'abs_median': 1.0,
        'mean_src': 1.0,
        'median_src': 1.0,
        'std_src': 0.1,
        'max_src': 1.0,
    }
    try:
        cd.control_flow(
            dataset_dir=str(ds3),
            metadata_filepath=str(tmp_path / 'm.csv'),
            survey_column='survey',
            exp_column='exp',
            id_column='id',
            url_column='url',
            allowed_survey=['IR'],
            split_dirs=['training', 'test', 'eval'],
            originals_subdir='originals',
            masked_images_dirname='masked',
            file_extension='.fits',
            filtered_metadata_output_file=str(filtered3),
            noisy_filtered_metadata_output_file=str(out3 / 'noisy.csv'),
            cropped_stats_output_file=str(out3 / 'crop.csv'),
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
            stats_column_map=_stats_map(),
            original_stats_prefix='org_',
            max_iterations=1,
            nan_value=0.0,
            posinf_value=0.0,
            neginf_value=0.0,
            download=False,
            cropping=True,
            stats_on_crops=False,
        )
    finally:
        cd.process_image_stats = original_process_image_stats

    cd.control_flow(
        dataset_dir=str(tmp_path / 'ds4'),
        metadata_filepath=str(tmp_path / 'm.csv'),
        survey_column='survey',
        exp_column='exp',
        id_column='id',
        url_column='url',
        allowed_survey=['IR'],
        split_dirs=['training', 'test', 'eval'],
        originals_subdir='originals',
        masked_images_dirname='masked',
        file_extension='.fits',
        filtered_metadata_output_file=str(tmp_path / 'out4' / 'filtered.csv'),
        noisy_filtered_metadata_output_file=str(tmp_path / 'out4' / 'noisy.csv'),
        cropped_stats_output_file=str(tmp_path / 'out4' / 'crop.csv'),
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
        stats_column_map=_stats_map(),
        original_stats_prefix='org_',
        max_iterations=1,
        nan_value=0.0,
        posinf_value=0.0,
        neginf_value=0.0,
        download=False,
        cropping=False,
        stats_on_crops=True,
    )


@pytest.mark.unit
def test_create_dataset_module_dunder_main_executes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    dataset_cfg = {
        'dataset_dir': str(tmp_path / 'dataset'),
        'metadata_filepath': str(tmp_path / 'meta.csv'),
        'survey_column': 'survey',
        'exp_column': 'exp',
        'id_column': 'id',
        'url_column': 'url',
        'allowed_survey': ['IR'],
        'filter_surveys': True,
        'split_dirs': ['training', 'test', 'eval'],
        'originals_subdir': 'originals',
        'masked_images_dirname': 'masked',
        'file_extension': '.fits',
        'filtered_metadata_output_file': str(tmp_path / 'filtered.csv'),
        'noisy_filtered_metadata_output_file': str(tmp_path / 'noisy.csv'),
        'cropped_stats_output_file': str(tmp_path / 'crop.csv'),
        'url_filename_split_token': '/',
        'crop_name_separator': '_',
        'crop_prefix_parts': 1,
        'original_filename_suffix': '_drz.fits',
        'stats_column_tokens': ['mean', 'median', 'std'],
        'temp_index_column': 'temp_index',
        'filename_column': 'filename',
        'original_filename_column': 'original_filename',
        'location_col': 'location',
        'masked_filename_prefix': 'masked_',
        'stats_column_map': _stats_map(),
        'original_stats_prefix': 'org_',
        'max_iterations': 1,
        'nan_value': 0.0,
        'posinf_value': 0.0,
        'neginf_value': 0.0,
        'download': False,
        'cropping': False,
        'stats_on_crops': False,
        'save': False,
        'size': 10,
        'low': 1,
        'high': 1000,
        'seed': 42,
        'split': [60, 20, 20],
        'max_requests': 1,
        'reset_after': 1,
        'type_of_image': 'SCI',
        'sigma': 3,
        'nsigma': 2,
        'npixels': 5,
        'footprint_radius': 3,
        'maxiters': 5,
        'bkg_box_size': 32,
        'exclude_percentile': 10.0,
        'ps': 64,
        'max_workers': 1,
        'step': 5,
        'filter_by_last_name': False,
        'last_name_filter_value': [],
        'last_name_col': 'last_name',
    }

    monkeypatch.setattr(starter, 'parse_config_overrides', lambda *args, **kwargs: {})
    monkeypatch.setattr(starter, 'load_config', lambda **kwargs: {'create_dataset': dataset_cfg})
    monkeypatch.setattr(os, 'chdir', lambda *_args, **_kwargs: None)

    runpy.run_path(str(REPO_ROOT / 'src' / 'data' / 'create_dataset.py'), run_name='__main__')
