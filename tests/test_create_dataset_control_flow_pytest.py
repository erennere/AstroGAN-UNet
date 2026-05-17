"""Coverage for the dataset pipeline orchestration in src/data/create_dataset.py."""
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
def test_control_flow_phase1_and_phase2_writes_expected_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset_dir = tmp_path / 'dataset'
    metadata_path = tmp_path / 'metadata.csv'
    metadata = pd.DataFrame({
        'url': ['sample1.fits', 'sample2.fits', 'sample3.fits'],
        'sci_pi_last_name': ['SMITH', 'DOE', 'FABER'],
        'sci_aper_1234': ['IR', 'IR', 'IR'],
        'exp_time': [120.0, 130.0, 140.0],
        'id': ['a', 'b', 'c'],
    })
    metadata.to_csv(metadata_path, index=False)

    def fake_filter_out_metadata(*args, **kwargs):
        return metadata.copy()

    def fake_split(df, url_column, split, seed):
        urls = df[url_column].tolist()
        return urls[:1], urls[1:2], urls[2:3]

    def fake_download_dataset(temp_data, id_column, url_column, save_dir, max_requests=5, reset_after=10):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        for url in temp_data[url_column]:
            (Path(save_dir) / url).write_text('fits', encoding='utf-8')
        return True

    def fake_process_image_cropping(filepath, split_dir, ps, type_of_image, crop_name_separator, output_extension):
        crop_path = Path(split_dir) / f'{Path(filepath).stem}_0_0{output_extension}'
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop_path.write_text('crop', encoding='utf-8')
        return str(crop_path)

    def fake_process_image_stats(filepath, *args, **kwargs):
        base = Path(filepath).name
        filename_column = kwargs['filename_column'] if 'filename_column' in kwargs else args[12]
        location_col = kwargs['location_col'] if 'location_col' in kwargs else args[13]
        stats_column_map = kwargs['stats_column_map'] if 'stats_column_map' in kwargs else args[14]
        return {
            filename_column: base,
            location_col: filepath,
            stats_column_map['mean_bkg']: 1.0,
            stats_column_map['median_bkg']: 1.0,
            stats_column_map['std_bkg']: 0.1,
            stats_column_map['max_bkg']: 2.0,
            stats_column_map['abs_mean']: 1.5,
            stats_column_map['abs_median']: 1.4,
            stats_column_map['mean_src']: 2.5,
            stats_column_map['median_src']: 2.4,
            stats_column_map['std_src']: 0.2,
            stats_column_map['max_src']: 3.0,
        }

    monkeypatch.setattr(cd, 'filter_out_metadata', fake_filter_out_metadata)
    monkeypatch.setattr(cd, 'test_train_validation_split', fake_split)
    monkeypatch.setattr(cd, 'download_dataset', fake_download_dataset)
    monkeypatch.setattr(cd, 'process_image_cropping', fake_process_image_cropping)
    monkeypatch.setattr(cd, 'process_image_stats', fake_process_image_stats)

    filtered_output = tmp_path / 'filtered' / 'metadata.csv'
    noisy_output = tmp_path / 'noisy' / 'metadata.csv'
    cropped_output = tmp_path / 'cropped' / 'stats.csv'

    cd.control_flow(
        dataset_dir=str(dataset_dir),
        metadata_filepath=str(metadata_path),
        survey_column='sci_aper_1234',
        exp_column='exp_time',
        id_column='id',
        url_column='url',
        allowed_survey=['IR'],
        split_dirs=['training', 'test', 'eval'],
        originals_subdir='originals',
        masked_images_dirname='masked',
        file_extension='.fits',
        filtered_metadata_output_file=str(filtered_output),
        noisy_filtered_metadata_output_file=str(noisy_output),
        cropped_stats_output_file=str(cropped_output),
        url_filename_split_token='/',
        crop_name_separator='_',
        crop_prefix_parts=1,
        original_filename_suffix='_drz.fits',
        stats_column_tokens=['mean', 'median', 'std', 'max', 'abs'],
        temp_index_column='temp_index',
        filename_column='filename',
        original_filename_column='original_filename',
        location_col='location',
        masked_filename_prefix='masked_',
        stats_column_map={
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
        },
        original_stats_prefix='orig_',
        max_iterations=2,
        nan_value=0.0,
        posinf_value=0.0,
        neginf_value=0.0,
        download=True,
        cropping=True,
        stats_on_crops=False,
        save=True,
        size=3,
        low=1,
        high=1000,
        seed=42,
        split=(60, 20, 20),
        max_requests=1,
        reset_after=1,
        type_of_image='SCI',
        sigma=3,
        n_sigma=2,
        n_pixels=10,
        footprint_radius=10,
        maxiters=10,
        bkg_box_size=64,
        exclude_percentile=10.0,
        ps=2,
        max_workers=1,
        step=5,
        filter_surveys=True,
        filter_by_last_name=False,
        last_name_filter_value=[],
        last_name_col='sci_pi_last_name',
    )

    assert filtered_output.exists()
    assert noisy_output.exists()
    filtered_df = pd.read_csv(filtered_output)
    noisy_df = pd.read_csv(noisy_output)
    assert not filtered_df.empty
    assert not noisy_df.empty
    assert set(noisy_df['filename']) <= {'sample1.fits', 'sample2.fits', 'sample3.fits'}
