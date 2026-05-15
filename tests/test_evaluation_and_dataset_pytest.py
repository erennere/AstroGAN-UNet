from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.data.create_dataset import filter_out_metadata, test_train_validation_split as split_dataset
from src.evaluation.merge_catalogs import merge_based_on_proximity
from src.evaluation.metrics import detect_sources_in_image, sliding_window_inference
from src.training.utils import post_filter


@pytest.mark.unit
@pytest.mark.parametrize(
    ('patch_size', 'stride', 'weighting'),
    [((32, 32, 1), (16, 16, 1), 'average'), ((32, 32, 1), (32, 32, 1), 'gaussian'), ((16, 16, 1), (8, 8, 1), 'distance')],
)
def test_sliding_window_inference_preserves_spatial_shape_and_dtype(patch_size, stride, weighting):
    image = np.random.default_rng(123).random((50, 50, 1), dtype=np.float32)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=patch_size),
        tf.keras.layers.Conv2D(1, 1, use_bias=False, kernel_initializer=tf.keras.initializers.Constant(0.5)),
    ])
    reconstructed = sliding_window_inference(image, model, patch_size=patch_size, stride=stride, weighting=weighting, batch_size=4, gaussian_sigma=4)

    assert reconstructed.shape == image.shape
    assert reconstructed.dtype == np.float32
    assert np.isfinite(reconstructed).all()
    assert not np.allclose(reconstructed, image)


@pytest.mark.unit
def test_merge_based_on_proximity_preserves_rec_rows_and_suffixes(tiny_catalog_df):
    rec_df = tiny_catalog_df.iloc[:5].copy()
    noise_df = tiny_catalog_df.iloc[:5].copy()
    org_df = tiny_catalog_df.iloc[:5].copy()
    for frame in (rec_df, noise_df):
        frame['image_id'] = 'img-1'
        frame['new_exp_time'] = 60.0
    org_df['image_id'] = 'img-1'

    merged = merge_based_on_proximity(rec_df, noise_df, org_df, threshold=0.0)
    matched = merged.iloc[:len(rec_df)]

    assert len(merged) >= len(rec_df)
    assert {'x_rec', 'y_rec', 'flux_rec', 'x_noise', 'flux_noise', 'x_org', 'flux_org'} <= set(merged.columns)
    assert merged['image_id'].eq('img-1').all()
    assert np.allclose(matched['x_rec'], matched['x_org'])
    assert np.allclose(matched['y_rec'], matched['y_org'])


@pytest.mark.unit
def test_merge_based_on_proximity_handles_far_and_empty_catalogs(tiny_catalog_df):
    rec_df = tiny_catalog_df.iloc[:2].copy()
    rec_df['image_id'] = 'img-1'
    rec_df['new_exp_time'] = 30.0
    noise_df = rec_df.copy()
    org_df = tiny_catalog_df.iloc[:2].copy()
    org_df['image_id'] = 'img-1'
    org_df[['x', 'y']] += 1000.0

    merged_far = merge_based_on_proximity(rec_df, noise_df, org_df, threshold=1.0)
    merged_empty = merge_based_on_proximity(rec_df, noise_df.iloc[0:0], org_df.iloc[0:0], threshold=1.0)

    assert merged_far['x_org'].isna().sum() >= len(rec_df)
    assert merged_empty is not None
    assert len(merged_empty) >= len(rec_df)


@pytest.mark.unit
def test_filter_out_metadata_filters_surveys_and_returns_empty_when_no_match(tmp_path: Path, tiny_metadata_df):
    metadata_csv = tmp_path / 'metadata.csv'
    tiny_metadata_df.to_csv(metadata_csv, index=False)

    filtered = filter_out_metadata(str(metadata_csv), 'sci_aper_1234', 'sci_actual_duration', allowed_survey=['IR'], size=20, low=0, high=1_000_000)
    empty = filter_out_metadata(str(metadata_csv), 'sci_aper_1234', 'sci_actual_duration', allowed_survey=['NOT_PRESENT'], size=20, low=0, high=1_000_000)

    assert not filtered.empty
    assert filtered['sci_aper_1234'].eq('IR').all()
    assert isinstance(empty, pd.DataFrame)
    assert empty.empty


@pytest.mark.unit
def test_post_filter_enforces_expected_invariants():
    info = pd.DataFrame(
        {
            'combined_sigma': [2.0, 0.9, 2.0, 2.0, 2.0, 2.0],
            'org_sigma': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'exp_time': [100.0, 100.0, 50.0, 100.0, 100.0, 100.0],
            'new_exp_time': [60.0, 60.0, 60.0, 60.0, 10.0, 60.0],
            'exp_ratio': [2.0, 2.0, 2.0, 0.5, 10.0, 3.0],
        }
    )
    filtered = post_filter(info, {'min_exp_ratio': 1.0, 'max_exp_ratio': 5.0, 'min_exp_time': 20.0})
    empty = post_filter(info.assign(combined_sigma=0.5, exp_time=10.0, exp_ratio=0.2, new_exp_time=1.0), {'min_exp_ratio': 1.0, 'max_exp_ratio': 5.0, 'min_exp_time': 20.0})

    assert len(filtered) == 2
    assert (filtered['combined_sigma'] > filtered['org_sigma']).all()
    assert (filtered['exp_time'] > filtered['new_exp_time']).all()
    assert filtered['exp_ratio'].between(1.0, 5.0).all()
    assert (filtered['new_exp_time'] >= 20.0).all()
    assert empty.empty


@pytest.mark.unit
def test_train_validation_split_has_no_overlap_and_respects_sizes(tiny_metadata_df):
    train, test, val = split_dataset(tiny_metadata_df[['location']].copy(), 'location', split=(60, 20, 20), seed=42)

    assert set(train).isdisjoint(test)
    assert set(train).isdisjoint(val)
    assert set(test).isdisjoint(val)
    assert len(train) == pytest.approx(7, abs=1)
    assert len(test) == pytest.approx(2, abs=1)
    assert len(val) == pytest.approx(2, abs=1)


@pytest.mark.integration
def test_evaluation_pipeline_chain_can_merge_and_roundtrip_parquet(tmp_path: Path, tiny_fits_array, mock_cfg):
    image = np.expand_dims(tiny_fits_array.astype(np.float32), axis=-1)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 1)),
        tf.keras.layers.Conv2D(1, 1, activation='linear', kernel_initializer=tf.keras.initializers.Constant(0.9), bias_initializer='zeros'),
    ])
    reconstructed = sliding_window_inference(image, model, patch_size=(32, 32, 1), stride=(16, 16, 1), weighting='average', batch_size=4)
    kwargs = dict(mock_cfg['evaluation']['kwargs_source'])

    org_result = detect_sources_in_image(tiny_fits_array, kwargs)
    noisy_result = detect_sources_in_image((tiny_fits_array + np.random.default_rng(11).normal(0, 0.1, tiny_fits_array.shape)).astype(np.float32), kwargs)
    rec_result = detect_sources_in_image(reconstructed[:, :, 0], kwargs)

    assert org_result is not None
    assert noisy_result is not None
    assert rec_result is not None

    def to_df(result, include_new_exp_time):
        x, y, flux, flux_err, _ = result
        data = {'image_id': ['img-1'] * len(x), 'x': x, 'y': y, 'flux': flux, 'flux_err': flux_err}
        if include_new_exp_time:
            data['new_exp_time'] = [60.0] * len(x)
        return pd.DataFrame(data)

    org_df = to_df(org_result, include_new_exp_time=False)
    noisy_df = to_df(noisy_result, include_new_exp_time=True)
    rec_df = to_df(rec_result, include_new_exp_time=True)
    merged = merge_based_on_proximity(rec_df, noisy_df, org_df, threshold=5.0)

    parquet_path = tmp_path / 'merged.parquet'
    merged.to_parquet(parquet_path, index=False)
    reloaded = pd.read_parquet(parquet_path)

    assert parquet_path.exists()
    assert list(reloaded.columns) == list(merged.columns)
