from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import uncropped_metrics as um


@pytest.mark.unit
def test_log_range_returns_sorted_decade_values():
    values = um.log_range(-1, 1)
    expected_max = 9 * (10.0 ** 1)

    assert isinstance(values, np.ndarray)
    assert len(values) == 27
    assert np.all(np.diff(values) > 0)
    assert values[0] == pytest.approx(0.1)
    assert values[-1] == pytest.approx(expected_max)


@pytest.mark.unit
def test_process_data_filters_by_last_name_and_low_threshold(tmp_path: Path, tiny_metadata_df: pd.DataFrame, mocker):
    metadata_csv = tmp_path / 'metadata.csv'
    output_csv = tmp_path / 'sampled.csv'
    tiny_metadata_df.to_csv(metadata_csv, index=False)

    def fake_candidates(row, _kwargs_data):
        return [
            {
                'name': row['filename'],
                'new_exp_time': float(row['exp_time']) / 2.0,
                'exp_ratio': float(row['exp_ratio']),
            },
            {
                'name': row['filename'],
                'new_exp_time': float(row['exp_time']) * 2.0,
                'exp_ratio': float(row['exp_ratio']),
            },
        ]

    mocker.patch('src.evaluation.uncropped_metrics.candidates_based_on_range', side_effect=fake_candidates)

    kwargs_data = {'low': 150.0}
    kwargs_eval = {
        'filter_by_last_name': True,
        'last_name_col': 'sci_pi_last_name',
        'last_name_filter_value': ['FABER'],
    }

    filtered = um.process_data(str(metadata_csv), kwargs_data, kwargs_eval, str(output_csv))

    assert output_csv.exists()
    assert not filtered.empty
    assert (filtered['new_exp_time'] >= 150.0).all()

    expected_rows = tiny_metadata_df[tiny_metadata_df['sci_pi_last_name'] == 'FABER']
    assert len(filtered) == len(expected_rows)


@pytest.mark.unit
def test_main_aggregates_histograms_and_catalog_outputs(tmp_path: Path, mocker):
    sample_metric = 0.5
    metadata_csv = tmp_path / 'uncropped_metadata.csv'
    metadata = pd.DataFrame(
        {
            'exp_ratio': [2.0, 2.0, 3.0, 3.0],
            'location': ['a.fits', 'b.fits', 'c.fits', 'd.fits'],
            'name': ['a', 'b', 'c', 'd'],
        }
    )
    metadata.to_csv(metadata_csv, index=False)

    class _FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _FakeExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return _FakeFuture(fn(*args, **kwargs))

    def fake_process_subdf(sub_df, _model_filepath, _output_dir, _kwargs, bins, _save_eval_images):
        ratios = sorted(float(v) for v in sub_df['exp_ratio'].unique())
        hists = {
            ratio: [
                np.ones(len(bins) - 1),
                np.ones(len(bins) - 1) * 2.0,
                np.ones(len(bins) - 1) * 3.0,
            ]
            for ratio in ratios
        }
        first_ratio = ratios[0]
        org_df = pd.DataFrame({'image_id': [f'org_{int(first_ratio)}'], 'x': [1.0]})
        noisy_df = pd.DataFrame({'image_id': [f'org_{int(first_ratio)}'], 'x': [2.0]})
        rec_df = pd.DataFrame({'image_id': [f'org_{int(first_ratio)}'], 'x': [3.0]})
        return ([{'exp_ratio': first_ratio, 'metric': sample_metric}], hists, (org_df, noisy_df, rec_df))

    mocker.patch('src.evaluation.uncropped_metrics.ProcessPoolExecutor', _FakeExecutor)
    mocker.patch('src.evaluation.uncropped_metrics.as_completed', side_effect=lambda futures: futures)
    mocker.patch('src.evaluation.uncropped_metrics.process_subdf', side_effect=fake_process_subdf)

    output_dir = tmp_path / 'plots'
    output_paths = {
        'results_csv': str(tmp_path / 'results' / 'results.csv'),
        'org_catalog_csv': str(tmp_path / 'results' / 'org.csv'),
        'noisy_catalog_csv': str(tmp_path / 'results' / 'noisy.csv'),
        'rec_catalog_csv': str(tmp_path / 'results' / 'rec.csv'),
        'hist_data_csv': str(tmp_path / 'results' / 'hist_data.csv'),
        'hist_png_template': str(tmp_path / 'results' / 'hist_{exp_ratio}.png'),
    }

    um.main(
        N=4,
        model_filepath='model.keras',
        metadata_filepath=str(metadata_csv),
        output_dir=str(output_dir),
        kwargs={'unused': True},
        workers=2,
        min_exp=-1,
        max_exp=1,
        output_paths=output_paths,
        save_eval_images=False,
    )

    results_df = pd.read_csv(output_paths['results_csv'])
    hist_df = pd.read_csv(output_paths['hist_data_csv'])
    org_df = pd.read_csv(output_paths['org_catalog_csv'])
    noisy_df = pd.read_csv(output_paths['noisy_catalog_csv'])
    rec_df = pd.read_csv(output_paths['rec_catalog_csv'])

    assert not results_df.empty
    assert set(hist_df['label'].unique()) == {'original', 'noisy', 'reconstructed'}
    assert set(hist_df['exp_ratio'].unique()) == {2.0, 3.0}
    assert not org_df.empty
    assert not noisy_df.empty
    assert not rec_df.empty
    for ratio in (2, 3):
        assert (tmp_path / 'results' / f'hist_{ratio}.png').exists()
