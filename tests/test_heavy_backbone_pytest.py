"""High-impact coverage tests for heavy orchestration functions."""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import create_dataset as cd
from src.evaluation import metrics as metrics_mod
from src.visualization import prepare_plots as plots_mod


@pytest.mark.unit
def test_detect_sources_in_image_fallback_background_and_deblend_timeout(monkeypatch: pytest.MonkeyPatch):
    image = np.zeros((64, 64), dtype=np.float32)
    image[20:28, 20:28] = 25.0
    image[40:48, 42:50] = 30.0

    def slow_deblend(*args, **kwargs):
        time.sleep(0.05)
        return args[1]

    monkeypatch.setattr(metrics_mod, 'deblend_sources', slow_deblend)

    kwargs = {
        'sigma': 3.0,
        'maxiters': 5,
        'nsigma': 1.5,
        'npixels': 5,
        'nlevels': 8,
        'contrast': 0.001,
        'footprint_radius': 3,
        'deblend': True,
        'deblend_timeout': 0.0001,
    }

    result = metrics_mod.detect_sources_in_image(image, kwargs)
    assert result is not None
    x, y, flux, flux_err, mask = result
    assert len(x) == len(y) == len(flux) == len(flux_err)
    assert mask.shape == image.shape
    assert mask.dtype == bool


@pytest.mark.unit
def test_detect_sources_in_image_missing_required_params_returns_none():
    image = np.ones((32, 32), dtype=np.float32)
    assert metrics_mod.detect_sources_in_image(image, {'sigma': 3.0}) is None


@pytest.mark.unit
def test_extract_sources_returns_dataframe_and_mask():
    rng = np.random.default_rng(123)
    image = rng.normal(loc=0.0, scale=0.01, size=(96, 96)).astype(np.float32)
    image[30:35, 33:38] += 3.0
    image[60:66, 62:67] += 5.0

    kwargs = {
        'thresh': 1.5,
        'org_thresh': 1.2,
        'radius_factor': 6.0,
        'PHOT_FLUXFRAC': 0.5,
        'r_min': 3.5,
        'elongation_fraction': 1.0,
        'PHOT_AUTOPARAMS': 2.5,
        'maskthresh': 0.0,
        'minarea': 5,
        'org_minarea': 5,
        'filter_type': 'matched',
        'deblend_nthresh': 16,
        'deblend_cont': 0.005,
        'clean': True,
        'clean_param': 1.0,
    }

    df, mask = metrics_mod.extract_sources(image, 'reconstructed', kwargs)
    assert isinstance(df, pd.DataFrame)
    assert mask.shape == image.shape
    for column in ('x', 'y', 'flux_err', 'kron_radius', 'is_galaxy'):
        assert column in df.columns
    assert 'flux' in df.columns or 'flux_y' in df.columns


@pytest.mark.unit
def test_compare_images_wrap_extract_sources_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    base_df = pd.DataFrame(
        {
            'x': [5.0, 10.0, 15.0],
            'y': [6.0, 11.0, 16.0],
            'a': [2.0, 2.2, 2.4],
            'b': [1.0, 1.1, 1.2],
            'theta': [0.0, 0.1, 0.2],
            'flux': [10.0, 20.0, 30.0],
            'flux_err': [1.0, 2.0, 3.0],
        }
    )

    def fake_wrap(image, flag, kwargs):
        if flag == 'reconstructed':
            x = np.array([5.1, np.nan, 15.2], dtype=float)
            y = np.array([6.1, np.nan, 16.3], dtype=float)
        elif flag == 'noisy':
            x = np.array([5.2, 10.3, np.nan], dtype=float)
            y = np.array([6.2, 11.3, np.nan], dtype=float)
        else:
            x = np.array([5.0, 10.0, 15.0], dtype=float)
            y = np.array([6.0, 11.0, 16.0], dtype=float)

        flux = np.array([10.0, 20.0, 30.0], dtype=float)
        flux_err = np.array([1.0, 0.0, 3.0], dtype=float)
        mask = np.zeros((32, 32), dtype=bool)
        df = base_df.copy()
        return x, y, flux, flux_err, mask, df['a'].to_numpy(), df['b'].to_numpy(), df['theta'].to_numpy(), df

    monkeypatch.setattr(metrics_mod, 'wrap_extract_sources', fake_wrap)
    monkeypatch.setattr(metrics_mod, 'plot_source_comparison_sep', lambda *args, **kwargs: None)

    image = np.ones((32, 32), dtype=np.float32)
    kwargs = {
        'func': metrics_mod.wrap_extract_sources,
        'distance_threshold': 2.5,
        'alpha': 1.0,
        'beta': 1.0,
        'gamma': 1.0,
        'k1': 0.01,
        'k2': 0.03,
        'win_size': 7,
        'win_sigma': 1.0,
    }

    result = metrics_mod.compare_images(
        image,
        image + 0.01,
        image + 0.02,
        'img_001',
        100.0,
        50.0,
        str(tmp_path / 'plots'),
        kwargs,
        if_selected=True,
    )

    assert result is not None
    stats, (flux_rec, flux_org), (_, _), (org_df, noisy_df, rec_df) = result
    assert stats['TP'] >= 1
    assert stats['FP'] >= 0
    assert stats['FN'] >= 0
    assert isinstance(flux_rec, list)
    assert isinstance(flux_org, list)
    assert 'image_id' in org_df.columns
    assert 'new_exp_time' in rec_df.columns
    assert 'new_exp_time' in noisy_df.columns


@pytest.mark.unit
def test_compare_images_non_sep_path_metric_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def fake_detect(image, kwargs):
        x = np.array([2.0, 6.0], dtype=float)
        y = np.array([2.0, 6.0], dtype=float)
        flux = np.array([10.0, 20.0], dtype=float)
        flux_err = np.array([1.0, 2.0], dtype=float)
        mask = np.zeros_like(image, dtype=bool)
        return x, y, flux, flux_err, mask

    monkeypatch.setattr(metrics_mod, 'compute_ssim', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('boom')))

    image = np.zeros((16, 16), dtype=np.float32)
    kwargs = {
        'func': fake_detect,
        'distance_threshold': 2.0,
        'alpha': 1.0,
        'beta': 1.0,
        'gamma': 1.0,
        'k1': 0.01,
        'k2': 0.03,
        'win_size': 7,
        'win_sigma': 1.0,
    }

    result = metrics_mod.compare_images(
        image,
        image,
        image,
        'img_002',
        120.0,
        60.0,
        str(tmp_path / 'plots2'),
        kwargs,
        if_selected=False,
    )

    assert result is not None
    stats = result[0]
    assert np.isnan(stats['SSIM_rec'])
    assert np.isnan(stats['PSNR_rec'])


@pytest.mark.unit
def test_download_dataset_runs_async_download_and_validates_columns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    async def fake_download_images(ids, urls, save_dir, max_requests, reset_after):
        return {
            'ids': list(ids),
            'urls': list(urls),
            'save_dir': save_dir,
            'max_requests': max_requests,
            'reset_after': reset_after,
        }

    monkeypatch.setattr(cd, 'download_images', fake_download_images)

    metadata = pd.DataFrame({'id': ['a', 'b'], 'url': ['u1', 'u2']})
    result = cd.download_dataset(metadata, 'id', 'url', str(tmp_path), max_requests=3, reset_after=7)

    assert result['ids'] == ['a', 'b']
    assert result['urls'] == ['u1', 'u2']
    assert result['max_requests'] == 3
    assert result['reset_after'] == 7

    assert cd.download_dataset(metadata[['id']], 'id', 'url', str(tmp_path)) is None


@pytest.mark.unit
def test_extract_filename_from_url_returns_split_basename():
    url = 'https://server/path%2Forig_file.fits'
    assert cd.extract_filename_from_url(url, '%2F') == 'orig_file.fits'


@pytest.mark.unit
def test_control_flow_phase3_merge_on_crops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset_dir = tmp_path / 'dataset'
    split_name = 'training'
    originals_dir = dataset_dir / split_name / 'originals'
    crops_dir = dataset_dir / split_name
    originals_dir.mkdir(parents=True, exist_ok=True)
    (originals_dir / 'origA.fits').write_text('fits', encoding='utf-8')
    (crops_dir / 'origA_0_0.fits').write_text('crop', encoding='utf-8')

    filtered_output = tmp_path / 'filtered.csv'
    noisy_output = tmp_path / 'noisy.csv'
    cropped_output = tmp_path / 'cropped.csv'

    pd.DataFrame({'url': ['https://host/data/origA.fits'], 'id': ['row1'], 'exp_time': [100.0], 'survey': ['IR']}).to_csv(filtered_output, index=False)

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

    def fake_process_image_stats(filepath, *args, **kwargs):
        filename_column = args[12]
        location_col = args[13]
        stats_column_map = args[14]
        original_filename_column = args[18] if len(args) > 18 else None
        row = {
            filename_column: Path(filepath).name,
            location_col: filepath,
            stats_column_map['mean_bkg']: 1.0,
            stats_column_map['median_bkg']: 1.0,
            stats_column_map['std_bkg']: 0.1,
            stats_column_map['max_bkg']: 2.0,
            stats_column_map['abs_mean']: 1.1,
            stats_column_map['abs_median']: 1.2,
            stats_column_map['mean_src']: 2.1,
            stats_column_map['median_src']: 2.2,
            stats_column_map['std_src']: 0.2,
            stats_column_map['max_src']: 3.0,
        }
        if original_filename_column:
            row[original_filename_column] = 'origA.fits'
        return row

    monkeypatch.setattr(cd, 'process_image_stats', fake_process_image_stats)

    cd.control_flow(
        dataset_dir=str(dataset_dir),
        metadata_filepath=str(filtered_output),
        survey_column='survey',
        exp_column='exp_time',
        id_column='id',
        url_column='url',
        allowed_survey=['IR'],
        split_dirs=[split_name],
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
        stats_column_map=stats_map,
        original_stats_prefix='orig_',
        max_iterations=1,
        nan_value=0.0,
        posinf_value=0.0,
        neginf_value=0.0,
        download=False,
        cropping=True,
        stats_on_crops=True,
        save=False,
        max_workers=1,
    )

    assert noisy_output.exists()
    assert cropped_output.exists()
    merged = pd.read_csv(cropped_output)
    assert 'orig_mean_bkg' in merged.columns
    assert 'original_filename' in merged.columns


@pytest.mark.unit
def test_render_hexbin_plot_single_and_paired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame(
        {
            'exp_ratio': rng.choice([0, 2, 4], size=n),
            'a_cflux_org': rng.uniform(1e-3, 1e-1, size=n),
            'a_cflux_rec': rng.uniform(1e-3, 1e-1, size=n),
            'a_cflux_noise': rng.uniform(1e-3, 1e-1, size=n),
        }
    )

    columns = {
        'org_flux': 'a_cflux_org',
        'rec_flux': 'a_cflux_rec',
        'noise_flux': 'a_cflux_noise',
    }

    single_spec = {
        'layout': 'single',
        'output_filename': 'single.png',
        'suptitle': 'Single',
        'rec_cmap': 'viridis',
        'xlabel': 'x',
        'ylabel': 'y',
        'legend_stat': 'median',
        'one_to_one': True,
        'mask': {'positive': ['org_flux', 'rec_flux']},
        'x': {'key': 'org_flux', 'transform': 'identity'},
        'y': {'key': 'rec_flux', 'transform': 'identity'},
    }

    paired_spec = {
        'layout': 'paired',
        'output_filename': 'paired.png',
        'suptitle': 'Paired',
        'rec_cmap': 'viridis',
        'noise_cmap': 'plasma',
        'legend_stat': 'median',
        'mask_mode': 'shared',
        'mask': {'positive': ['org_flux', 'rec_flux', 'noise_flux']},
        'clip_mode': 'aligned',
        'rec': {
            'x': {'key': 'org_flux', 'transform': 'identity'},
            'y': {'key': 'rec_flux', 'transform': 'identity'},
        },
        'noise': {
            'x': {'key': 'org_flux', 'transform': 'identity'},
            'y': {'key': 'noise_flux', 'transform': 'identity'},
        },
        'xlabel_rec': 'x',
        'xlabel_noise': 'x',
        'ylabel_rec': 'y',
        'ylabel_noise': 'y',
    }

    out_single = tmp_path / 'single.png'
    out_paired = tmp_path / 'paired.png'

    monkeypatch.setattr(plots_mod, 'plot_hexbin_panel', lambda *args, **kwargs: True)

    plots_mod.render_hexbin_plot(df, str(out_single), single_spec, columns, (5, 95))
    plots_mod.render_hexbin_plot(df, str(out_paired), paired_spec, columns, (5, 95))

    assert out_single.exists()
    assert out_paired.exists()


@pytest.mark.unit
def test_create_flux_flux_error_diagram_saves_plot(tmp_path: Path):
    rng = np.random.default_rng(3)
    n = 120
    df = pd.DataFrame(
        {
            'exp_ratio': rng.choice([0, 2, 4], size=n),
            'a_cflux_org': rng.uniform(1e-3, 1e-1, size=n),
            'a_cflux_noise': rng.uniform(1e-3, 1e-1, size=n),
            'a_cflux_rec': rng.uniform(1e-3, 1e-1, size=n),
            'a_flux_err_org': rng.uniform(1e-4, 1e-2, size=n),
            'a_flux_err_noise': rng.uniform(1e-4, 1e-2, size=n),
            'a_flux_err_rec': rng.uniform(1e-4, 1e-2, size=n),
        }
    )

    out = tmp_path / 'snr_flux.png'
    plots_mod.create_flux_flux_error_diagram(
        df,
        'a_cflux_org',
        'a_cflux_noise',
        'a_cflux_rec',
        'a_flux_err_org',
        'a_flux_err_noise',
        'a_flux_err_rec',
        str(out),
    )

    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.unit
def test_prepare_plots_main_runs_with_mocked_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    uncropped_dir = tmp_path / 'uncropped'
    output_dir = tmp_path / 'plots'
    uncropped_dir.mkdir(parents=True, exist_ok=True)

    raw_parquet = uncropped_dir / 'photometry.parquet'
    pd.DataFrame({'dummy': [1]}).to_parquet(raw_parquet, index=False)

    cfg = {
        'prepare_plots': {
            'output_dir': str(output_dir),
            'uncropped_output_dir': str(uncropped_dir),
            'photometrical_data_filename': 'photometry.parquet',
            'uncropped_results_csv': 'metrics.csv',
            'metadata_filepath': str(tmp_path / 'metadata.csv'),
            'rec_cmap': 'viridis',
            'noise_cmap': 'plasma',
            'norm_quantiles': [5, 95],
        }
    }

    augmented_df = pd.DataFrame(
        {
            'exp_ratio': [0, 2, 2],
            'abmag_cflux_rec': [24.0, 24.5, 26.0],
            'abmag_cflux_org': [24.0, 24.8, 26.2],
            'flux_x_org': [1.0, np.nan, 2.0],
            'flux_x_rec': [1.1, 2.0, np.nan],
        }
    )

    monkeypatch.setattr(plots_mod, 'parse_config_overrides', lambda: {})
    monkeypatch.setattr(plots_mod, 'load_config', lambda **kwargs: cfg)
    monkeypatch.setattr(plots_mod, 'add_columns', lambda df: augmented_df.copy())
    monkeypatch.setattr(
        plots_mod,
        'create_table',
        lambda metadata_path, metrics_path: pd.DataFrame({'exp_ratio': [0, 2], 'TP': [1, 1], 'FP': [0, 0], 'FN': [0, 0]}),
    )
    monkeypatch.setattr(plots_mod, 'render_hexbin_plot', lambda *args, **kwargs: None)
    monkeypatch.setattr(plots_mod, 'create_flux_flux_error_diagram', lambda *args, **kwargs: None)

    plots_mod.main()

    out_csv = output_dir / 'all_metrics.csv'
    assert out_csv.exists()
    result = pd.read_csv(out_csv)
    assert 'exp_ratio' in result.columns
