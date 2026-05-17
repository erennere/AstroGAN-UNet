from __future__ import annotations

import runpy
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import Normalize
import starter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization import prepare_plots as plots_mod


@pytest.mark.unit
def test_build_norm_non_finite_bounds_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(plots_mod.np, 'percentile', lambda values, q: (np.nan, 1.0))
    with pytest.raises(ValueError):
        plots_mod.build_norm(pd.Series([1.0, 2.0]), (1, 99))


@pytest.mark.unit
def test_clip_helpers_empty_after_filters():
    assert plots_mod.clip_single_series(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), x_percentiles=(60, 40), y_percentiles=(60, 40)) is None
    assert plots_mod.clip_aligned_paired_series(pd.Series([np.nan]), pd.Series([np.nan]), pd.Series([np.nan]), pd.Series([np.nan])) is None
    assert plots_mod.clip_aligned_paired_series(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), y_percentiles=(60, 40)) is None
    assert plots_mod.clip_separate_paired_series(pd.Series([], dtype=float), pd.Series([], dtype=float), pd.Series([1.0]), pd.Series([1.0])) is None
    assert plots_mod.clip_separate_paired_series(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), y_percentiles=(60, 40)) is None


@pytest.mark.unit
def test_plot_hexbin_panel_norm_and_legend_color_path():
    fig, ax = plt.subplots(1, 1)
    cmap = plt.get_cmap('viridis')
    fig.colorbar = lambda *args, **kwargs: None
    ok = plots_mod.plot_hexbin_panel(
        fig,
        ax,
        np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),
        np.array([1.5, 1.6, 2.5, 2.6, 3.5, 3.6]),
        cmap_name='viridis',
        cmap=cmap,
        gridsize=20,
        norm=Normalize(vmin=1.0, vmax=3.0),
        legend_label='L',
        legend_value=2.0,
    )
    assert ok is True
    plt.close(fig)


@pytest.mark.unit
def test_prepare_single_and_paired_panel_edge_paths(monkeypatch: pytest.MonkeyPatch):
    df = pd.DataFrame({'x': [1.0, 2.0], 'y': [2.0, 3.0], 'z': [1.0, 2.0]})
    columns = {'x': 'x', 'y': 'y', 'z': 'z'}

    single_spec = {
        'mask': {'finite': ['x', 'y']},
        'x': {'key': 'x'},
        'y': {'key': 'y'},
        'xlim_floor': 0.0,
        'norm_mode': 'dynamic',
    }

    monkeypatch.setattr(plots_mod, 'clip_single_series', lambda *args, **kwargs: None)
    assert plots_mod.prepare_single_panel(df, single_spec, columns, (1, 99)) is None

    monkeypatch.setattr(plots_mod, 'clip_single_series', lambda *args, **kwargs: (pd.Series([1.0, 2.0]), pd.Series([2.0, 3.0]), (1.0, 2.0), (2.0, 3.0)))
    out = plots_mod.prepare_single_panel(df, single_spec, columns, (1, 99))
    assert out is not None
    assert out['xlim'][0] == 0.0

    pair_spec_shared = {
        'pre_window': {'key': 'x', 'positive': True, 'percentiles': (1, 99)},
        'mask_mode': 'shared',
        'mask': {'finite': ['x', 'y']},
        'rec': {'x': {'key': 'x'}, 'y': {'key': 'y'}},
        'noise': {'x': {'key': 'x'}, 'y': {'key': 'y'}},
        'clip_mode': 'aligned',
    }
    monkeypatch.setattr(plots_mod, 'build_mask', lambda *args, **kwargs: pd.Series([False, False]))
    assert plots_mod.prepare_paired_panel(df, pair_spec_shared, columns) is None

    pair_spec_sep = {
        'pre_window': {'key': 'x', 'positive': False, 'percentiles': (1, 99)},
        'mask_mode': 'separate',
        'rec_mask': {'finite': ['x']},
        'noise_mask': {'finite': ['x']},
        'rec': {'x': {'key': 'x'}, 'y': {'key': 'y'}},
        'noise': {'x': {'key': 'x'}, 'y': {'key': 'y'}},
        'clip_mode': 'separate',
    }
    monkeypatch.setattr(plots_mod, 'build_mask', lambda *args, **kwargs: pd.Series([True, True]))
    monkeypatch.setattr(plots_mod, 'clip_separate_paired_series', lambda *args, **kwargs: None)
    assert plots_mod.prepare_paired_panel(df, pair_spec_sep, columns) is None


@pytest.mark.unit
def test_prepare_paired_panel_explicit_shared_and_separate_mask_branches(monkeypatch: pytest.MonkeyPatch):
    df = pd.DataFrame({'x': [1.0, 2.0], 'y': [2.0, 3.0]})
    columns = {'x': 'x', 'y': 'y'}

    shared_spec = {
        'pre_window': None,
        'mask_mode': 'shared',
        'mask': {'finite': ['x', 'y']},
        'rec': {'x': {'key': 'x'}, 'y': {'key': 'y'}},
        'noise': {'x': {'key': 'x'}, 'y': {'key': 'y'}},
        'clip_mode': 'aligned',
    }
    monkeypatch.setattr(plots_mod, 'build_mask', lambda *_args, **_kwargs: pd.Series([False, False]))
    assert plots_mod.prepare_paired_panel(df, shared_spec, columns) is None

    separate_spec = {
        'pre_window': None,
        'mask_mode': 'separate',
        'rec_mask': {'finite': ['x']},
        'noise_mask': {'finite': ['x']},
        'rec': {'x': {'key': 'x'}, 'y': {'key': 'y'}},
        'noise': {'x': {'key': 'x'}, 'y': {'key': 'y'}},
        'clip_mode': 'separate',
    }

    masks = [pd.Series([True, True]), pd.Series([True, True])]

    def _next_mask(*_args, **_kwargs):
        return masks.pop(0)

    monkeypatch.setattr(plots_mod, 'build_mask', _next_mask)
    monkeypatch.setattr(plots_mod, 'clip_separate_paired_series', lambda *args, **kwargs: None)
    assert plots_mod.prepare_paired_panel(df, separate_spec, columns) is None

    masks_early = [pd.Series([True, True]), pd.Series([False, False])]

    def _next_mask_early(*_args, **_kwargs):
        return masks_early.pop(0)

    monkeypatch.setattr(plots_mod, 'build_mask', _next_mask_early)
    assert plots_mod.prepare_paired_panel(df, separate_spec, columns) is None


@pytest.mark.unit
def test_prepare_plots_module_main_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = {
        'visualization': {
            'prepare_plots': {
                'output_dir': str(tmp_path / 'out_main_guard'),
                'uncropped_output_dir': str(tmp_path),
                'photometrical_data_filename': 'missing.parquet',
                'uncropped_results_csv': 'metrics.csv',
                'metadata_filepath': str(tmp_path / 'meta.csv'),
                'rec_cmap': 'viridis',
                'noise_cmap': 'plasma',
                'norm_quantiles': [5, 95],
            }
        }
    }

    monkeypatch.setattr(starter, 'parse_config_overrides', lambda *args, **kwargs: {})
    monkeypatch.setattr(starter, 'load_config', lambda **kwargs: cfg)
    with pytest.raises(FileNotFoundError):
        runpy.run_module('src.visualization.prepare_plots', run_name='__main__')


@pytest.mark.unit
def test_render_single_and_paired_grid_logging_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    df = pd.DataFrame({'exp_ratio': list(range(10)), 'x': [1.0] * 10, 'y': [2.0] * 10, 'z': [3.0] * 10})
    columns = {'org_flux': 'x', 'rec_flux': 'y', 'noise_flux': 'z', 'org_kron': 'x', 'rec_kron': 'y', 'noise_kron': 'z', 'org_mag': 'x'}

    single_spec = {
        'layout': 'single',
        'output_filename': 'a.png',
        'suptitle': 's',
        'rec_cmap': 'viridis',
        'mask': {'finite': ['org_flux']},
        'x': {'key': 'org_flux'},
        'y': {'key': 'rec_flux'},
        'xlabel': 'x',
        'ylabel': 'y',
    }
    monkeypatch.setattr(plots_mod, 'prepare_single_panel', lambda *args, **kwargs: None)
    plots_mod.render_single_hexbin_grid(df, str(tmp_path / 'single.png'), single_spec, columns, (1, 99))

    pair_spec = {
        'layout': 'paired',
        'output_filename': 'b.png',
        'suptitle': 'p',
        'rec_cmap': 'viridis',
        'noise_cmap': 'plasma',
        'mask_mode': 'shared',
        'mask': {'finite': ['org_flux']},
        'rec': {'x': {'key': 'org_flux'}, 'y': {'key': 'rec_flux'}},
        'noise': {'x': {'key': 'org_flux'}, 'y': {'key': 'noise_flux'}},
        'clip_mode': 'aligned',
        'ylabel_rec': 'y',
    }
    monkeypatch.setattr(plots_mod, 'prepare_paired_panel', lambda *args, **kwargs: None)
    plots_mod.render_paired_hexbin_grid(df, str(tmp_path / 'pair.png'), pair_spec, columns, (1, 99))


@pytest.mark.unit
def test_create_flux_flux_error_diagram_skips_empty_positive_flux(tmp_path: Path):
    df = pd.DataFrame({
        'exp_ratio': [1],
        'fo': [0.0], 'fn': [0.0], 'fr': [0.0],
        'efo': [1.0], 'efn': [1.0], 'efr': [1.0],
    })
    plots_mod.create_flux_flux_error_diagram(df, 'fo', 'fn', 'fr', 'efo', 'efn', 'efr', str(tmp_path / 'snr.png'))


@pytest.mark.unit
def test_prepare_plots_main_validation_and_error_catch_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    base_cfg = {
        'visualization': {
            'prepare_plots': {
                'output_dir': str(tmp_path / 'out'),
                'uncropped_output_dir': str(tmp_path),
                'photometrical_data_filename': 'data.parquet',
                'uncropped_results_csv': 'metrics.csv',
                'metadata_filepath': str(tmp_path / 'meta.csv'),
                'rec_cmap': 'viridis',
                'noise_cmap': 'plasma',
                'norm_quantiles': [5, 95],
            }
        }
    }

    monkeypatch.setattr(plots_mod, 'parse_config_overrides', lambda: {})

    bad_len = {**base_cfg}
    bad_len['visualization'] = {'prepare_plots': {**base_cfg['visualization']['prepare_plots'], 'norm_quantiles': [5]}}
    monkeypatch.setattr(plots_mod, 'load_config', lambda **kwargs: bad_len)
    with pytest.raises(ValueError):
        plots_mod.main()

    bad_order = {**base_cfg}
    bad_order['visualization'] = {'prepare_plots': {**base_cfg['visualization']['prepare_plots'], 'norm_quantiles': [95, 5]}}
    monkeypatch.setattr(plots_mod, 'load_config', lambda **kwargs: bad_order)
    with pytest.raises(ValueError):
        plots_mod.main()

    bad_range = {**base_cfg}
    bad_range['visualization'] = {'prepare_plots': {**base_cfg['visualization']['prepare_plots'], 'norm_quantiles': [-1, 95]}}
    monkeypatch.setattr(plots_mod, 'load_config', lambda **kwargs: bad_range)
    with pytest.raises(ValueError):
        plots_mod.main()

    monkeypatch.setattr(plots_mod, 'load_config', lambda **kwargs: base_cfg)
    monkeypatch.setattr(plots_mod.os.path, 'exists', lambda path: False)
    with pytest.raises(FileNotFoundError):
        plots_mod.main()


@pytest.mark.unit
def test_prepare_plots_main_catches_render_and_snr_exceptions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    parquet_path = tmp_path / 'data.parquet'
    edited_path = tmp_path / 'edited_data.parquet'
    parquet_path.write_text('x', encoding='utf-8')

    cfg = {
        'visualization': {
            'prepare_plots': {
                'output_dir': str(tmp_path / 'out'),
                'uncropped_output_dir': str(tmp_path),
                'photometrical_data_filename': parquet_path.name,
                'uncropped_results_csv': 'metrics.csv',
                'metadata_filepath': str(tmp_path / 'meta.csv'),
                'rec_cmap': 'viridis',
                'noise_cmap': 'plasma',
                'norm_quantiles': [5, 95],
            }
        }
    }

    df = pd.DataFrame({
        'exp_ratio': [1],
        'flux_x_org': [1.0],
        'flux_x_rec': [1.0],
        'abmag_cflux_rec': [24.0],
        'abmag_cflux_org': [24.0],
        'a_cflux_org': [1.0],
        'a_cflux_rec': [1.0],
        'a_cflux_noise': [1.0],
        'kron_radius_org': [1.0],
        'kron_radius_rec': [1.0],
        'kron_radius_noise': [1.0],
        'a_flux_err_org': [0.1],
        'a_flux_err_noise': [0.1],
        'a_flux_err_rec': [0.1],
    })

    monkeypatch.setattr(plots_mod, 'parse_config_overrides', lambda: {})
    monkeypatch.setattr(plots_mod, 'load_config', lambda **kwargs: cfg)
    monkeypatch.setattr(plots_mod.os.path, 'exists', lambda path: str(path).endswith('data.parquet') or str(path).endswith('edited_data.parquet'))
    monkeypatch.setattr(plots_mod.pd, 'read_parquet', lambda *args, **kwargs: df.copy())
    monkeypatch.setattr(plots_mod, 'add_columns', lambda in_df: in_df)
    monkeypatch.setattr(plots_mod, 'create_table', lambda *args, **kwargs: pd.DataFrame({'exp_ratio': [1]}))
    monkeypatch.setattr(plots_mod, 'build_hexbin_plot_specs', lambda *args, **kwargs: [{'output_filename': 'x.png', 'layout': 'single', 'suptitle': 's', 'rec_cmap': 'viridis', 'mask': {'finite': ['org_flux']}, 'x': {'key': 'org_flux'}, 'y': {'key': 'rec_flux'}, 'xlabel': 'x', 'ylabel': 'y'}])
    monkeypatch.setattr(plots_mod, 'render_hexbin_plot', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('render fail')))
    monkeypatch.setattr(plots_mod, 'create_flux_flux_error_diagram', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('snr fail')))

    plots_mod.main()
    assert (Path(cfg['visualization']['prepare_plots']['output_dir']) / 'all_metrics.csv').exists()
