"""Tests for save_images (metrics) and uncropped_save_combined_images (uncropped_metrics) flags."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import metrics as metrics_mod
from src.evaluation import uncropped_metrics as um


# ---------------------------------------------------------------------------
# metrics.py — save_images flag
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_process_single_model_save_images_false_skips_fits_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """When save_images=False, save_fits must never be called even when frac=1.0."""
    model_file = tmp_path / 'model_sif_001.keras'
    model_file.write_text('x', encoding='utf-8')
    img_path = tmp_path / 'img.fits'
    img_path.write_text('x', encoding='utf-8')

    df = pd.DataFrame({
        'location': [str(img_path)],
        'sci_actual_duration': [100.0],
        'new_exp_time': [50.0],
        'combined_sigma': [1.0],
    })

    save_calls = []

    class DummyModel:
        pass

    monkeypatch.setattr(metrics_mod, 'build_checkpoint_custom_objects', lambda: {})
    monkeypatch.setattr(metrics_mod, 'load_checkpoint_model', lambda *a, **kw: DummyModel())
    monkeypatch.setattr(metrics_mod, 'open_fits', lambda *a, **kw: np.ones((4, 4), dtype=np.float32))
    monkeypatch.setattr(metrics_mod, '_reconstruct_patch', lambda noisy, *a, **kw: noisy + 1.0)
    monkeypatch.setattr(metrics_mod, 'ensure_directory_exists', lambda *a, **kw: None)
    monkeypatch.setattr(metrics_mod, 'save_fits', lambda img, fname, dst, **kw: save_calls.append(fname))
    monkeypatch.setattr(metrics_mod, 'scale_image', lambda img: (None, 0.0, 1.0))
    monkeypatch.setattr(metrics_mod, 'compare_images', lambda *a, **kw: None)

    out = metrics_mod.process_single_model(
        str(model_file), 'm', None, df,
        kwargs_source={},
        frac=1.0,
        noise_fn=lambda base, row, sigma: base,
        combined_images_dir=str(tmp_path / 'c'),
        png_dir=str(tmp_path / 'p'),
        org_dir=str(tmp_path / 'o'),
        noisy_dir=str(tmp_path / 'n'),
        rec_dir=str(tmp_path / 'r'),
        save_images=False,
    )

    assert out is not None
    assert save_calls == [], 'save_fits should not be called when save_images=False'


@pytest.mark.unit
def test_process_single_model_save_images_true_allows_fits_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """When save_images=True and frac=1.0, save_fits must be called for org/noisy/rec."""
    model_file = tmp_path / 'model_sif_002.keras'
    model_file.write_text('x', encoding='utf-8')
    img_path = tmp_path / 'img.fits'
    img_path.write_text('x', encoding='utf-8')

    df = pd.DataFrame({
        'location': [str(img_path)],
        'sci_actual_duration': [100.0],
        'new_exp_time': [50.0],
        'combined_sigma': [1.0],
    })

    save_calls = []

    class DummyModel:
        pass

    monkeypatch.setattr(metrics_mod, 'build_checkpoint_custom_objects', lambda: {})
    monkeypatch.setattr(metrics_mod, 'load_checkpoint_model', lambda *a, **kw: DummyModel())
    monkeypatch.setattr(metrics_mod, 'open_fits', lambda *a, **kw: np.ones((4, 4), dtype=np.float32))
    monkeypatch.setattr(metrics_mod, '_reconstruct_patch', lambda noisy, *a, **kw: noisy + 1.0)
    monkeypatch.setattr(metrics_mod, 'ensure_directory_exists', lambda *a, **kw: None)
    monkeypatch.setattr(metrics_mod, 'save_fits', lambda img, fname, dst, **kw: save_calls.append(fname))
    monkeypatch.setattr(metrics_mod, 'scale_image', lambda img: (None, 0.0, 1.0))
    monkeypatch.setattr(metrics_mod, 'compare_images', lambda *a, **kw: None)

    metrics_mod.process_single_model(
        str(model_file), 'm', None, df,
        kwargs_source={},
        frac=1.0,
        noise_fn=lambda base, row, sigma: base,
        combined_images_dir=str(tmp_path / 'c'),
        png_dir=str(tmp_path / 'p'),
        org_dir=str(tmp_path / 'o'),
        noisy_dir=str(tmp_path / 'n'),
        rec_dir=str(tmp_path / 'r'),
        save_images=True,
    )

    # Expect three save_fits calls: org, rec, noisy
    assert len(save_calls) == 3


@pytest.mark.unit
def test_process_models_forwards_save_images_to_process_single_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """process_models must pass save_images down to process_single_model."""
    received = {}

    def fake_psm(*args, **kwargs):
        received['save_images'] = kwargs.get('save_images', 'NOT_PASSED')
        return (
            pd.DataFrame({'epoch': ['001'], 'metric': [1.0]}),
            pd.DataFrame({'epoch': ['001'], 'agg': [1.0]}),
            (pd.DataFrame({'x': [1]}), pd.DataFrame({'x': [2]}), pd.DataFrame({'x': [3]})),
        )

    monkeypatch.setattr(metrics_mod, 'process_single_model', fake_psm)

    model_file = str(tmp_path / 'model_001.keras')

    metrics_mod.process_models(
        job=(
            'model_dir',
            pd.DataFrame({'filepath': [model_file], 'epoch': ['001']}),
            None,
            pd.DataFrame({'location': ['img.fits'], 'sci_actual_duration': [100.0],
                          'new_exp_time': [50.0], 'combined_sigma': [1.0]}),
            {'model_alias_hex': 'alias'},
            'data_alias',
        ),
        kwargs_source={},
        workers=1,
        frac=1.0,
        parallel=False,
        save_images=False,
        all_metrics_csv=str(tmp_path / 'all.csv'),
        aggregated_metrics_csv=str(tmp_path / 'agg.csv'),
        org_catalog_csv=str(tmp_path / 'org.csv'),
        noisy_catalog_csv=str(tmp_path / 'noisy.csv'),
        rec_catalog_csv=str(tmp_path / 'rec.csv'),
    )

    assert received['save_images'] is False


# ---------------------------------------------------------------------------
# uncropped_metrics.py — uncropped_save_combined_images flag
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_orchestrate_single_run_save_combined_images_false_skips_dir_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """When save_combined_images=False, ensure_directory_exists must not be called for output_dir."""
    metadata_csv = tmp_path / 'meta.csv'
    pd.DataFrame({
        'exp_ratio': [2.0],
        'location': ['a.fits'],
        'exp_time': [100.0],
        'new_exp_time': [50.0],
        'combined_sigma': [1.0],
    }).to_csv(metadata_csv, index=False)

    dir_calls = []
    monkeypatch.setattr(um, 'ensure_directory_exists', lambda p: dir_calls.append(p))
    monkeypatch.setattr(
        um,
        'process_subdf',
        lambda *a, **kw: (
            [],
            {2.0: [np.zeros(1), np.zeros(1), np.zeros(1)]},
            (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
        ),
    )

    output_dir = str(tmp_path / 'combined')
    output_paths = {
        'results_csv': str(tmp_path / 'results.csv'),
        'org_catalog_csv': str(tmp_path / 'org.csv'),
        'noisy_catalog_csv': str(tmp_path / 'noisy.csv'),
        'rec_catalog_csv': str(tmp_path / 'rec.csv'),
        'hist_data_csv': str(tmp_path / 'hist_data.csv'),
        'hist_png_template': str(tmp_path / 'hist_{exp_ratio}.png'),
    }

    um.orchestrate_single_run(
        N=1,
        model_filepath='model.keras',
        metadata_filepath=str(metadata_csv),
        output_dir=output_dir,
        kwargs={
            'uncropped_patch_size': (2, 2, 1),
            'uncropped_stride': (2, 2, 1),
            'uncropped_weighting': 'average',
            'uncropped_batch_size': 1,
            'nan_value': 0.0,
            'posinf_value': 0.0,
            'neginf_value': 0.0,
            'sigma_key': 'combined_sigma',
            'type_of_image': 'SCI',
            'noise_fn': lambda img, row, sigma: img,
            'uncropped_use_mosaic': True,
        },
        workers=1,
        min_exp=-1,
        max_exp=1,
        output_paths=output_paths,
        save_eval_images=True,
        single_parallel=False,
        write_histograms=False,
        save_combined_images=False,
    )

    assert output_dir not in dir_calls, (
        'ensure_directory_exists must not be called for output_dir when save_combined_images=False'
    )


@pytest.mark.unit
def test_orchestrate_single_run_save_combined_images_false_passes_false_to_subdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """When save_combined_images=False, process_subdf must receive save_eval_images=False."""
    metadata_csv = tmp_path / 'meta.csv'
    pd.DataFrame({
        'exp_ratio': [2.0],
        'location': ['a.fits'],
        'exp_time': [100.0],
        'new_exp_time': [50.0],
        'combined_sigma': [1.0],
    }).to_csv(metadata_csv, index=False)

    received_flags = []

    def fake_process_subdf(sub_df, model_filepath, output_dir, kwargs, bins, save_eval_images):
        received_flags.append(save_eval_images)
        return (
            [],
            {2.0: [np.zeros(1), np.zeros(1), np.zeros(1)]},
            (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
        )

    monkeypatch.setattr(um, 'ensure_directory_exists', lambda p: None)
    monkeypatch.setattr(um, 'process_subdf', fake_process_subdf)

    output_paths = {
        'results_csv': str(tmp_path / 'results.csv'),
        'org_catalog_csv': str(tmp_path / 'org.csv'),
        'noisy_catalog_csv': str(tmp_path / 'noisy.csv'),
        'rec_catalog_csv': str(tmp_path / 'rec.csv'),
        'hist_data_csv': str(tmp_path / 'hist_data.csv'),
        'hist_png_template': str(tmp_path / 'hist_{exp_ratio}.png'),
    }

    um.orchestrate_single_run(
        N=1,
        model_filepath='model.keras',
        metadata_filepath=str(metadata_csv),
        output_dir=str(tmp_path / 'combined'),
        kwargs={
            'uncropped_patch_size': (2, 2, 1),
            'uncropped_stride': (2, 2, 1),
            'uncropped_weighting': 'average',
            'uncropped_batch_size': 1,
            'nan_value': 0.0,
            'posinf_value': 0.0,
            'neginf_value': 0.0,
            'sigma_key': 'combined_sigma',
            'type_of_image': 'SCI',
            'noise_fn': lambda img, row, sigma: img,
            'uncropped_use_mosaic': True,
        },
        workers=1,
        min_exp=-1,
        max_exp=1,
        output_paths=output_paths,
        save_eval_images=True,   # outer flag is True but should be overridden
        single_parallel=False,
        write_histograms=False,
        save_combined_images=False,
    )

    assert all(flag is False for flag in received_flags), (
        'process_subdf must receive False for save_eval_images when save_combined_images=False'
    )


@pytest.mark.unit
def test_orchestrate_uncropped_evaluation_reads_save_combined_images_from_eval_cfg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """orchestrate_uncropped_evaluation must pass eval_cfg['uncropped_save_combined_images']
    to orchestrate_single_run as save_combined_images."""
    metadata_csv = tmp_path / 'meta.csv'
    pd.DataFrame({'exp_ratio': [2.0], 'location': ['a.fits'],
                  'exp_time': [100.0], 'new_exp_time': [50.0]}).to_csv(metadata_csv, index=False)

    received = {}
    monkeypatch.setattr(um, 'process_data', lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(um, 'orchestrate_single_run', lambda *a, **kw: received.update(kw))

    eval_cfg = {
        'uncropped_output_dir': str(tmp_path / 'out'),
        'uncropped_combined_images_dir': str(tmp_path / 'combined'),
        'uncropped_sampled_data_csv': str(tmp_path / 'sampled.csv'),
        'uncropped_results_csv': 'results.csv',
        'uncropped_org_catalog_csv': 'org.csv',
        'uncropped_noisy_catalog_csv': 'noisy.csv',
        'uncropped_rec_catalog_csv': 'rec.csv',
        'uncropped_hist_data_csv': 'hist.csv',
        'uncropped_hist_png_template': 'hist_{exp_ratio}.png',
        'uncropped_n': 1,
        'uncropped_workers': 1,
        'hist_min_exp': -1,
        'hist_max_exp': 1,
        'uncropped_save_images': False,
        'uncropped_save_combined_images': False,
        'uncropped_overwrite': True,
        'uncropped_single_parallel': False,
        'uncropped_write_histograms': False,
        'filter_by_last_name': False,
        'last_name_col': 'sci_pi_last_name',
        'last_name_filter_value': ['FABER'],
        'kwargs_source': {},
    }

    um.orchestrate_uncropped_evaluation(
        eval_cfg=eval_cfg,
        data_cfg={'low': 1.0},
        metadata_filepath=str(metadata_csv),
        model_filepath='model.keras',
        data_alias_enriched_hex='da',
        model_alias_hex='ma',
        epoch='1',
    )

    assert received.get('save_combined_images') is False
    assert received.get('overwrite') is True


# ---------------------------------------------------------------------------
# overwrite flag tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_process_models_overwrite_false_skips_when_csv_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """process_models must return empty frames without running inference when
    overwrite=False and all_metrics_csv already exists."""
    import pandas as pd
    from src.evaluation.metrics import process_models

    # Write a sentinel CSV so the file "already exists"
    existing_csv = tmp_path / 'all_alias_metrics.csv'
    existing_csv.write_text('dummy')

    called = []
    monkeypatch.setattr(metrics_mod, 'process_single_model', lambda *a, **kw: called.append(True))

    all_metrics, aggregated_metrics, dfs = process_models(
        job=('model_dir', pd.DataFrame({'filepath': ['m.keras'], 'epoch': ['001']}),
             None, pd.DataFrame(), {'model_alias_hex': 'alias'}, 'data_alias'),
        kwargs_source={},
        workers=1,
        parallel=False,
        overwrite=False,
        all_metrics_csv=str(tmp_path / 'all_*_metrics.csv').replace('*', 'alias'),
        aggregated_metrics_csv=str(tmp_path / 'agg_alias_metrics.csv'),
        org_catalog_csv=str(tmp_path / 'org_alias_catalog.csv'),
        noisy_catalog_csv=str(tmp_path / 'noisy_alias_catalog.csv'),
        rec_catalog_csv=str(tmp_path / 'rec_alias_catalog.csv'),
    )

    assert not called, 'process_single_model must not be called when overwrite=False and csv exists'
    assert all_metrics.empty
    assert aggregated_metrics.empty


@pytest.mark.unit
def test_process_models_overwrite_false_proceeds_when_csv_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """process_models must proceed normally when overwrite=False but csv does not yet exist."""
    import pandas as pd
    from src.evaluation.metrics import process_models

    called = []
    monkeypatch.setattr(
        metrics_mod, 'process_single_model',
        lambda *a, **kw: called.append(True) or (pd.DataFrame(), pd.DataFrame(), (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    )

    process_models(
        job=('model_dir', pd.DataFrame({'filepath': ['m.keras'], 'epoch': ['001']}),
             None, pd.DataFrame(), {'model_alias_hex': 'alias'}, 'da'),
        kwargs_source={},
        workers=1,
        parallel=False,
        overwrite=False,
        all_metrics_csv=str(tmp_path / 'all_alias_metrics.csv'),
        aggregated_metrics_csv=str(tmp_path / 'agg_alias_metrics.csv'),
        org_catalog_csv=str(tmp_path / 'org_alias_catalog.csv'),
        noisy_catalog_csv=str(tmp_path / 'noisy_alias_catalog.csv'),
        rec_catalog_csv=str(tmp_path / 'rec_alias_catalog.csv'),
    )

    assert called, 'process_single_model must be called when csv does not exist'


@pytest.mark.unit
def test_orchestrate_single_run_overwrite_false_skips_when_results_csv_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """orchestrate_single_run must return immediately without reading metadata when
    overwrite=False and output_paths['results_csv'] already exists."""
    from src.evaluation.uncropped_metrics import orchestrate_single_run

    results_csv = tmp_path / 'results.csv'
    results_csv.write_text('dummy')

    read_called = []
    monkeypatch.setattr(um, 'process_subdf', lambda *a, **kw: read_called.append(True))

    output_paths = {
        'results_csv': str(results_csv),
        'org_catalog_csv': str(tmp_path / 'org.csv'),
        'noisy_catalog_csv': str(tmp_path / 'noisy.csv'),
        'rec_catalog_csv': str(tmp_path / 'rec.csv'),
        'hist_data_csv': str(tmp_path / 'hist.csv'),
        'hist_png_template': str(tmp_path / 'hist_{exp_ratio}.png'),
    }

    # Use a nonexistent metadata_filepath — if the function reads it, it will raise
    orchestrate_single_run(
        N=1,
        model_filepath='model.keras',
        metadata_filepath=str(tmp_path / 'nonexistent_metadata.csv'),
        output_dir=str(tmp_path / 'out'),
        kwargs={},
        workers=1,
        min_exp=-1,
        max_exp=1,
        output_paths=output_paths,
        save_eval_images=False,
        overwrite=False,
    )

    assert not read_called, 'process_subdf must not be called when overwrite=False and results_csv exists'
