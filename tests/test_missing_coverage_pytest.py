"""Targeted coverage for public helpers that were not directly exercised by the existing suite."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starter import resolve_data_generator, resolve_gan_loss, resolve_generator_loss
from src.evaluation import metrics as metrics_mod
from src.evaluation.metrics import create_prediction_dataset, get_test_images, process_models, process_single_model
from src.training.math_helpers import sigma_kernel_from_fit
from src.visualization.prepare_images import create_image
from src.visualization.prepare_plots import plot_hexbin_panel, render_hexbin_plot


@pytest.mark.unit
def test_resolve_registry_helpers_dispatch_and_reject_unknown_names():
    generator_map = {'mae': object()}
    loss_map = {'bce': object()}
    data_generator_map = {'train': object()}

    assert resolve_generator_loss('bce', loss_map) is loss_map['bce']
    assert resolve_gan_loss('bce', loss_map) is loss_map['bce']
    assert resolve_data_generator('train', data_generator_map) is data_generator_map['train']

    with pytest.raises(ValueError, match='Unsupported generator loss'):
        resolve_generator_loss('missing', loss_map)
    with pytest.raises(ValueError, match='Unsupported GAN loss'):
        resolve_gan_loss('missing', loss_map)
    with pytest.raises(ValueError, match='Unsupported data generator'):
        resolve_data_generator('missing', data_generator_map)


@pytest.mark.unit
def test_sigma_kernel_from_fit_uses_perturbed_power_law(monkeypatch: pytest.MonkeyPatch):
    fit_data = pd.DataFrame({'t': [2.0], 'a': [3.0], 'dt': [0.4], 'da': [0.5]})
    row = pd.Series({'exp_time': 4.0})

    monkeypatch.setattr('src.training.math_helpers.np.random.uniform', lambda *args, **kwargs: 0.0)

    result = sigma_kernel_from_fit(row, fit_data, 't', 'a', 'dt', 'da', 'exp_time')

    assert result == pytest.approx(3.0 * (4.0 ** 2.0))


@pytest.mark.unit
def test_get_test_images_sets_test_flag_and_returns_pipeline_result(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_data_augment_pluggable(images, kwargs_data, scaling):
        captured['images'] = images
        captured['kwargs_data'] = dict(kwargs_data)
        captured['scaling'] = scaling
        return pd.DataFrame({'location': ['sample.fits'], 'exp_time': [120.0]})

    monkeypatch.setattr(metrics_mod, 'data_augment_pluggable', fake_data_augment_pluggable)

    kwargs_data = {'location': ['sample.fits'], 'training': False}
    result = get_test_images(['sample.fits'], kwargs_data, scaling='min_max')

    assert kwargs_data['test'] is True
    assert captured['kwargs_data']['test'] is True
    assert captured['scaling'] == 'min_max'
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['location', 'exp_time']


@pytest.mark.unit
def test_create_prediction_dataset_batches_patches_and_positions():
    image = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    dataset = create_prediction_dataset(image, patch_size=(2, 2, 1), stride=(2, 2, 1), batch_size=2)

    batches = list(dataset)
    assert len(batches) == 2

    patches, positions = batches[0]
    assert patches.shape == (2, 2, 2, 1)
    assert positions.shape == (2, 2)
    assert positions.numpy().tolist() == [[0, 0], [0, 2]]


@pytest.mark.unit
def test_create_image_returns_none_when_no_candidates(monkeypatch: pytest.MonkeyPatch):
    import src.visualization.prepare_images as prep_images

    monkeypatch.setattr(prep_images, 'open_fits', lambda *_args, **_kwargs: np.ones((4, 4), dtype=np.float32))
    monkeypatch.setattr(prep_images, 'candidates_based_on_ratio', lambda *_args, **_kwargs: [])

    row = pd.Series({'location': 'image.fits'})
    kwargs_data = {'type_of_image': 'SCI', 'nan_value': 0.0, 'posinf_value': 0.0, 'neginf_value': 0.0}

    assert list(create_image(row, object(), kwargs_data, ps=2)) == []


@pytest.mark.unit
def test_create_image_yields_crops_and_model_outputs(monkeypatch: pytest.MonkeyPatch):
    import src.visualization.prepare_images as prep_images

    crop = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    monkeypatch.setattr(prep_images, 'open_fits', lambda *_args, **_kwargs: np.ones((4, 4), dtype=np.float32))
    monkeypatch.setattr(
        prep_images,
        'candidates_based_on_ratio',
        lambda *_args, **_kwargs: [
            {'combined_sigma': 0.5, 'crop_median_bkg': 1.0, 'exp_ratio': 2.0},
            {'combined_sigma': 1.5, 'crop_median_bkg': 2.0, 'exp_ratio': 4.0},
        ],
    )
    monkeypatch.setattr(prep_images, 'crop_image_generator', lambda *_args, **_kwargs: iter([crop]))
    monkeypatch.setattr(prep_images, 'create_simulated_image_gaussian', lambda image, _median, sigma: image + sigma)

    class DummyModel:
        def predict(self, input_image, verbose=0):
            return input_image + 1.0

    row = pd.Series({'location': 'image.fits'})
    kwargs_data = {'type_of_image': 'SCI', 'nan_value': 0.0, 'posinf_value': 0.0, 'neginf_value': 0.0}

    results = list(create_image(row, DummyModel(), kwargs_data, ps=2))

    assert len(results) == 1
    org, noisy, recs, gammas = results[0]
    assert np.array_equal(org, crop)
    assert len(noisy) == len(recs) == len(gammas) == 2
    assert gammas == [2.0, 4.0]
    assert np.allclose(noisy[0], crop + 0.5)
    assert np.allclose(recs[0], crop + 1.5)


@pytest.mark.unit
def test_plot_hexbin_panel_and_render_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import matplotlib.pyplot as plt
    import src.visualization.prepare_plots as prep_plots

    fig, ax = plt.subplots(figsize=(4, 4))
    assert plot_hexbin_panel(fig, ax, np.array([]), np.array([]), cmap_name='viridis', cmap=plt.get_cmap('viridis'), gridsize=10) is False

    fig, ax = plt.subplots(figsize=(4, 4))
    monkeypatch.setattr(fig, 'colorbar', lambda *args, **kwargs: None)
    x_values = np.array([1.0, 2.0, 3.0, 4.0])
    y_values = np.array([1.5, 2.5, 3.5, 4.5])
    rendered = plot_hexbin_panel(
        fig,
        ax,
        x_values,
        y_values,
        cmap_name='viridis',
        cmap=plt.get_cmap('viridis'),
        gridsize=10,
        xlabel='x',
        ylabel='y',
        title='panel',
        xlim=(0, 5),
        ylim=(0, 5),
        legend_label='sample',
        legend_value=2.0,
        add_one_to_one=True,
    )
    assert rendered is True
    assert ax.get_legend() is not None
    assert len(ax.lines) == 1

    calls = []

    def fake_single(*args, **kwargs):
        calls.append(('single', args[1]))

    def fake_paired(*args, **kwargs):
        calls.append(('paired', args[1]))

    monkeypatch.setattr(prep_plots, 'render_single_hexbin_grid', fake_single)
    monkeypatch.setattr(prep_plots, 'render_paired_hexbin_grid', fake_paired)

    df = pd.DataFrame({'exp_ratio': [2.0], 'x': [1.0], 'y': [2.0]})
    spec_single = {'layout': 'single', 'output_filename': str(tmp_path / 'single.png')}
    spec_paired = {'layout': 'paired', 'output_filename': str(tmp_path / 'paired.png')}

    render_hexbin_plot(df, spec_single['output_filename'], spec_single, {'x': 'x', 'y': 'y'}, (5, 95))
    render_hexbin_plot(df, spec_paired['output_filename'], spec_paired, {'x': 'x', 'y': 'y'}, (5, 95))

    assert calls == [('single', spec_single['output_filename']), ('paired', spec_paired['output_filename'])]

    with pytest.raises(ValueError, match='Unsupported plot layout'):
        render_hexbin_plot(df, str(tmp_path / 'bad.png'), {'layout': 'bad', 'output_filename': 'bad.png'}, {'x': 'x'}, (5, 95))