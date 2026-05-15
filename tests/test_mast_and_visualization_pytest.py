from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest

from src.data.mast import _chunked, _choose_best_product, merge_products_with_metadata
from src.visualization.prepare_images import build_composite_axes, coordinate_detect_source, plot_source_comparison_sep


@pytest.mark.unit
def test_chunked_uses_minimum_chunk_size_of_one():
    chunks = list(_chunked([1, 2, 3], 0))

    assert chunks == [[1], [2], [3]]


@pytest.mark.unit
def test_choose_best_product_prefers_science_drz_with_highest_calibration_level():
    products = pd.DataFrame(
        {
            'productFilename': ['a_flt.fits', 'a_drz.fits', 'b_drz.fits'],
            'productType': ['SCIENCE', 'SCIENCE', 'PREVIEW'],
            'calib_level': [1, 3, 4],
            'dataURI': ['uri-a', 'uri-b', 'uri-c'],
        }
    )

    best = _choose_best_product(products, prefer_token='drz')

    assert best is not None
    assert best['productFilename'] == 'a_drz.fits'
    assert best['dataURI'] == 'uri-b'


@pytest.mark.unit
def test_merge_products_with_metadata_fills_missing_urls_only(mocker):
    metadata = pd.DataFrame(
        {
            'dataset_id': ['A', 'B', 'C'],
            'resolved_url': [None, 'https://existing', None],
        }
    )
    resolver = mocker.patch(
        'src.data.mast._resolve_products_bulk',
        side_effect=lambda ids, **kwargs: {dataset_id: f'https://resolved/{dataset_id}' for dataset_id in ids},
    )

    merged = merge_products_with_metadata(
        metadata,
        id_column='dataset_id',
        url_column='resolved_url',
        max_workers=1,
        chunk_size=2,
    )

    assert resolver.called
    assert merged.loc[merged['dataset_id'] == 'A', 'resolved_url'].item() == 'https://resolved/A'
    assert merged.loc[merged['dataset_id'] == 'B', 'resolved_url'].item() == 'https://existing'
    assert merged.loc[merged['dataset_id'] == 'C', 'resolved_url'].item() == 'https://resolved/C'


@pytest.mark.unit
def test_merge_products_with_metadata_returns_input_when_id_column_missing():
    metadata = pd.DataFrame({'other': ['A']})

    merged = merge_products_with_metadata(metadata, id_column='dataset_id', url_column='resolved_url')

    assert merged.equals(metadata)


@pytest.mark.unit
def test_build_composite_axes_creates_expected_panel_pairs():
    fig, ax0, panel_axes = build_composite_axes(5)
    try:
        assert ax0 is not None
        assert len(panel_axes) == 5
        assert len(fig.axes) == 11
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


@pytest.mark.unit
def test_plot_source_comparison_sep_returns_ellipse_groups():
    image = np.ones((16, 16), dtype=np.float32)
    result = plot_source_comparison_sep(
        image,
        image,
        image,
        np.array([4.0, 8.0]),
        np.array([4.0, 8.0]),
        np.array([4.0, 10.0]),
        np.array([4.0, 10.0]),
        np.array([5.0, 12.0]),
        np.array([5.0, 12.0]),
        np.array([0]),
        np.array([1]),
        np.array([0]),
        np.array([1]),
        np.array([0]),
        np.array([1]),
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
    )

    assert result is not None
    assert len(result) == 3
    assert [len(group) for group in result] == [2, 2, 2]


@pytest.mark.unit
def test_coordinate_detect_source_filters_failed_pairs(mocker, tmp_path: Path):
    noisy = [np.ones((8, 8), dtype=np.float32), np.full((8, 8), 2.0, dtype=np.float32)]
    recs = [np.ones((8, 8), dtype=np.float32), np.full((8, 8), 3.0, dtype=np.float32)]
    gammas = [2.0, 4.0]
    ellipses = [[['org'], ['rec'], ['noisy']]]
    compare = mocker.patch(
        'src.visualization.prepare_images.compare_images',
        side_effect=[ellipses[0], None],
    )
    create_plot = mocker.patch('src.visualization.prepare_images.create_composite_plot_detections')

    coordinate_detect_source(
        org=np.zeros((8, 8), dtype=np.float32),
        noisy=noisy,
        recs=recs,
        gammas=gammas,
        label='demo',
        kwargs={'distance_threshold': 2.0, 'func': object()},
        output_filepath=str(tmp_path / 'detections.png'),
    )

    assert compare.call_count == 2
    create_plot.assert_called_once()
    _, filtered_noisy, filtered_recs, filtered_gammas, filtered_ellipses, _, _ = create_plot.call_args.args
    assert filtered_noisy == [noisy[0]]
    assert filtered_recs == [recs[0]]
    assert filtered_gammas == [gammas[0]]
    assert filtered_ellipses == ellipses
