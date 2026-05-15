"""Tests for advanced mast.py paths: async download helpers and bulk-resolution retry."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mast import (
    _chunked,
    _choose_best_product,
    _resolve_products_bulk,
    download_image,
    download_images,
    merge_products_with_metadata,
    plot_histogram,
)


# ---------------------------------------------------------------------------
# _chunked – additional edge cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_chunked_exact_multiple_and_remainder():
    items = list(range(7))
    chunks = list(_chunked(items, 3))
    assert chunks == [[0, 1, 2], [3, 4, 5], [6]]


@pytest.mark.unit
def test_chunked_size_larger_than_list():
    chunks = list(_chunked([1, 2], 10))
    assert chunks == [[1, 2]]


@pytest.mark.unit
def test_chunked_empty_list():
    assert list(_chunked([], 3)) == []


# ---------------------------------------------------------------------------
# _choose_best_product – additional edge cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_choose_best_product_returns_none_for_empty_df():
    assert _choose_best_product(pd.DataFrame()) is None
    assert _choose_best_product(None) is None


@pytest.mark.unit
def test_choose_best_product_no_matching_token_returns_none():
    products = pd.DataFrame({
        'productFilename': ['a_flt.fits', 'b_flt.fits'],
        'productType': ['SCIENCE', 'SCIENCE'],
        'calib_level': [1, 2],
        'dataURI': ['uri-a', 'uri-b'],
    })
    result = _choose_best_product(products, prefer_token='drz')
    assert result is None


@pytest.mark.unit
def test_choose_best_product_prefers_science_over_preview():
    products = pd.DataFrame({
        'productFilename': ['a_drz.fits', 'b_drz.fits'],
        'productType': ['PREVIEW', 'SCIENCE'],
        'calib_level': [5, 3],
        'dataURI': ['uri-preview', 'uri-science'],
    })
    result = _choose_best_product(products, prefer_token='drz')
    assert result is not None
    assert result['dataURI'] == 'uri-science'


# ---------------------------------------------------------------------------
# _resolve_products_bulk – mocked Observations
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_resolve_products_bulk_returns_empty_for_empty_input():
    result = _resolve_products_bulk([])
    assert result == {}


@pytest.mark.unit
def test_resolve_products_bulk_returns_empty_for_blank_ids():
    result = _resolve_products_bulk(['', '  ', None])
    assert result == {}


@pytest.mark.unit
def test_resolve_products_bulk_maps_ids_to_urls(mocker):
    obs_result = MagicMock()
    obs_result.__len__ = MagicMock(return_value=2)

    products = pd.DataFrame({
        'obs_id': ['ID1', 'ID2'],
        'productFilename': ['id1_drz.fits', 'id2_drz.fits'],
        'productType': ['SCIENCE', 'SCIENCE'],
        'calib_level': [3, 3],
        'dataURI': ['mast:HST/id1', 'mast:HST/id2'],
        'dataURL': [None, None],
    })

    products_table = MagicMock()
    products_table.to_pandas.return_value = products
    products_table.__len__ = MagicMock(return_value=len(products))

    mocker.patch('src.data.mast.Observations.query_criteria', return_value=obs_result)
    mocker.patch('src.data.mast.Observations.get_product_list', return_value=products_table)

    result = _resolve_products_bulk(['ID1', 'ID2'], prefer_token='drz')

    assert 'ID1' in result or 'ID2' in result
    for url in result.values():
        assert url.startswith('https://mast.stsci.edu')


@pytest.mark.unit
def test_resolve_products_bulk_retries_on_exception_and_returns_empty(mocker):
    mocker.patch('src.data.mast.Observations.query_criteria', side_effect=RuntimeError('API down'))
    mock_sleep = mocker.patch('src.data.mast.time.sleep')

    result = _resolve_products_bulk(['ID1'], max_retries=2, retry_delay=0.01)

    assert result == {}
    assert mock_sleep.call_count == 2  # sleeps between retries, not on last attempt


@pytest.mark.unit
def test_resolve_products_bulk_returns_empty_when_products_none(mocker):
    obs_result = MagicMock()
    obs_result.__len__ = MagicMock(return_value=1)
    mocker.patch('src.data.mast.Observations.query_criteria', return_value=obs_result)
    mocker.patch('src.data.mast.Observations.get_product_list', return_value=None)

    result = _resolve_products_bulk(['ID1'])
    assert result == {}


@pytest.mark.unit
def test_resolve_products_bulk_returns_empty_when_no_obs(mocker):
    mocker.patch('src.data.mast.Observations.query_criteria', return_value=None)
    result = _resolve_products_bulk(['ID1'])
    assert result == {}


# ---------------------------------------------------------------------------
# merge_products_with_metadata – parallel workers path
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_merge_products_with_metadata_parallel_path(mocker):
    metadata = pd.DataFrame({
        'dataset_id': [f'DS{i}' for i in range(6)],
        'resolved_url': [None] * 6,
    })

    def mock_resolve(ids, **kwargs):
        return {dataset_id: f'https://resolved/{dataset_id}' for dataset_id in ids}

    mocker.patch('src.data.mast._resolve_products_bulk', side_effect=mock_resolve)

    result = merge_products_with_metadata(
        metadata,
        id_column='dataset_id',
        url_column='resolved_url',
        max_workers=3,
        chunk_size=2,
    )

    assert len(result) == 6
    assert result['resolved_url'].notna().all()
    for _, row in result.iterrows():
        assert row['resolved_url'] == f"https://resolved/{row['dataset_id']}"


@pytest.mark.unit
def test_merge_products_with_metadata_empty_table_returns_unchanged():
    empty = pd.DataFrame({'dataset_id': pd.Series(dtype=str)})
    result = merge_products_with_metadata(empty, 'dataset_id', 'url')
    assert result.equals(empty)


@pytest.mark.unit
def test_merge_products_with_metadata_all_already_resolved():
    metadata = pd.DataFrame({
        'dataset_id': ['A', 'B'],
        'resolved_url': ['https://existing/A', 'https://existing/B'],
    })
    result = merge_products_with_metadata(metadata, 'dataset_id', 'resolved_url')
    # No new resolution needed, original URLs must be preserved
    assert result['resolved_url'].tolist() == ['https://existing/A', 'https://existing/B']


# ---------------------------------------------------------------------------
# download_image – async helper (mocked HTTP)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_download_image_skips_existing_file(tmp_path: Path):
    existing = tmp_path / 'existing.fits'
    existing.write_bytes(b'dummy')

    async def run():
        semaphore = asyncio.Semaphore(1)
        result = await download_image('id', 'http://example.com/f.fits', str(tmp_path), None, semaphore, 'existing.fits')
        return result

    assert asyncio.run(run()) is True
    assert existing.stat().st_size == 5  # unchanged


@pytest.mark.unit
def test_download_image_saves_valid_fits_on_200(tmp_path: Path):
    # Build a minimal in-memory FITS file as bytes
    import io
    hdul = fits.HDUList([fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.float32))])
    buf = io.BytesIO()
    hdul.writeto(buf)
    fits_bytes = buf.getvalue()

    response_mock = AsyncMock()
    response_mock.status = 200
    response_mock.read = AsyncMock(return_value=fits_bytes)
    response_mock.__aenter__ = AsyncMock(return_value=response_mock)
    response_mock.__aexit__ = AsyncMock(return_value=False)

    session_mock = MagicMock()
    session_mock.get = MagicMock(return_value=response_mock)

    async def run():
        semaphore = asyncio.Semaphore(1)
        return await download_image('id', 'http://fake/img.fits', str(tmp_path), session_mock, semaphore, 'downloaded.fits')

    result = asyncio.run(run())
    assert result is True
    assert (tmp_path / 'downloaded.fits').exists()


@pytest.mark.unit
def test_download_image_returns_false_on_non_200(tmp_path: Path):
    response_mock = AsyncMock()
    response_mock.status = 404
    response_mock.__aenter__ = AsyncMock(return_value=response_mock)
    response_mock.__aexit__ = AsyncMock(return_value=False)

    session_mock = MagicMock()
    session_mock.get = MagicMock(return_value=response_mock)

    async def run():
        semaphore = asyncio.Semaphore(1)
        return await download_image('id', 'http://fake/missing.fits', str(tmp_path), session_mock, semaphore, 'missing.fits')

    result = asyncio.run(run())
    assert result is False


# ---------------------------------------------------------------------------
# download_images – NaN URL skipping
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_download_images_skips_nan_urls(tmp_path: Path, mocker):
    calls = []

    async def fake_download(id_, url, save_dir, session, semaphore, filename):
        calls.append(url)
        return True

    mocker.patch('src.data.mast.download_image', side_effect=fake_download)

    ids = ['A', 'B', 'C']
    urls = ['http://valid/a.fits', float('nan'), 'http://valid/c.fits']

    asyncio.run(download_images(ids, urls, str(tmp_path), max_requests=2, reset_after=10))

    assert len(calls) == 2
    assert float('nan') not in calls


# ---------------------------------------------------------------------------
# plot_histogram – saves file output
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_plot_histogram_creates_output_file(tmp_path: Path):
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(12345)
    data = pd.Series(rng.exponential(scale=200.0, size=200))
    output_path = str(tmp_path / 'hist.png')

    plot_histogram(data, bins=10, output_filename=output_path)
    plt.close('all')

    assert Path(output_path).exists()
    assert Path(output_path).stat().st_size > 0
