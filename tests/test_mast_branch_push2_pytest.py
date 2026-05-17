"""Additional branch coverage tests for src/data/mast.py."""
from __future__ import annotations

import sys
import asyncio
import runpy
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import mast


@pytest.mark.unit
def test_filter_out_mast_handles_empty_missions_and_none_page(monkeypatch: pytest.MonkeyPatch):
    class NoMission:
        def __bool__(self):
            return False

    monkeypatch.setattr(mast, 'MastMissions', lambda mission: NoMission())
    assert mast.filter_out_mast('HST', {'a': 1}) is None

    class MissionNonePage:
        def __bool__(self):
            return True

        def get_column_list(self):
            return {'name': ['a']}

        def query_criteria(self, **kwargs):
            return None

    monkeypatch.setattr(mast, 'MastMissions', lambda mission: MissionNonePage())
    assert mast.filter_out_mast('HST', {'a': 1}) is None


@pytest.mark.unit
def test_filter_out_mast_handles_exception(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mast, 'MastMissions', lambda mission: (_ for _ in ()).throw(RuntimeError('boom')))
    assert mast.filter_out_mast('HST', {'a': 1}) is None


@pytest.mark.unit
def test_main_missing_keys_and_no_metadata_return(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(mast, 'parse_config_overrides', lambda *args, **kwargs: {})

    monkeypatch.setattr(mast, 'load_config', lambda **kwargs: {'mast': {'mission': 'HST'}})
    mast.main()

    cfg = {
        'mission': 'HST',
        'filters': {'exp': {'column': 'exp', 'conditions': {'greater': 1}}},
        'main_column': 'exp',
        'max_requests': 2,
        'reset_after': 10,
        'max_workers': 1,
        'download': False,
        'chunk_size': 10,
        'resolve_max_retries': 1,
        'resolve_retry_delay': 0,
        'metadata_output': str(tmp_path / 'm.csv'),
        'id_column': 'obsid',
        'url_column': 'url',
        'fetch_metadata': True,
        'resolve_urls': False,
        'prefer_token': 'drz',
        'save_dir': str(tmp_path / 'dl'),
    }
    monkeypatch.setattr(mast, 'load_config', lambda **kwargs: {'mast': cfg})
    monkeypatch.setattr(mast, 'filter_out_mast', lambda *args, **kwargs: pd.DataFrame())
    mast.main()


@pytest.mark.unit
def test_main_loads_metadata_from_disk_and_merges(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    mpath = tmp_path / 'metadata.csv'
    pd.DataFrame({'obsid': ['A'], 'url': [None], 'exp': [1.0]}).to_csv(mpath, index=False)

    cfg = {
        'mission': 'HST',
        'filters': {'exp': {'column': 'exp', 'conditions': {'greater': 1}}},
        'main_column': 'exp',
        'max_requests': 2,
        'reset_after': 10,
        'max_workers': 1,
        'download': False,
        'chunk_size': 10,
        'resolve_max_retries': 1,
        'resolve_retry_delay': 0,
        'metadata_output': str(mpath),
        'id_column': 'obsid',
        'url_column': 'url',
        'fetch_metadata': False,
        'resolve_urls': True,
        'prefer_token': 'drz',
        'save_dir': str(tmp_path / 'dl'),
    }

    monkeypatch.setattr(mast, 'parse_config_overrides', lambda *args, **kwargs: {})
    monkeypatch.setattr(mast, 'load_config', lambda **kwargs: {'mast': cfg})
    monkeypatch.setattr(mast, 'merge_products_with_metadata', lambda table, **kwargs: table.assign(url='https://x'))

    mast.main()
    out = pd.read_csv(mpath)
    assert out['url'].iloc[0] == 'https://x'


@pytest.mark.unit
def test_filter_out_mast_result_not_sized_path(monkeypatch: pytest.MonkeyPatch):
    class MissionNotSized:
        def __bool__(self):
            return True

        def get_column_list(self):
            return {'name': ['a']}

        def query_criteria(self, **kwargs):
            return object()

    monkeypatch.setattr(mast, 'MastMissions', lambda mission: MissionNotSized())
    assert mast.filter_out_mast('HST', {'a': 1}) is None


@pytest.mark.unit
def test_chunked_non_positive_size_branch():
    assert list(mast._chunked([1, 2, 3], 0)) == [[1], [2], [3]]


@pytest.mark.unit
def test_choose_best_product_without_filename_column():
    products = pd.DataFrame({
        'productType': ['PREVIEW', 'SCIENCE'],
        'calib_level': [1, 5],
        'dataURI': ['uri-a', 'uri-b'],
    })
    best = mast._choose_best_product(products)
    assert best is not None
    assert best['dataURI'] == 'uri-b'


@pytest.mark.unit
def test_resolve_products_bulk_key_column_missing(monkeypatch: pytest.MonkeyPatch):
    class Products:
        def to_pandas(self):
            return pd.DataFrame({'productFilename': ['x_drz.fits'], 'dataURI': ['mast:x']})

    monkeypatch.setattr(mast.Observations, 'query_criteria', lambda **kwargs: [1])
    monkeypatch.setattr(mast.Observations, 'get_product_list', lambda obs: Products())
    assert mast._resolve_products_bulk(['ID1']) == {}


@pytest.mark.unit
def test_resolve_products_bulk_fallback_to_data_url(monkeypatch: pytest.MonkeyPatch):
    class Products:
        def to_pandas(self):
            return pd.DataFrame(
                {
                    'obs_id': ['ID1'],
                    'productFilename': ['id1_drz.fits'],
                    'productType': ['SCIENCE'],
                    'calib_level': [1],
                    'dataURI': [None],
                    'dataURL': ['https://example/id1.fits'],
                }
            )

    monkeypatch.setattr(mast.Observations, 'query_criteria', lambda **kwargs: [1])
    monkeypatch.setattr(mast.Observations, 'get_product_list', lambda obs: Products())
    assert mast._resolve_products_bulk(['ID1']) == {'ID1': 'https://example/id1.fits'}


@pytest.mark.unit
def test_merge_products_with_metadata_missing_id_column_returns_original():
    frame = pd.DataFrame({'url': [None]})
    out = mast.merge_products_with_metadata(frame, id_column='dataset_id', url_column='url')
    assert out.equals(frame)


@pytest.mark.unit
def test_main_resolve_urls_missing_metadata_csv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = {
        'mission': 'HST',
        'filters': {'exp': {'column': 'exp', 'conditions': {'greater': 1}}},
        'main_column': 'exp',
        'max_requests': 2,
        'reset_after': 10,
        'max_workers': 1,
        'download': False,
        'chunk_size': 10,
        'resolve_max_retries': 1,
        'resolve_retry_delay': 0,
        'metadata_output': str(tmp_path / 'does_not_exist.csv'),
        'id_column': 'obsid',
        'url_column': 'url',
        'fetch_metadata': False,
        'resolve_urls': True,
        'prefer_token': 'drz',
        'save_dir': str(tmp_path / 'dl'),
    }
    monkeypatch.setattr(mast, 'parse_config_overrides', lambda *args, **kwargs: {})
    monkeypatch.setattr(mast, 'load_config', lambda **kwargs: {'mast': cfg})
    mast.main()


@pytest.mark.unit
def test_download_image_creates_session_when_missing_and_download_images_resets_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    class FakeResponse:
        status = 404

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeSession:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def get(self, _url):
            return FakeResponse()

        async def close(self):
            self.closed = True

    created = []

    def make_session(*args, **kwargs):
        session = FakeSession()
        created.append(session)
        return session

    monkeypatch.setattr(mast.aiohttp, 'ClientSession', make_session)

    async def run_download_image():
        sem = asyncio.Semaphore(1)
        return await mast.download_image('id1', 'https://x/1.fits', str(tmp_path), None, sem, 'a.fits')

    assert asyncio.run(run_download_image()) is False
    assert len(created) >= 1

    async def fake_download_image(*args, **kwargs):
        return False

    monkeypatch.setattr(mast, 'download_image', fake_download_image)
    asyncio.run(mast.download_images(['a', 'b'], ['https://x/a.fits', 'https://x/b.fits'], str(tmp_path), max_requests=1, reset_after=1))
    assert len(created) >= 3


@pytest.mark.unit
def test_download_image_exception_and_download_images_continue_and_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    class BoomResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def read(self):
            raise RuntimeError('read-fail')

    class SessionForException:
        def get(self, _url):
            return BoomResponse()

        async def close(self):
            return None

    async def run_download_image_exception():
        sem = asyncio.Semaphore(1)
        return await mast.download_image('id1', 'https://x/1.fits', str(tmp_path), SessionForException(), sem, 'boom.fits')

    assert asyncio.run(run_download_image_exception()) is False

    async def boom_download(*args, **kwargs):
        raise RuntimeError('boom-download-images')

    monkeypatch.setattr(mast, 'download_image', boom_download)
    asyncio.run(
        mast.download_images(
            ['a', 'b', 'c'],
            ['', 'https://x/b.fits', 'https://x/c.fits'],
            str(tmp_path),
            max_requests=1,
            reset_after=10,
        )
    )


@pytest.mark.unit
def test_resolve_products_bulk_empty_filtered_and_best_none(monkeypatch: pytest.MonkeyPatch):
    class EmptyProducts:
        def to_pandas(self):
            return pd.DataFrame()

    monkeypatch.setattr(mast.Observations, 'query_criteria', lambda **kwargs: [1])
    monkeypatch.setattr(mast.Observations, 'get_product_list', lambda obs: EmptyProducts())
    assert mast._resolve_products_bulk(['ID1']) == {}

    class MismatchProducts:
        def to_pandas(self):
            return pd.DataFrame(
                {
                    'obs_id': ['NOT_ID1'],
                    'productFilename': ['id1_drz.fits'],
                    'productType': ['SCIENCE'],
                    'calib_level': [1],
                    'dataURI': ['mast:x'],
                }
            )

    monkeypatch.setattr(mast.Observations, 'get_product_list', lambda obs: MismatchProducts())
    assert mast._resolve_products_bulk(['ID1']) == {}

    class BestNoneProducts:
        def to_pandas(self):
            return pd.DataFrame(
                {
                    'obs_id': ['ID1'],
                    'productFilename': ['id1_flt.fits'],
                    'productType': ['SCIENCE'],
                    'calib_level': [1],
                    'dataURI': ['mast:x'],
                }
            )

    monkeypatch.setattr(mast.Observations, 'get_product_list', lambda obs: BestNoneProducts())
    assert mast._resolve_products_bulk(['ID1']) == {}


@pytest.mark.unit
def test_merge_products_with_metadata_adds_missing_url_column(monkeypatch: pytest.MonkeyPatch):
    frame = pd.DataFrame({'dataset_id': ['A']})
    monkeypatch.setattr(mast, '_resolve_products_bulk', lambda *args, **kwargs: {})
    out = mast.merge_products_with_metadata(frame, id_column='dataset_id', url_column='resolved_url', max_workers=1)
    assert 'resolved_url' in out.columns


@pytest.mark.unit
def test_main_missing_mast_section_and_module_main_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(mast, 'parse_config_overrides', lambda *args, **kwargs: {})
    monkeypatch.setattr(mast, 'load_config', lambda **kwargs: {})
    mast.main()

    import starter

    cfg = {
        'mission': 'HST',
        'filters': {},
        'main_column': 'exp',
        'max_requests': 1,
        'reset_after': 1,
        'max_workers': 1,
        'download': False,
        'chunk_size': 10,
        'resolve_max_retries': 1,
        'resolve_retry_delay': 0,
        'metadata_output': str(tmp_path / 'module_main.csv'),
        'id_column': 'obsid',
        'url_column': 'url',
        'fetch_metadata': False,
        'resolve_urls': False,
        'prefer_token': 'drz',
        'save_dir': str(tmp_path / 'dl'),
    }

    monkeypatch.setattr(starter, 'parse_config_overrides', lambda *args, **kwargs: {})
    monkeypatch.setattr(starter, 'load_config', lambda **kwargs: {'mast': cfg})
    runpy.run_module('src.data.mast', run_name='__main__')
