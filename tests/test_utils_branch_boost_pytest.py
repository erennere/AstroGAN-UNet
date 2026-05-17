"""Branch-focused coverage boosts for src/training/utils.py."""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training import utils as utils_mod


@pytest.mark.unit
def test_normalize_runtime_filepath_posix_conversion(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(utils_mod.os, 'name', 'posix', raising=False)
    out = utils_mod._normalize_runtime_filepath(r'D:\data\\img.fits')
    assert out.startswith('/mnt/d/')


@pytest.mark.unit
def test_ensure_directory_exists_handles_exception(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(utils_mod.os, 'makedirs', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('fail')))
    assert utils_mod.ensure_directory_exists('x') is None


@pytest.mark.unit
def test_is_not_nan_handles_type_errors():
    class BadFloat:
        def __float__(self):
            raise TypeError('bad')

    assert utils_mod.is_not_nan(BadFloat()) is False


@pytest.mark.unit
def test_open_fits_missing_named_hdu_returns_none(tmp_path: Path):
    fp = tmp_path / 'x.fits'
    fits.HDUList([fits.PrimaryHDU(data=np.ones((4, 4), dtype=np.float32))]).writeto(fp)
    assert utils_mod.open_fits(str(fp), type_of_image='SCI') is None


@pytest.mark.unit
def test_open_fits_invalid_ratio_and_bad_exptime_paths(tmp_path: Path):
    fp = tmp_path / 'sci.fits'
    hdu = fits.ImageHDU(data=np.ones((4, 4), dtype=np.float32), name='SCI')
    hdu.header['EXPTIME'] = 'not-a-number'
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(fp)

    assert utils_mod.open_fits(str(fp), ratio='oops', type_of_image='SCI') is None
    assert utils_mod.open_fits(str(fp), ratio=2.0, type_of_image='SCI') is None


@pytest.mark.unit
def test_open_fits_low_high_filtering(tmp_path: Path):
    fp = tmp_path / 'sci2.fits'
    hdu = fits.ImageHDU(data=np.ones((4, 4), dtype=np.float32), name='SCI')
    hdu.header['EXPTIME'] = 100.0
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(fp)

    assert utils_mod.open_fits(str(fp), ratio=2.0, type_of_image='SCI', low=250.0) is None
    assert utils_mod.open_fits(str(fp), ratio=2.0, type_of_image='SCI', high=150.0) is None
    ok = utils_mod.open_fits(str(fp), ratio=2.0, type_of_image='SCI', low=10.0, high=250.0)
    assert ok is not None


@pytest.mark.unit
def test_save_fits_returns_none_if_dir_creation_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(utils_mod, 'ensure_directory_exists', lambda *_: None)
    assert utils_mod.save_fits(np.ones((4, 4), dtype=np.float32), 'a.fits', str(tmp_path)) is None


@pytest.mark.unit
def test_read_checkpoint_info_returns_empty_for_non_dict_payload(tmp_path: Path):
    cp = tmp_path / 'model.keras'
    with zipfile.ZipFile(cp, 'w') as zf:
        zf.writestr(utils_mod.CHECKPOINT_INFO_FILENAME, json.dumps([1, 2, 3]))
    assert utils_mod.read_checkpoint_info(str(cp)) == {}


@pytest.mark.unit
def test_restore_model_validates_loader_kwargs_type(tmp_path: Path):
    with pytest.raises(TypeError):
        utils_mod.restore_model(str(tmp_path), 'model', 1, '{prefix}_{epoch:03d}.keras', loader_kwargs='bad')


@pytest.mark.unit
def test_load_model_validation_and_empty_selection_paths(tmp_path: Path):
    with pytest.raises(TypeError):
        utils_mod.load_model(str(tmp_path), False, False, None, restore_kwargs='bad')
    with pytest.raises(ValueError):
        utils_mod.load_model(str(tmp_path), False, False, None, restore_kwargs={})
    with pytest.raises(ValueError):
        utils_mod.load_model(str(tmp_path), True, True, None, restore_kwargs={'filename_pattern': '{prefix}_{epoch}.keras'})

    model, epoch = utils_mod.load_model(
        str(tmp_path),
        start_from_best=False,
        start_from_last=False,
        custom_epoch=None,
        restore_kwargs={'filename_pattern': '{prefix}_{epoch:03d}.keras'},
    )
    assert model is None and epoch == 0


@pytest.mark.unit
def test_filtering_df_raises_without_sort_strategy():
    import pandas as pd

    df = pd.DataFrame(
        {
            'dataset': ['a', 'b'],
            'abs_mean': [1.0, 2.0],
            'full_abs_mean': [1.0, 1.0],
            'crop_abs_mean': [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError):
        utils_mod.filtering_df(df, 1, 0.5, 'dataset', 'abs_mean')


@pytest.mark.unit
def test_candidates_based_on_range_and_ratio_nan_paths():
    import pandas as pd

    kwargs = {
        'name_col': 'name',
        'location_col': 'location',
        'dataset': 'dataset',
        'exposure_col': 'exp',
        'lowest_power': -2,
        'highest_power': 1,
        'n_samples_per_magnitude': 2,
        'ratio_initial': 2,
        'ratio_count': 2,
        'ratio_growth': 1,
        'stats_column_map': {
            'std_bkg': 'std_bkg',
            'abs_mean': 'abs_mean',
            'abs_median': 'abs_median',
            'median_bkg': 'median_bkg',
            'mean_bkg': 'mean_bkg',
            'max_bkg': 'max_bkg',
            'median_src': 'median_src',
            'mean_src': 'mean_src',
            'max_src': 'max_src',
        },
        'original_stats_prefix': 'orig_',
    }

    row = pd.Series(
        {
            'name': 'img',
            'location': 'a.fits',
            'dataset': 'd',
            'exp': np.nan,
            'std_bkg': np.nan,
            'orig_std_bkg': np.nan,
        }
    )

    assert utils_mod.candidates_based_on_range(row, kwargs) == []
    assert utils_mod.candidates_based_on_ratio(row, kwargs) == []


@pytest.mark.unit
def test_resolve_registry_function_rejects_unknown_category():
    with pytest.raises(ValueError):
        utils_mod.resolve_registry_function('unknown_category', 'x')


@pytest.mark.unit
def test_normalize_runtime_filepath_backslash_path_on_posix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(utils_mod.os, 'name', 'posix', raising=False)
    out = utils_mod._normalize_runtime_filepath('folder\\sub\\file.fits')
    assert out == 'folder/sub/file.fits'


@pytest.mark.unit
def test_ensure_parent_dir_exists_none_and_empty_parent(monkeypatch: pytest.MonkeyPatch):
    assert utils_mod.ensure_parent_dir_exists(None) is None
    monkeypatch.setattr(utils_mod.os.path, 'abspath', lambda *_: 'x')
    monkeypatch.setattr(utils_mod.os.path, 'dirname', lambda *_: '')
    assert utils_mod.ensure_parent_dir_exists('x') == ''


@pytest.mark.unit
def test_open_fits_skips_non_image_hdus_and_returns_none(tmp_path: Path):
    fp = tmp_path / 'table_only.fits'
    col = fits.Column(name='x', format='E', array=np.array([1.0], dtype=np.float32))
    table_hdu = fits.BinTableHDU.from_columns([col])
    fits.HDUList([fits.PrimaryHDU(), table_hdu]).writeto(fp)

    assert utils_mod.open_fits(str(fp), ratio=2.0, type_of_image='SCI') is None


@pytest.mark.unit
def test_save_fits_exception_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def _boom(*args, **kwargs):
        raise OSError('write fail')

    monkeypatch.setattr('astropy.io.fits.HDUList.writeto', _boom)
    out = utils_mod.save_fits(np.ones((2, 2), dtype=np.float32), 'x.fits', str(tmp_path))
    assert out is None


@pytest.mark.unit
def test_restore_model_handles_loader_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cp = tmp_path / 'model_001.keras'
    cp.write_text('dummy', encoding='utf-8')

    monkeypatch.setattr(utils_mod, 'load_checkpoint_model', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('boom')))

    restored = utils_mod.restore_model(
        str(tmp_path),
        checkpoint_prefix='model',
        epoch=1,
        filename_pattern='{prefix}_{epoch:03d}.keras',
        loader_kwargs={},
    )
    assert restored is None


@pytest.mark.unit
def test_resolve_registry_function_additional_unsupported_names():
    with pytest.raises(ValueError):
        utils_mod.resolve_registry_function('post_filter_fn', 'nope')
    with pytest.raises(ValueError):
        utils_mod.resolve_registry_function('sample_fn', 'nope')
    with pytest.raises(ValueError):
        utils_mod.resolve_registry_function('noise_fn', 'nope')
    with pytest.raises(ValueError):
        utils_mod.resolve_registry_function('sigma_kernel_fn', 'nope')
    with pytest.raises(ValueError):
        utils_mod.resolve_registry_function('stats_name_fn', 'nope')


@pytest.mark.unit
def test_resolve_registry_function_noise_registry_success():
    fn = utils_mod.resolve_registry_function('noise_fn', '_simulated_image_from_poisson')
    assert callable(fn)
