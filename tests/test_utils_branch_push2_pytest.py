from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from astropy.io import fits



REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training import utils as ut


@pytest.mark.unit
def test_restore_model_missing_success_and_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    missing = ut.restore_model(str(tmp_path), 'model', 3, '{prefix}_{epoch:03d}.keras', {})
    assert missing is None

    file_path = tmp_path / 'model_003.keras'
    file_path.write_text('x', encoding='utf-8')

    monkeypatch.setattr(ut, 'load_checkpoint_model', lambda *args, **kwargs: 'loaded')
    ok = ut.restore_model(str(tmp_path), 'model', 3, '{prefix}_{epoch:03d}.keras', {})
    assert ok == ('loaded', 3)

    monkeypatch.setattr(ut, 'load_checkpoint_model', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('bad load')))
    err = ut.restore_model(str(tmp_path), 'model', 3, '{prefix}_{epoch:03d}.keras', {})
    assert err is None


@pytest.mark.unit
def test_load_model_additional_restore_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(TypeError):
        ut.load_model(str(tmp_path), False, False, None, restore_kwargs='bad')
    with pytest.raises(ValueError):
        ut.load_model(str(tmp_path), False, False, None, restore_kwargs={})

    assert ut.load_model(str(tmp_path), False, False, None, restore_kwargs={'filename_pattern': '{prefix}_{epoch:03d}.keras'}) == (None, 0)

    ck = tmp_path / 'ck'
    ck.mkdir()
    (ck / 'best_model_005.keras').write_text('x', encoding='utf-8')
    (ck / 'model_007.keras').write_text('x', encoding='utf-8')

    calls = []

    def fake_restore(checkpoint_dir, checkpoint_prefix, epoch, filename_pattern, loader_kwargs):
        calls.append((checkpoint_prefix, epoch))
        if checkpoint_prefix == 'model' and epoch == 7:
            return ('m7', 7)
        return None

    monkeypatch.setattr(ut, 'restore_model', fake_restore)

    out_custom_regular = ut.load_model(
        str(ck),
        start_from_best=False,
        start_from_last=True,
        custom_epoch=7,
        restore_kwargs={'filename_pattern': '{prefix}_{epoch:03d}.keras', 'compile': False},
    )
    assert out_custom_regular == ('m7', 7)
    assert ('model', 7) in calls

    calls.clear()

    def fake_restore_best(checkpoint_dir, checkpoint_prefix, epoch, filename_pattern, loader_kwargs):
        calls.append((checkpoint_prefix, epoch))
        if checkpoint_prefix == 'best_model' and epoch == 5:
            return ('b5', 5)
        return None

    monkeypatch.setattr(ut, 'restore_model', fake_restore_best)
    out_best = ut.load_model(
        str(ck),
        start_from_best=True,
        start_from_last=False,
        custom_epoch=None,
        restore_kwargs={'filename_pattern': '{prefix}_{epoch:03d}.keras'},
    )
    assert out_best == ('b5', 5)
    assert calls[0] == ('best_model', 5)

    monkeypatch.setattr(ut, 'restore_model', lambda *args, **kwargs: None)
    out_none = ut.load_model(
        str(ck),
        start_from_best=False,
        start_from_last=True,
        custom_epoch=999,
        restore_kwargs={'filename_pattern': '{prefix}_{epoch:03d}.keras'},
    )
    assert out_none == (None, 0)


@pytest.mark.unit
def test_create_tf_dataset_log_min_max_and_augment_lines(monkeypatch: pytest.MonkeyPatch):
    class FakeDataset:
        def __init__(self):
            self.mapped = False
            self.batched = False
            self.prefetched = False

        def map(self, fn, num_parallel_calls=None):
            noisy = tf.ones((4, 4, 1), dtype=tf.float32)
            clean = tf.ones((4, 4, 1), dtype=tf.float32)
            meta = tf.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
            fn(noisy, clean, meta)
            self.mapped = True
            return self

        def batch(self, batch_size):
            self.batched = True
            return self

        def prefetch(self, value):
            self.prefetched = True
            return self

    monkeypatch.setattr(ut.tf.data.Dataset, 'from_generator', lambda *args, **kwargs: FakeDataset())

    def fake_gen(images, kwargs_data, scaling):
        yield np.ones((4, 4, 1), dtype=np.float32), np.ones((4, 4, 1), dtype=np.float32), np.array(['a'] * 7, dtype=object)

    monkeypatch.setattr(ut.tf.random, 'uniform', lambda *args, **kwargs: tf.constant(1 if kwargs.get('dtype') == tf.int32 else 0.9))

    ds = ut.create_tf_dataset(
        images=['x'],
        sample_generator=fake_gen,
        generator_kwargs={'ps': 4},
        batch_size=1,
        scaling='log_min_max',
        augment=True,
    )
    assert ds.mapped is True
    assert ds.batched is True
    assert ds.prefetched is True


@pytest.mark.unit
def test_filtering_df_v2_validation_edges_and_backfill_paths():
    df = pd.DataFrame({'a': [1.0, 0.9], 'b': [1.0, 0.9], 'c': [1.0, 1.0], 'd': ['x', 'y']})

    with pytest.raises(ValueError):
        ut.filtering_df_v2(df, -1, 'a', 'b', 'c', 'd', 1, quantiles=(100,), percentages=(1,))
    with pytest.raises(ValueError):
        ut.filtering_df_v2(df, 1, 'a', 'b', 'c', 'd', -1, quantiles=(100,), percentages=(1,))
    with pytest.raises(ValueError):
        ut.filtering_df_v2(df, 1, 'a', 'b', 'c', 'd', 1, quantiles=(), percentages=())
    with pytest.raises(ValueError):
        ut.filtering_df_v2(df, 1, 'a', 'b', 'c', 'd', 1, quantiles=(50, 100), percentages=(1,))
    with pytest.raises(ValueError):
        ut.filtering_df_v2(df, 1, 'a', 'b', 'c', 'd', 1, quantiles=(50, 90), percentages=(1, 1))
    with pytest.raises(ValueError):
        ut.filtering_df_v2(df, 1, 'a', 'b', 'c', 'd', 1, quantiles=(100,), percentages=(0,))


@pytest.mark.unit
def test_filtering_df_groupby_accumulation_branch():
    df = pd.DataFrame(
        {
            'dataset': ['A', 'A', 'A', 'A', 'A', 'A'],
            'abs_mean': [10, 9, 8, 7, 6, 5],
            'noise_ratio': [1, 1, 1, 1, 1, 1],
            'full_abs_mean': [10, 10, 10, 10, 10, 10],
            'crop_abs_mean': [9, 9, 9, 9, 9, 9],
        }
    )
    out = ut.filtering_df(df.copy(), n_samples=5, delta=0.5, id_='dataset', abs_='abs_mean', noise_ratio='noise_ratio')
    assert len(out) <= 5


@pytest.mark.unit
def test_candidates_based_on_range_nan_entry_continue_and_ratio_nan_exposure_return():
    scm = {
        'std_bkg': 'std_bkg',
        'abs_mean': 'abs_mean',
        'abs_median': 'abs_median',
        'median_bkg': 'median_bkg',
        'mean_bkg': 'mean_bkg',
        'max_bkg': 'max_bkg',
        'median_src': 'median_src',
        'mean_src': 'mean_src',
        'max_src': 'max_src',
    }
    kwargs = {
        'name_col': 'name',
        'location_col': 'location',
        'dataset': 'dataset',
        'exposure_col': 'exp_time',
        'lowest_power': -1,
        'highest_power': 1,
        'n_samples_per_magnitude': 2,
        'stats_column_map': scm,
        'original_stats_prefix': 'orig_',
        'ratio_initial': 2.0,
        'ratio_count': 2,
        'ratio_growth': 2.0,
    }

    row = pd.Series(
        {
            'name': 'img', 'location': '/tmp/img.fits', 'dataset': 'D', 'exp_time': 100.0,
            'std_bkg': 2.0, 'abs_mean': 5.0, 'abs_median': 5.0,
            'median_bkg': 1.0, 'mean_bkg': 1.0, 'max_bkg': 2.0,
            'median_src': 3.0, 'mean_src': 3.0, 'max_src': 4.0,
            'orig_std_bkg': 2.0, 'orig_abs_mean': 5.0, 'orig_abs_median': 5.0,
            'orig_median_bkg': 1.0, 'orig_mean_bkg': 1.0, 'orig_max_bkg': 2.0,
            'orig_median_src': 3.0, 'orig_mean_src': 3.0, 'orig_max_src': 4.0,
        }
    )

    original_aug = ut._augment_samples_based_on_range
    ut._augment_samples_based_on_range = lambda *args, **kwargs: [(np.nan, 1, 0, 0), (3.0, 3, 0, 0)]
    try:
        out = ut.candidates_based_on_range(row, kwargs)
    finally:
        ut._augment_samples_based_on_range = original_aug
    assert len(out) == 1

    row_nan_exp = row.copy()
    row_nan_exp['exp_time'] = np.nan
    out_ratio = ut.candidates_based_on_ratio(row_nan_exp, kwargs)
    assert out_ratio == []


@pytest.mark.unit
def test_open_fits_exception_and_create_tf_dataset_log_metadata_shape(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ut.fits, 'open', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('open fail')))
    assert ut.open_fits('missing.fits') is None

    def sample_generator(images, kwargs_data, scaling):
        noisy = np.ones((2, 2, 1), dtype=np.float32)
        clean = np.ones((2, 2, 1), dtype=np.float32)
        metadata = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g'], dtype=object)
        yield noisy, clean, metadata

    ds = ut.create_tf_dataset(
        images=['x'],
        sample_generator=sample_generator,
        generator_kwargs={'ps': 2},
        batch_size=1,
        scaling='log_min_max',
        augment=False,
    )
    first_batch = next(iter(ds))
    assert first_batch[2].shape[-1] == 7


@pytest.mark.unit
def test_read_checkpoint_info_non_dict_and_load_model_pattern_validation(tmp_path: Path):
    ckpt = tmp_path / 'm.keras'
    import zipfile
    with zipfile.ZipFile(ckpt, 'w') as zf:
        zf.writestr('checkpoint_info.json', '[1,2,3]')

    assert ut.read_checkpoint_info(str(ckpt)) == {}

    ck_dir = tmp_path / 'ckp'
    ck_dir.mkdir()
    (ck_dir / 'model_001.keras').write_text('x', encoding='utf-8')

    with pytest.raises(TypeError):
        ut.load_model(
            str(ck_dir),
            start_from_best=True,
            start_from_last=False,
            custom_epoch=None,
            restore_kwargs={'filename_pattern': 123},
        )

    with pytest.raises(ValueError):
        ut.load_model(
            str(ck_dir),
            start_from_best=True,
            start_from_last=False,
            custom_epoch=None,
            restore_kwargs={'filename_pattern': 'model_only_{epoch:03d}.keras'},
        )

    missing_dir_out = ut.load_model(
        str(tmp_path / 'does_not_exist'),
        start_from_best=True,
        start_from_last=False,
        custom_epoch=None,
        restore_kwargs={'filename_pattern': '{prefix}_{epoch:03d}.keras'},
    )
    assert missing_dir_out == (None, 0)


@pytest.mark.unit
def test_filtering_df_after_relevance_second_early_return():
    df = pd.DataFrame(
        {
            'dataset': ['A', 'B', 'C', 'D'],
            'abs_mean': [10.0, 9.0, 8.0, 7.0],
            'noise_ratio': [1.0, 1.0, 1.0, 1.0],
            'full_abs_mean': [100.0, 100.0, 100.0, 100.0],
            'crop_abs_mean': [80.0, 10.0, 10.0, 10.0],
        }
    )
    out = ut.filtering_df(df.copy(), n_samples=2, delta=0.5, id_='dataset', abs_='abs_mean', noise_ratio='noise_ratio')
    assert len(out) == 1


@pytest.mark.unit
def test_normalize_runtime_filepath_passthrough_posix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ut.os, 'name', 'posix', raising=False)
    path = 'folder/sub/file.fits'
    assert ut._normalize_runtime_filepath(path) == path


@pytest.mark.unit
def test_open_fits_invalid_exptime_type_branch(tmp_path: Path):
    fp = tmp_path / 'bad_exptime_type.fits'
    hdu = fits.ImageHDU(data=np.ones((4, 4), dtype=np.float32), name='SCI')
    hdu.header['EXPTIME'] = '[1,2]'
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(fp)

    # Force a non-numeric/non-string EXPTIME after file open.
    with fits.open(fp, mode='update') as hdul:
        hdul['SCI'].header['EXPTIME'] = 1
    monkeypatch = pytest.MonkeyPatch()
    original_open = ut.fits.open

    def fake_open(*args, **kwargs):
        hdul = original_open(*args, **kwargs)
        hdul['SCI'].header['EXPTIME'] = {'bad': 'type'}
        return hdul

    monkeypatch.setattr(ut.fits, 'open', fake_open)
    try:
        assert ut.open_fits(str(fp), ratio=1.0, type_of_image='SCI') is None
    finally:
        monkeypatch.undo()


@pytest.mark.unit
def test_open_fits_invalid_exptime_type_with_fake_hdu(monkeypatch: pytest.MonkeyPatch):
    class FakeHDU:
        def __init__(self):
            self.data = np.ones((2, 2), dtype=np.float32)
            self.name = 'SCI'
            self.header = {'EXPTIME': object()}

    class FakeOpen:
        def __enter__(self):
            return [FakeHDU()]

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(ut.fits, 'ImageHDU', FakeHDU)
    monkeypatch.setattr(ut.fits, 'PrimaryHDU', FakeHDU)
    monkeypatch.setattr(ut.fits, 'CompImageHDU', FakeHDU)
    monkeypatch.setattr(ut.fits, 'open', lambda *args, **kwargs: FakeOpen())
    assert ut.open_fits('fake.fits', ratio=1.0, type_of_image='SCI') is None


@pytest.mark.unit
def test_save_checkpoint_model_early_return_when_not_zip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    class DummyModel:
        def save(self, path):
            Path(path).write_text('not-a-keras-zip', encoding='utf-8')

    cp = tmp_path / 'plain_file.keras'
    out = ut.save_checkpoint_model(DummyModel(), str(cp), checkpoint_info={'a': 1})
    assert out is None


@pytest.mark.unit
def test_create_tf_dataset_generator_yield_from_path(monkeypatch: pytest.MonkeyPatch):
    called = {'n': 0}

    def sample_generator(images, kwargs_data, scaling):
        called['n'] += 1
        yield (
            np.ones((2, 2, 1), dtype=np.float32),
            np.ones((2, 2, 1), dtype=np.float32),
            np.array(['a'], dtype=object),
        )

    ds = ut.create_tf_dataset(
        images=['x'],
        sample_generator=sample_generator,
        generator_kwargs={'ps': 2},
        batch_size=1,
        scaling=None,
        augment=False,
    )
    _ = next(iter(ds))
    assert called['n'] >= 1


@pytest.mark.unit
def test_create_tf_dataset_generator_wrapper_invoked(monkeypatch: pytest.MonkeyPatch):
    called = {'n': 0}

    class FakeDataset:
        def batch(self, _batch_size):
            return self

        def prefetch(self, _value):
            return self

    def fake_from_generator(generator, output_signature):
        _ = list(generator())
        return FakeDataset()

    def sample_generator(images, kwargs_data, scaling):
        called['n'] += 1
        yield (
            np.ones((2, 2, 1), dtype=np.float32),
            np.ones((2, 2, 1), dtype=np.float32),
            np.array(['a'], dtype=object),
        )

    monkeypatch.setattr(ut.tf.data.Dataset, 'from_generator', fake_from_generator)
    ds = ut.create_tf_dataset(
        images=['x'],
        sample_generator=sample_generator,
        generator_kwargs={'ps': 2},
        batch_size=1,
        scaling=None,
        augment=False,
    )
    assert isinstance(ds, FakeDataset)
    assert called['n'] == 1


@pytest.mark.unit
def test_filtering_df_append_and_data_empty_break_branch():
    df = pd.DataFrame(
        {
            'dataset': ['A', 'A', 'A', 'A'],
            'abs_mean': [10, 9, 8, 7],
            'noise_ratio': [1, 1, 1, 1],
            'full_abs_mean': [10, 10, 10, 10],
            'crop_abs_mean': [9, 9, 9, 9],
        }
    )
    out = ut.filtering_df(df.copy(), n_samples=3, delta=0.5, id_='dataset', abs_='abs_mean', noise_ratio='noise_ratio')
    assert len(out) <= 3


@pytest.mark.unit
def test_filtering_df_loop_append_and_update_branch_with_duplicate_abs():
    df = pd.DataFrame(
        {
            'dataset': ['A'] * 5 + ['B'] * 5,
            'abs_mean': [10.0] * 10,
            'noise_ratio': [1.0] * 10,
            'full_abs_mean': [10.0] * 10,
            'crop_abs_mean': [9.0] * 10,
        }
    )
    out = ut.filtering_df(df.copy(), n_samples=5, delta=0.5, id_='dataset', abs_='abs_mean', noise_ratio='noise_ratio')
    assert len(out) <= 5


@pytest.mark.unit
def test_filtering_df_v2_remaining_needed_and_no_remainder_branches():
    df = pd.DataFrame(
        {
            'a': [5.0, 4.0, 3.0],
            'b': [5.0, 4.0, 3.0],
            'c': [1.0, 1.0, 1.0],
            'd': ['x', 'y', 'z'],
        }
    )

    # remaining_needed <= 0 branch
    out_full = ut.filtering_df_v2(
        df,
        n_samples=2,
        col_A='a',
        col_B='b',
        col_C='c',
        col_D='d',
        occurrences_per_col_D=1,
        quantiles=(100,),
        percentages=(1,),
    )
    assert len(out_full) == 2

    # not remainder_positions branch (all selected in stage 1)
    out_no_remainder = ut.filtering_df_v2(
        df,
        n_samples=3,
        col_A='a',
        col_B='b',
        col_C='c',
        col_D='d',
        occurrences_per_col_D=1,
        quantiles=(100,),
        percentages=(1,),
    )
    assert len(out_no_remainder) == 3
