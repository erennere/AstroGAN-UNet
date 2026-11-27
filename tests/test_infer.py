"""astrounets denoise: tiling, blending, no-data handling, files, the default (final) model."""
import json
import logging

import keras
import numpy as np
import pytest
from astropy.io import fits

from astrounets.config import replace, save_config
from astrounets.data.mock import add_dead_patch, make_image
from astrounets.infer import TrainedModel, blend_window, denoise_file, find_images, tile_starts
from conftest import SMOKE, smoke_config

PS = 64  # smoke patch size


def _identity(ps=PS):
    inp = keras.Input((ps, ps, 1))
    return keras.Model(inp, keras.layers.Activation("linear")(inp))


def _random_model(ps=PS):
    keras.utils.set_random_seed(3)
    inp = keras.Input((ps, ps, 1))
    x = keras.layers.Conv2D(4, 3, padding="same", activation="tanh")(inp)
    return keras.Model(inp, keras.layers.Add()([inp, keras.layers.Conv2D(1, 3, padding="same")(x)]))


def _model_folder(cfg, model, folder=None, layout="run"):
    """A trained-model folder: a run folder (checkpoints/) or a report copy (flat)."""
    folder = folder or cfg.run_dir
    ckpt = folder / "checkpoints" if layout == "run" else folder
    ckpt.mkdir(parents=True, exist_ok=True)
    save_config(cfg, folder / "config.yaml")
    model.save(ckpt / "best.keras")
    (ckpt / "best.json").write_text(json.dumps({"epoch": 7, "val_rec_loss": 0.1}))
    return folder


def _image(shape, seed=0, weight=False):
    """A mock sky cut to `shape`, with a drizzle border, NaN pixels and (weight) a low-weight disc."""
    rng = np.random.default_rng(seed)
    size = max(max(shape), 32) + 8
    image, sigma, _ = make_image(rng, size, 1000.0, n_sources=max(3, size * size // 800))
    w = np.where(np.isfinite(image) & (image != 0), 1000.0, 0.0).astype(np.float32)
    if weight:
        image, w = add_dead_patch(rng, image, w, sigma)
    y, x = (size - shape[0]) // 2, (size - shape[1]) // 2  # the middle: small cuts hold sky, large ones the border
    return image[y:y + shape[0], x:x + shape[1]].copy(), w[y:y + shape[0], x:x + shape[1]].copy()


@pytest.mark.parametrize("n, size, stride", [(1, 64, 48), (64, 64, 48), (65, 64, 48), (300, 64, 48), (300, 64, 64), (129, 64, 32)])
def test_tiles_cover_every_pixel_and_end_flush(n, size, stride):
    starts = tile_starts(max(n, size), size, stride)
    covered = np.zeros(max(n, size), bool)
    for s in starts:
        covered[s:s + size] = True
    assert covered.all() and starts[0] == 0 and starts[-1] == max(n, size) - size
    assert all(0 < b - a <= stride for a, b in zip(starts, starts[1:]))


def test_blend_weights_add_up_to_one_where_tiles_overlap():
    w = blend_window(64, 16)[32]  # one row through the middle
    assert (w > 0).all() and w[16:48].tolist() == [1.0] * 32
    np.testing.assert_allclose(w[48:] + w[:16], 1.0, atol=1e-6)  # next tile starts 48 pixels on
    assert (blend_window(64, 0) == 1).all()


@pytest.mark.parametrize("mode", ["none", "z_scale", "robust_z", "min_max", "log_min_max"])
def test_identity_model_returns_the_input_at_any_size(tmp_path, mode):
    """Tiling, per-tile scaling, mirroring and blending add nothing: an identity model gives the input back."""
    cfg = replace(smoke_config(tmp_path), {"scaling": {"mode": mode}})
    m = TrainedModel.load(_model_folder(cfg, _identity()))
    for shape, overlap in [((1, 1), 16), ((30, 200), 16), ((64, 64), 0), ((64, 64), 32), ((65, 150), 16), ((173, 130), 8)]:
        image, weight = _image(shape, seed=shape[0])
        rec = m.denoise(image, weight, overlap=overlap, batch_size=5)
        assert rec.shape == image.shape
        np.testing.assert_allclose(rec, image, rtol=2e-5, atol=1e-6, equal_nan=True, err_msg=f"{shape} {overlap}")
    rec = m.denoise(image, weight, overlap=16, tta=True)
    np.testing.assert_allclose(rec, image, rtol=2e-5, atol=1e-6, equal_nan=True)


def test_a_whole_image_of_one_tile_is_the_evaluation_reconstruction(tmp_path):
    """Denoising and evaluation share one path: an image of exactly one tile is reconstructed exactly
    like an evaluation crop."""
    cfg = smoke_config(tmp_path)
    m = TrainedModel.load(_model_folder(cfg, _random_model()))
    image, _ = _image((PS, PS), seed=4)
    image = np.nan_to_num(image, nan=0.6)
    image[image == 0] = 0.6  # no holes: the crop is used as it is
    expected = m.reconstruct(image[None], batch_size=1)[0]
    np.testing.assert_array_equal(m.denoise(image, None, overlap=0), expected)
    assert not np.allclose(expected, image), "the test model must change the image"


def test_no_data_pixels_are_sky_for_the_model_and_keep_their_values(tmp_path, monkeypatch):
    cfg = smoke_config(tmp_path)
    m = TrainedModel.load(_model_folder(cfg, _random_model()))
    image, weight = _image((150, 150), seed=5, weight=True)
    hole = m.no_data_mask(image, weight)
    dead = (weight > 0) & (weight < 0.5 * np.median(weight[weight > 0]))
    assert dead.any() and hole[dead].all(), "low-weight pixels count as no data, as in training"
    seen = []
    original = TrainedModel.reconstruct
    monkeypatch.setattr(TrainedModel, "reconstruct", lambda self, x, *a, **k: seen.append(x) or original(self, x, *a, **k))
    rec = m.denoise(image, weight, overlap=16)
    tiles = np.concatenate(seen)
    assert np.isfinite(tiles).all() and (tiles != 0).all(), "the model never sees zeros or NaN"
    assert np.nanmin(tiles) > 0.3, "holes are filled with the sky level (about 0.6 e-/s), not 0"
    np.testing.assert_array_equal(rec[hole & np.isfinite(image)], image[hole & np.isfinite(image)])
    assert np.isnan(rec[np.isnan(image)]).all()
    assert not np.allclose(rec[~hole], image[~hole])


def _fits(path, image, weight, header):
    from astrounets.data.fits_io import write_image

    write_image(path, image, "SCI", header, weight=weight)
    return path


WCS_HEADER = {"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRPIX1": 50.0, "CRPIX2": 40.0, "CRVAL1": 53.1,
              "CRVAL2": -27.8, "CD1_1": -0.12825 / 3600, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 0.12825 / 3600}


def test_denoise_file_writes_the_input_layout_with_wcs_and_a_figure(tmp_path, caplog):
    cfg = smoke_config(tmp_path)
    m = TrainedModel.load(_model_folder(cfg, _identity()))
    image, weight = _image((90, 120), seed=6, weight=True)
    header = {"EXPTIME": 1000.0, "FILTER": "F160W", "INSTRUME": "WFC3", "BUNIT": "ELECTRONS/S", **WCS_HEADER}
    src = _fits(tmp_path / "field_drz.fits", image, weight, header)
    written = denoise_file(m, src, tmp_path / "out", overlap=16, batch_size=4, tta=False, figure=True, zoom=64,
                           centres=[(10, 10)], dpi=60)
    assert [p.name for p in written] == ["field_drz_rec.fits", "field_drz_compare.png"]
    assert written[1].stat().st_size > 10_000
    with fits.open(src) as a, fits.open(written[0]) as b:
        assert [h.name for h in b] == [h.name for h in a] == ["PRIMARY", "SCI", "WHT"]
        for key in ("CRVAL1", "CRVAL2", "CRPIX1", "CD1_1", "CD2_2", "CTYPE1", "BUNIT", "FILTER"):
            assert b["SCI"].header[key] == a["SCI"].header[key], key
        assert b["SCI"].header["DNMODEL"] == m.name and b["SCI"].header["DNEPOCH"] == 7
        assert b["SCI"].header["DNOVERLP"] == 16 and b["SCI"].header["DNSCALE"] == m.scaler.mode
        np.testing.assert_array_equal(b["WHT"].data, a["WHT"].data)
        np.testing.assert_allclose(b["SCI"].data, a["SCI"].data, rtol=2e-5, atol=1e-6, equal_nan=True)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], "an HST-like image raises no warning"


def test_images_in_electrons_are_converted_and_foreign_images_warned_about(tmp_path, caplog):
    cfg = replace(smoke_config(tmp_path), {"scaling": {"mode": "log_min_max"}})
    m = TrainedModel.load(_model_folder(cfg, _identity()))
    image, _ = _image((80, 80), seed=7)
    header = {"EXPTIME": 500.0, "FILTER": "F125W", "BUNIT": "ELECTRONS", **WCS_HEADER, "CD2_2": 0.06 / 3600}
    src = _fits(tmp_path / "x.fits", image * 500.0, None, header)
    seen = []
    original = TrainedModel.denoise
    m.denoise = lambda img, *a, **k: seen.append(img) or original(m, img, *a, **k)
    (rec_path,) = denoise_file(m, src, tmp_path / "out", 16, 4, False, False, 64)
    np.testing.assert_allclose(np.nanmax(seen[0]), np.nanmax(image), rtol=1e-6)  # the model gets e-/s
    with fits.open(rec_path) as b:
        np.testing.assert_allclose(b["SCI"].data, image * 500.0, rtol=2e-5, atol=1e-4, equal_nan=True)
    warned = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "filter F125W" in warned and "pixel scale" in warned and "weight map" in warned


def test_find_images_takes_files_folders_and_patterns(tmp_path):
    for name in ("a_drz.fits", "b_drz.fits.gz", "a_drz_rec.fits", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in find_images([str(tmp_path)])] == ["a_drz.fits", "b_drz.fits.gz"]
    assert [p.name for p in find_images([str(tmp_path / "a*.fits")])] == ["a_drz.fits", "a_drz_rec.fits"]
    with pytest.raises(FileNotFoundError):
        find_images([str(tmp_path / "missing.fits")])


def test_cli_uses_the_final_model_by_default(tmp_path, capsys):
    """denoise.model: best is the study's final model: its run folder, or the report's copy of it."""
    import shutil

    from astrounets.cli import main
    from astrounets.workflow import final_path, report_index

    cfg = smoke_config(tmp_path)
    key = f"{cfg.dataset.name}/{cfg.run_dir.name}"
    image, weight = _image((70, 90), seed=8)
    src = _fits(tmp_path / "img.fits", image, weight, {"EXPTIME": 1000.0, "FILTER": "F160W"})
    args = ["denoise", "-c", str(SMOKE), "--set", f"paths.data={tmp_path / 'data'}", "--set",
            f"paths.runs={tmp_path / 'runs'}", "--out", str(tmp_path / "out"), "--no-figure", str(src)]
    assert main(args) == 1 and "no final model yet" in capsys.readouterr().err

    _model_folder(cfg, _identity())
    final_path(cfg).parent.mkdir(parents=True)
    final_path(cfg).write_text(json.dumps({"run": key}))
    assert main(args) == 0
    with fits.open(tmp_path / "out" / "img_rec.fits") as b:
        assert b["SCI"].header["DNMODEL"] == key
    # only the report at hand (copied from the cluster): its tables/final.json and models/<key>
    report = report_index(cfg).parent
    _model_folder(cfg, _identity(), folder=report / "models" / key, layout="flat")
    (report / "tables").mkdir()
    shutil.move(final_path(cfg), report / "tables" / "final.json")
    shutil.rmtree(cfg.run_dir)
    (tmp_path / "out" / "img_rec.fits").unlink()
    assert main(args + ["--overlap", "8"]) == 0
    with fits.open(tmp_path / "out" / "img_rec.fits") as b:
        assert b["SCI"].header["DNMODEL"] == key and b["SCI"].header["DNOVERLP"] == 8
    # an explicit model folder wins
    assert main(args + ["--model", str(report / "models" / key)]) == 0
    assert main(args + ["--overlap", "40"]) == 1 and "denoise.overlap" in capsys.readouterr().err


def test_choose_final_ranks_the_stage2_models_on_validation(tmp_path, monkeypatch):
    import astrounets.evaluate.run as evaluation
    from astrounets import workflow as wf
    from astrounets.evaluate.run import eval_dir

    cfg = smoke_config(tmp_path)
    setups = [{"model": {"type": "unet", "attention": a}} for a in (False, True)] + [{"model": {"type": "gan"}}]
    runs = []
    for setup in setups:
        for name in ("mock", "mock-faber"):
            run_cfg = replace(cfg, {**setup, "dataset": {"name": name}})
            runs.append(wf.Run("datasets", wf.run_key(run_cfg), run_cfg, setup, []))
    monkeypatch.setattr(wf.Plan, "runs", lambda self, stage: runs)
    scatter = {"unet_noattn": 0.2, "unet_attn": 0.1, "gan_attn": 0.05}  # the GAN is best, but on mock-faber only

    def fake_evaluate(run_cfg, run_dir, dataset, split):
        value = scatter["_".join(run_cfg.run_name.split("_")[:2])] if run_cfg.dataset.name == "mock" else 0.01
        ranking = {f"{m}_{g}": 1 - value if m == "ssim" else value
                   for m in ("flux_scatter", "flux_bias", "ssim", "radius_error", "ellipticity_error")
                   for g in ("overall", "sparse", "crowded")}
        ranking.update({f"guard_ok_{g}": True for g in ("overall", "sparse", "crowded")})
        ranking.update({f"n_samples_{g}": 10 for g in ("overall", "sparse", "crowded")})
        wf.write_json(eval_dir(run_dir, dataset, split) / "summary.json", {"ranking": ranking})
        calls.append((run_cfg.dataset.name, dataset, split))

    calls = []
    monkeypatch.setattr(evaluation, "evaluate", fake_evaluate)
    for run in runs:
        wf.write_json(wf.marker_path(cfg, "datasets", run.key), {"train": {"status": "finished"}})
    scatter["gan_attn"] = 0.3  # on mock, the GAN is worst
    out = wf.select(cfg, "final")
    chosen = json.loads(out.read_text())
    assert chosen["run"].startswith("mock/unet_attn_") and chosen["selected_on"] == "mock:val"
    assert chosen["candidates"] == 3 and calls == [("mock", "mock", "val")] * 3
    assert (out.parent / "ranking.csv").exists()
    calls.clear()
    wf.select(cfg, "final")
    assert calls == [], "evaluations on the validation split are reused"


def test_figure_cutouts_show_their_own_region():
    from astrounets.plots import _cutouts, compare_rows

    image = np.arange(300 * 400, dtype=float).reshape(300, 400)
    boxes = _cutouts(image.shape, 64, [(50, 60), (350, 250), (1000, -5)])
    assert boxes == [(28, 92, 18, 82), (218, 282, 318, 382), (0, 64, 336, 400)]  # clipped to the image
    rows = compare_rows(image, image + 1, image - 1, boxes, factor=2)
    assert rows[0][0].shape == (150, 200) and rows[0][0][0, 0] == np.mean(image[:2, :2])
    for (y0, y1, x0, x1), row in zip(boxes, rows[1:]):
        np.testing.assert_array_equal(row[0], image[y0:y1, x0:x1])
        np.testing.assert_array_equal(row[1], image[y0:y1, x0:x1] + 1)
