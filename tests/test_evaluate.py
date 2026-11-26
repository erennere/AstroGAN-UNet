import json

import keras
import numpy as np
import pandas as pd
import pytest

from astrounets.config import replace
from astrounets.data.dataset import load_crops
from astrounets.data.mock import make_image
from astrounets.data.noise import add_noise, noisy_sigma
from astrounets.data.pairs import eval_table
from astrounets.evaluate.detect import measure
from astrounets.evaluate.match import match
from astrounets.evaluate.metrics import compare, summary


def test_matching_is_one_to_one():
    truth = np.array([[10.0, 10.0], [11.0, 10.0], [50.0, 50.0]])
    found = np.array([[10.4, 10.0], [80.0, 80.0]])
    i_true, i_found = match(truth, found, radius=3.0)
    assert list(zip(i_true, i_found)) == [(0, 0)]  # two true sources near one detection: one match only
    assert match(truth, np.empty((0, 2)), 3.0)[0].size == 0


def test_detection_recovers_injected_sources(cfg):
    rng = np.random.default_rng(4)
    image, sigma, catalog = make_image(rng, 256, 5000.0, n_sources=10, border=False)
    found = measure(image, 3 * sigma, cfg.eval)
    i_true, i_found = match(catalog[["x", "y"]].to_numpy(), found[["x", "y"]].to_numpy(), 2.0)
    bright = catalog["flux"] > 1.0
    assert bright[i_true].sum() == bright.sum()
    ratio = found["flux"].to_numpy()[i_found] / catalog["flux"].to_numpy()[i_true]
    assert np.median(ratio[bright.to_numpy()[i_true]]) == pytest.approx(1.0, abs=0.1)  # Kron flux ~ total flux


def test_degenerate_sep_shapes_do_not_break_photometry(cfg, monkeypatch):
    """Real data gave theta = float32(pi/2), just above SEP's float64 limit, and SEP can return NaN or
    zero axes; such sources are measured with a circular aperture instead of crashing the evaluation."""
    import sep

    import astrounets.evaluate.detect as detect

    rng = np.random.default_rng(4)
    image, sigma, catalog = make_image(rng, 128, 5000.0, n_sources=6, border=False)
    real = sep.extract
    edge = np.float32(np.pi / 2)

    def odd_extract(*args, **kwargs):
        objects = real(*args, **kwargs).copy()
        objects["theta"][0], objects["theta"][1] = edge, -edge
        objects["a"][2], objects["b"][3] = np.nan, 0.0
        objects["x"][4] = np.nan
        return objects

    monkeypatch.setattr(detect.sep, "extract", odd_extract)
    found = measure(image, 3 * sigma, cfg.eval)
    assert len(found) >= 4 and found["x"].notna().all() and np.isfinite(found["flux"]).all()
    assert found["ellipticity"].isna().sum() == 2  # the NaN-axis and zero-axis sources have no shape


def test_shared_background_removes_the_flux_bias_of_noisy_images(cfg):
    """In a crowded field (like PHAT M31 in the pilot) a background estimated on the noisy image takes in
    more of the sources' light than one estimated on the deep image, so noisy-input fluxes came out
    ~0.2-0.3 mag too faint. With one background (from the deep image) for every image the bias is gone."""
    e = cfg.eval
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:128, 0:128]
    clean = rng.normal(0.0, 0.05, (128, 128))
    for _ in range(400):
        x0, y0 = rng.uniform(0, 128, 2)
        clean += np.exp(rng.uniform(np.log(0.5), np.log(20))) * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / 2.88) / 9.05
    noisy = add_noise(clean, 0.05, 16.0, rng)
    threshold = e.detect_nsigma * noisy_sigma(0.05, 16.0)

    def dmag(true, found):
        i_true, i_found = match(true[["x", "y"]].to_numpy(), found[["x", "y"]].to_numpy(), e.match_radius)
        return float(-2.5 * np.log10(np.nanmedian(found["flux"].to_numpy()[i_found] / true["flux"].to_numpy()[i_true])))

    own = dmag(measure(clean, threshold, e), measure(noisy, threshold, e))
    _, sources = compare(clean, noisy, clean, threshold, e, 1.9e-20, 15370.0)
    matched = sources[sources["matched_noisy"]]
    shared = float(-2.5 * np.log10(np.nanmedian(matched["flux_noisy"] / matched["flux_true"])))
    assert own > 0.1 and abs(shared) < 0.05


def test_perfect_reconstruction_scores_perfectly(cfg):
    rng = np.random.default_rng(6)
    clean, sigma, _ = make_image(rng, 128, 2500.0, n_sources=8, border=False)
    noisy = add_noise(clean, sigma, 16.0, rng)
    threshold = cfg.eval.detect_nsigma * noisy_sigma(sigma, 16.0)
    sample, sources = compare(clean, noisy, clean.copy(), threshold, cfg.eval, 1.9275602e-20, 15369.176)
    samples = pd.DataFrame([{**sample, "sample_id": 0, "ratio": 16.0}])
    sources["sample_id"] = 0
    result = summary(samples, sources, cfg.eval.mag_bins)
    o = result["overall"]
    assert o["completeness_rec"] == 1.0 and o["purity_rec"] == 1.0
    assert o["flux_bias_rec"] == 0.0 and o["flux_scatter_rec"] == 0.0 and o["ssim_rec"] == pytest.approx(1.0)
    assert result["ranking"]["guard_ok_overall"] is True and set(result["groups"]) == {"overall", "sparse"}
    # returning the noisy input unchanged scores exactly like the noisy input
    sample2, sources2 = compare(clean, noisy, noisy.copy(), threshold, cfg.eval, 1.9275602e-20, 15369.176)
    assert sample2["tp_rec"] == sample2["tp_noisy"] and sample2["ssim_rec"] == sample2["ssim_noisy"]


def test_eval_set_is_frozen_and_identical_for_every_model(mock_data):
    crops = load_crops(mock_data)
    a = eval_table(crops, mock_data, "test", mock_data.dataset_dir())
    b = eval_table(crops, mock_data, "test", mock_data.dataset_dir())
    pd.testing.assert_frame_equal(a, b)
    assert set(a["split"]) == {"test"}
    assert (a["ratio"] <= np.minimum(mock_data.noise.ratio_max, a["exptime"] / mock_data.noise.min_new_exposure)).all()
    changed = replace(mock_data, {"eval": {"n_crops": 3}})
    chosen = eval_table(crops, changed, "test", mock_data.dataset_dir()).drop_duplicates("crop_id")
    assert chosen.groupby("crowded").size().to_dict() == {False: 3, True: 3}  # n_crops from each group


def _fake_run(cfg, model):
    """A run folder holding a hand-made 'best' generator."""
    from astrounets.config import save_config

    run = cfg.run_dir
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    save_config(cfg, run / "config.yaml")
    model.save(run / "checkpoints" / "best.keras")
    (run / "checkpoints" / "best.json").write_text(json.dumps({"epoch": 1, "val_rec_loss": 0.0}))
    return run


def test_evaluate_identity_model_equals_noisy_input(mock_data):
    from astrounets.evaluate.run import evaluate

    ps = mock_data.dataset.patch_size
    inp = keras.Input((ps, ps, 1))
    identity = keras.Model(inp, keras.layers.Activation("linear")(inp))
    run = _fake_run(mock_data, identity)
    out = evaluate(mock_data, run, "mock", "test")
    result = json.loads((out / "summary.json").read_text())
    assert set(result["groups"]) == {"overall", "sparse", "crowded"}  # the mock test split has both
    for o in result["groups"].values():
        assert o["completeness_rec"] == pytest.approx(o["completeness_noisy"])
        assert o["flux_scatter_rec"] == pytest.approx(o["flux_scatter_noisy"], abs=1e-6)
        assert o["ssim_rec"] == pytest.approx(o["ssim_noisy"], abs=1e-6)
    groups = result["groups"]
    assert groups["sparse"]["n_samples"] + groups["crowded"]["n_samples"] == groups["overall"]["n_samples"]
    samples = pd.read_parquet(out / "samples.parquet")
    assert samples["crowded"].sum() == groups["crowded"]["n_samples"]
    samples = pd.read_parquet(out / "samples.parquet")
    assert len(samples) == len(eval_table(load_crops(mock_data), mock_data, "test", mock_data.dataset_dir()))
    assert {"dmag_rec", "matched_noisy", "mag_true"} <= set(pd.read_parquet(out / "sources.parquet").columns)


def test_psnr_is_the_median_over_crops():
    from astrounets.evaluate.metrics import summarise

    samples = pd.DataFrame({"n_true": [1, 1, 1], "range": [1.0, 1.0, 100.0], "ratio": 2.0,
                            "tp_rec": 1, "n_rec": 1, "mse_rec": [0.01, 0.01, 1e6], "ssim_rec": 0.9,
                            "tp_noisy": 1, "n_noisy": 1, "mse_noisy": 0.01, "ssim_noisy": 0.9})
    cols = ["dmag_rec", "dmag_noisy", "matched_rec", "matched_noisy", "radius_rec", "radius_noisy",
            "radius_true", "ellipticity_rec", "ellipticity_noisy", "ellipticity_true"]
    out = summarise(samples, pd.DataFrame({c: pd.Series(dtype=float) for c in cols}))
    assert out["psnr_rec"] == pytest.approx(20.0)  # one wild crop (PSNR -20 dB) does not drag it down
