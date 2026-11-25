"""Physics invariants: the checks that would have caught the errors in the old pipeline."""
import numpy as np
import pytest

from astrounets.config import replace
from astrounets.data.mock import make_image
from astrounets.data.noise import add_noise, max_ratio, noisy_sigma, sample_ratio
from astrounets.data.scaling import Scaler
from astrounets.evaluate.metrics import ab_mag


@pytest.mark.parametrize("exptime", [500.0, 2500.0])
def test_sigma_estimate_recovers_injected_noise(cfg, exptime):
    from astrounets.data.dataset import measure_background

    rng = np.random.default_rng(3)
    image, sigma, _ = make_image(rng, 512, exptime)
    _, rms = measure_background(image, replace(cfg, {"dataset": {"bkg_box_size": 64}}).dataset)
    valid = np.isfinite(image) & (image != 0)
    estimate = np.median(rms[valid])
    assert estimate == pytest.approx(sigma, rel=0.05)


def test_added_noise_has_target_spread_and_zero_mean():
    rng = np.random.default_rng(0)
    clean = np.full((256, 256), 0.5, np.float32)
    for ratio in (2.0, 10.0, 60.0):
        noisy = add_noise(clean, 0.03, ratio, rng)
        diff = noisy - clean
        assert diff.mean() == pytest.approx(0, abs=0.002)
        assert diff.std() == pytest.approx(0.03 * np.sqrt(ratio - 1), rel=0.02)
        # clean already carries sigma; the sum has the noise of the shorter exposure
        assert np.hypot(0.03, diff.std()) == pytest.approx(noisy_sigma(0.03, ratio), rel=0.02)


def test_noise_is_reproducible_and_keeps_empty_pixels_empty():
    clean = np.ones((32, 32), np.float32); clean[:4] = 0
    a = add_noise(clean, 0.1, 4.0, np.random.default_rng(5))
    b = add_noise(clean, 0.1, 4.0, np.random.default_rng(5))
    assert np.array_equal(a, b) and np.all(a[:4] == 0)


def test_ratio_range_respects_minimum_new_exposure(cfg):
    assert max_ratio(600.0, cfg.noise) == pytest.approx(10.0)
    assert max_ratio(1e6, cfg.noise) == cfg.noise.ratio_max
    rng = np.random.default_rng(1)
    ratios = [sample_ratio(rng, 600.0, cfg.noise) for _ in range(500)]
    assert min(ratios) >= cfg.noise.ratio_min and max(ratios) <= 10.0


@pytest.mark.parametrize("mode", ["none", "z_scale", "min_max", "log_min_max"])
def test_perfect_prediction_returns_true_flux(mode):
    rng = np.random.default_rng(2)
    clean, sigma, catalog = make_image(rng, 128, 1000.0, n_sources=6, border=False)
    noisy = add_noise(clean, sigma, 16.0, rng)
    scaler = Scaler(mode)
    params = scaler.fit(noisy)
    target = scaler.forward(clean, params)          # what the network is trained to output
    recovered = scaler.inverse(target, params)       # what evaluation turns that output into
    np.testing.assert_allclose(recovered, clean, rtol=1e-4, atol=1e-5)
    yy, xx = np.mgrid[0:128, 0:128]
    for _, src in catalog.iterrows():
        aperture = (xx - src.x) ** 2 + (yy - src.y) ** 2 < 25
        assert recovered[aperture].sum() == pytest.approx(clean[aperture].sum(), rel=0.01, abs=1e-3)


def test_scaler_rejects_constant_input():
    assert Scaler("z_scale").fit(np.zeros((8, 8))) is None
    assert Scaler("min_max").fit(np.ones((8, 8))) is None


def test_ab_magnitude_matches_hand_calculation():
    photflam, photplam = 1.9275602e-20, 15369.176
    zeropoint = -2.5 * np.log10(photflam) - 5 * np.log10(photplam) - 2.408   # STScI recipe
    assert float(ab_mag(1.0, photflam, photplam)) == pytest.approx(zeropoint, abs=0.002)
    assert float(ab_mag(1.0, photflam, photplam)) == pytest.approx(25.95, abs=0.02)  # published F160W zeropoint
    assert np.isnan(ab_mag(-1.0, photflam, photplam))
