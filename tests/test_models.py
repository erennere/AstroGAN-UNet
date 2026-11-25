import itertools

import keras
import numpy as np
import pytest
import tensorflow as tf

from astrounets.config import ConfigError, load_config
from astrounets.models import build, optimizers
from astrounets.models.losses import LOSSES
from conftest import SMOKE

GRID = list(itertools.product(["unet", "gan"], [False, True], ["none", "z_scale", "min_max", "log_min_max"],
                              ["log_cosh", "simae", "mae", "ssim"]))
TINY = ["dataset.patch_size=32", "model.depth=2", "model.filters=4", "gan.disc_filters=4", "gan.disc_layers=2"]


def _cfg(model_type, attention, scaling, loss):
    return load_config(SMOKE, TINY + [f"model.type={model_type}", f"model.attention={str(attention).lower()}",
                                      f"scaling.mode={scaling}", f"train.loss={loss}"])


def test_grid_has_56_valid_and_8_invalid_setups():
    valid, invalid = 0, 0
    for combo in GRID:
        try:
            _cfg(*combo); valid += 1
        except ConfigError:
            invalid += 1
            assert combo[3] == "ssim" and combo[2] in ("none", "z_scale")
    assert (valid, invalid) == (56, 8)


@pytest.mark.parametrize("model_type,attention", [("unet", False), ("unet", True), ("gan", False), ("gan", True)])
def test_every_architecture_trains_saves_and_reloads(model_type, attention, tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 32, 32, 1)).astype("float32")
    y = (x * 0.5).astype("float32")
    for loss in LOSSES:  # each loss for each architecture: one real training step
        cfg = _cfg(model_type, attention, "min_max", loss)
        model, generator, discriminator = build(cfg)
        g_opt, d_opt = optimizers(cfg)
        model.compile(g_opt, d_opt)
        logs = model.fit(x, y, batch_size=2, epochs=1, verbose=0).history
        assert np.isfinite(logs["rec_loss"][0])
        assert ("d_loss" in logs) == (model_type == "gan")
    generator.save(tmp_path / "g.keras")
    again = keras.models.load_model(tmp_path / "g.keras", compile=False)
    np.testing.assert_allclose(generator(x, training=False), again(x, training=False), rtol=1e-5, atol=1e-6)


def test_unet_upsampling_follows_pool_size():
    cfg = load_config(SMOKE, ["dataset.patch_size=27", "model.depth=3", "model.pool_size=3", "model.growth=2", "model.filters=4"])
    model, generator, _ = build(cfg)
    assert generator.output_shape == (None, 27, 27, 1)


def test_initializer_reaches_every_convolution():
    cfg = load_config(SMOKE, TINY + ["model.initializer=zeros"])
    _, generator, _ = build(cfg)
    convs = [l for l in generator.layers if isinstance(l, (keras.layers.Conv2D, keras.layers.Conv2DTranspose))
             and l.name != "clean" and "conv2d" in l.name and l.kernel_size != (1, 1)]
    assert convs and all(float(tf.reduce_max(tf.abs(l.kernel))) == 0 for l in convs)


def test_discriminator_is_conditional_and_patch_based():
    cfg = load_config(SMOKE, TINY + ["model.type=gan"])
    _, _, disc = build(cfg)
    assert len(disc.inputs) == 2                      # sees the noisy input and the candidate
    assert disc.output_shape[1:3] == (8, 8)            # a map of logits, not one number
    assert disc.count_params() < 50_000


@pytest.mark.parametrize("name", list(LOSSES))
def test_losses_are_zero_for_perfect_prediction(name):
    y = tf.random.uniform((2, 32, 32, 1))
    assert float(LOSSES[name](y, y)) == pytest.approx(0, abs=1e-5)
    assert float(LOSSES[name](y, y + 0.3)) > 0
