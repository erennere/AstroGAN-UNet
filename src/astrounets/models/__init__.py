"""Models: U-Net generator, conditional PatchGAN discriminator, losses and the training wrapper."""
from __future__ import annotations

import keras

from astrounets.config import Config


def build(cfg: Config):
    """(Denoiser, generator, discriminator or None) for a config, not yet compiled."""
    from astrounets.models.gan import Denoiser, build_patch_discriminator
    from astrounets.models.losses import LOSSES
    from astrounets.models.unet import build_unet

    ps = cfg.dataset.patch_size
    generator = build_unet(ps, cfg.model)
    discriminator = build_patch_discriminator(ps, cfg.gan) if cfg.model.type == "gan" else None
    model = Denoiser(generator, LOSSES[cfg.train.loss], discriminator, cfg.gan)
    return model, generator, discriminator


def optimizers(cfg: Config):
    """Adam for the generator and (GAN only) the discriminator, whose rate is gan.d_lr_ratio times the
    generator's; train.LearningRate updates both every epoch."""
    lr = cfg.train.learning_rate
    g = keras.optimizers.Adam(lr, beta_1=cfg.train.beta_1)
    d = keras.optimizers.Adam(lr * cfg.gan.d_lr_ratio, beta_1=cfg.train.beta_1) if cfg.model.type == "gan" else None
    return g, d
