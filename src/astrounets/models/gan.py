"""Training wrapper for both model types and the conditional PatchGAN discriminator.

`Denoiser` trains a generator alone (U-Net) or against a discriminator (GAN). Both report the same
`rec_loss` (reconstruction loss, epoch mean) so U-Net and GAN runs are directly comparable.
"""
from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from astrounets.config import Gan


def build_patch_discriminator(patch_size: int, g: Gan) -> keras.Model:
    """Conditional PatchGAN (pix2pix): sees [noisy, candidate] and returns a map of real/fake logits."""
    noisy = keras.Input((patch_size, patch_size, 1), name="noisy")
    candidate = keras.Input((patch_size, patch_size, 1), name="candidate")
    x = layers.Concatenate()([noisy, candidate])
    filters = g.disc_filters
    for i in range(g.disc_layers):
        x = layers.Conv2D(filters, 4, strides=2, padding="same", use_bias=i == 0)(x)
        if i > 0:
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        filters *= 2
    x = layers.Conv2D(filters, 4, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    logits = layers.Conv2D(1, 4, padding="same", name="logits")(x)
    return keras.Model([noisy, candidate], logits, name="patchgan")


class Denoiser(keras.Model):
    def __init__(self, generator, rec_loss, discriminator=None, gan: Gan | None = None, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.rec_loss = rec_loss
        self.gan_cfg = gan
        self.bce = keras.losses.BinaryCrossentropy(from_logits=True)
        names = ["rec_loss"] + (["d_loss", "adv_loss", "g_loss"] if discriminator is not None else [])
        self.trackers = {name: keras.metrics.Mean(name=name) for name in names}

    def compile(self, g_optimizer, d_optimizer=None):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    @property
    def metrics(self):
        return list(self.trackers.values())

    def call(self, x, training=False):
        return self.generator(x, training=training)

    def train_step(self, data):
        x, y = data
        g_vars = self.generator.trainable_variables
        if self.discriminator is None:
            with tf.GradientTape() as tape:
                rec = self.rec_loss(y, self.generator(x, training=True))
            self.g_optimizer.apply_gradients(zip(tape.gradient(rec, g_vars), g_vars))
            self.trackers["rec_loss"].update_state(rec)
            return {name: t.result() for name, t in self.trackers.items()}

        g = self.gan_cfg
        d_vars = self.discriminator.trainable_variables
        with tf.GradientTape() as tape:
            fake = self.generator(x, training=True)
            real_logits = self.discriminator([x, y], training=True)
            fake_logits = self.discriminator([x, fake], training=True)
            d_loss = (self.bce(tf.ones_like(real_logits) * (1.0 - g.label_smoothing), real_logits)
                      + self.bce(tf.zeros_like(fake_logits), fake_logits))
        self.d_optimizer.apply_gradients(zip(tape.gradient(d_loss, d_vars), d_vars))
        with tf.GradientTape() as tape:
            fake = self.generator(x, training=True)
            fake_logits = self.discriminator([x, fake], training=True)
            adv = self.bce(tf.ones_like(fake_logits), fake_logits)
            rec = self.rec_loss(y, fake)
            g_loss = g.adversarial_weight * adv + g.reconstruction_weight * rec
        self.g_optimizer.apply_gradients(zip(tape.gradient(g_loss, g_vars), g_vars))
        for name, value in (("rec_loss", rec), ("d_loss", d_loss), ("adv_loss", adv), ("g_loss", g_loss)):
            self.trackers[name].update_state(value)
        return {name: t.result() for name, t in self.trackers.items()}

    def test_step(self, data):
        x, y = data
        rec = self.rec_loss(y, self.generator(x, training=False))
        self.trackers["rec_loss"].update_state(rec)
        return {"rec_loss": self.trackers["rec_loss"].result()}
