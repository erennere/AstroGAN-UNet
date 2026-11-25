"""U-Net generator with optional attention gates (Oktay et al. 2018)."""
from __future__ import annotations

import keras
from keras import layers

from astrounets.config import Model


def _activation(name: str):
    if name == "leakyrelu":
        return layers.LeakyReLU()  # Keras default slope 0.3, as in the thesis runs
    return layers.Activation(name)


def conv_block(x, filters: int, m: Model):
    for i in range(m.block_depth):
        x = layers.Conv2D(filters, m.kernel_size, padding="same", use_bias=m.use_bias,
                          kernel_initializer=m.initializer)(x)
        if m.batch_norm:
            x = layers.BatchNormalization()(x)
        x = _activation(m.activation)(x)
        if m.dropout > 0 and i >= m.dropout_from_layer:
            x = layers.Dropout(m.dropout)(x)
    return x


def attention_gate(skip, gate, filters: int):
    theta = layers.Conv2D(filters, 1, padding="same")(skip)
    phi = layers.Conv2D(filters, 1, padding="same")(gate)
    f = layers.Activation("relu")(layers.Add()([theta, phi]))
    alpha = layers.Activation("sigmoid")(layers.Conv2D(1, 1, padding="same")(f))
    return layers.Multiply()([skip, alpha])


def build_unet(patch_size: int, m: Model) -> keras.Model:
    """Encoder of `depth` levels (the last one is the bottleneck); upsampling uses pool_size."""
    inputs = keras.Input((patch_size, patch_size, 1), name="noisy")
    skips, x, filters = [], inputs, m.filters
    for level in range(m.depth):
        x = conv_block(x, filters, m)
        if level < m.depth - 1:
            skips.append((x, filters))
            x = layers.MaxPooling2D(m.pool_size)(x)
            filters *= m.growth
    for skip, filters in reversed(skips):
        x = layers.Conv2DTranspose(filters, m.pool_size, strides=m.pool_size, padding="same",
                                   kernel_initializer=m.initializer)(x)
        if m.attention:
            skip = attention_gate(skip, x, max(1, filters // 2))
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, filters, m)
    activation = None if m.output_activation == "none" else m.output_activation
    outputs = layers.Conv2D(1, 1, activation=activation, kernel_initializer=m.initializer, name="clean")(x)
    return keras.Model(inputs, outputs, name="unet")
