import numpy as np
import pytest
import tensorflow as tf

from src.models.network import GAN, conv_block, get_discriminator, network, upsample_and_concat
from src.training.utils import log_cosh_loss, scale_invariant_mae, ssim_loss


@pytest.mark.unit
@pytest.mark.parametrize('depth', [2, 3])
@pytest.mark.parametrize('attention', [False, True])
@pytest.mark.parametrize('output_activation', [None, 'relu', 'sigmoid', 'tanh'])
def test_network_output_shape_and_trainable_weights(depth, attention, output_activation):
    model = network(
        (64, 64, 1),
        depth=depth,
        kernel_size=3,
        filter_size=2,
        pooling_size=2,
        n_of_initial_channels=4,
        func=tf.keras.layers.ReLU,
        func_kwargs={},
        batch_normalization=False,
        use_bias=True,
        dropout_rate=0.0,
        block_depth=2,
        dropout_from_layer=99,
        attention=attention,
        output_activation=output_activation,
        kernel_initializer='he_normal',
    )
    output = model(tf.random.uniform((2, 64, 64, 1)))

    assert output.shape == (2, 64, 64, 1)
    assert model.trainable_weights


@pytest.mark.unit
@pytest.mark.parametrize('depth', [2, 3])
def test_discriminator_output_shape_and_trainable_weights(depth):
    model = get_discriminator(
        (64, 64, 1),
        depth=depth,
        n_initial_filters=4,
        filter_size=2,
        kernel_size=(3, 3),
        dropout_rate=0.0,
        func=tf.keras.layers.LeakyReLU,
        func_kwargs={},
        output_activation='sigmoid',
        block_depth=1,
        batch_normalization=False,
        use_bias=True,
        dropout_from_layer=99,
    )
    output = model(tf.random.uniform((2, 64, 64, 1)))

    assert output.shape == (2, 1)
    assert model.trainable_weights


@pytest.mark.unit
def test_conv_block_and_upsample_concat_shapes():
    inputs = tf.keras.Input(shape=(32, 32, 1))
    conv_out = conv_block(inputs, output_channels=4, kernel_size=3, depth=2, func=tf.keras.layers.ReLU, func_kwargs={}, batch_normalization=False, use_bias=True, dropout_rate=0.0, dropout_from_layer=99)
    conv_model = tf.keras.Model(inputs, conv_out)

    low_res = tf.keras.Input(shape=(16, 16, 8))
    skip = tf.keras.Input(shape=(32, 32, 4))
    upsampled = upsample_and_concat(low_res, skip, kernel_size=3, filter_size=2, func=tf.keras.layers.ReLU, func_kwargs={}, batch_normalization=False, use_bias=True, dropout_rate=0.0, block_depth=2, dropout_from_layer=99, attention=False, kernel_initializer='he_normal')
    upsample_model = tf.keras.Model([low_res, skip], upsampled)

    assert conv_model(tf.random.uniform((1, 32, 32, 1))).shape == (1, 32, 32, 4)
    assert upsample_model([tf.random.uniform((1, 16, 16, 8)), tf.random.uniform((1, 32, 32, 4))]).shape == (1, 32, 32, 4)


@pytest.mark.unit
def test_gan_train_step_returns_expected_metrics(tiny_gan):
    x = tf.random.uniform((2, 64, 64, 1))
    y = tf.random.uniform((2, 64, 64, 1))
    metrics = tiny_gan.train_step((x, y))

    assert {'d_loss', 'g_loss', 'g_adv_loss', 'g_rec_loss'} <= set(metrics)
    assert np.isfinite(float(metrics['d_loss']))
    assert np.isfinite(float(metrics['g_loss']))


@pytest.mark.unit
def test_gan_generator_loss_changes_with_reconstruction_weight(mock_cfg):
    x = tf.random.uniform((2, 64, 64, 1), seed=11)
    y = tf.random.uniform((2, 64, 64, 1), seed=12)
    base_generator = network(mock_cfg['input_shape'], **mock_cfg['training']['network_kwargs'])
    base_discriminator = get_discriminator(mock_cfg['input_shape'], **mock_cfg['training']['discriminator_kwargs'])
    _ = base_generator(x)
    _ = base_discriminator(y)

    def build_gan(rec_weight):
        generator = tf.keras.models.clone_model(base_generator)
        discriminator = tf.keras.models.clone_model(base_discriminator)
        generator(tf.random.uniform((1, 64, 64, 1)))
        discriminator(tf.random.uniform((1, 64, 64, 1)))
        generator.set_weights(base_generator.get_weights())
        discriminator.set_weights(base_discriminator.get_weights())
        gan = GAN(
            generator=generator,
            discriminator=discriminator,
            g_optimizer=tf.keras.optimizers.SGD(0.0),
            d_optimizer=tf.keras.optimizers.SGD(0.0),
            adversarial_loss_fn=tf.keras.losses.BinaryCrossentropy(),
            reconstruction_loss_fn=tf.keras.losses.MeanAbsoluteError(),
            adversarial_loss_weight=1.0,
            reconstruction_loss_weight=rec_weight,
            label_smoothing=0.0,
        )
        gan.compile()
        return gan

    gan_a = build_gan(1.0)
    gan_b = build_gan(3.0)
    metrics_a = gan_a.train_step((x, y))
    metrics_b = gan_b.train_step((x, y))

    assert float(metrics_b['g_loss']) > float(metrics_a['g_loss'])
    assert float(metrics_b['g_loss']) - float(metrics_a['g_loss']) == pytest.approx(2.0 * float(metrics_a['g_rec_loss']), rel=0.15)


@pytest.mark.unit
def test_sigmoid_generator_output_stays_in_unit_interval(tiny_gan):
    generated = tiny_gan(tf.random.uniform((2, 64, 64, 1)), training=False)

    assert tf.reduce_min(generated).numpy() >= 0.0
    assert tf.reduce_max(generated).numpy() <= 1.0


@pytest.mark.unit
@pytest.mark.parametrize('reconstruction_loss_fn', [tf.keras.losses.MeanAbsoluteError(), scale_invariant_mae, log_cosh_loss, ssim_loss])
def test_gan_compile_accepts_supported_reconstruction_losses(tiny_unet, tiny_discriminator, reconstruction_loss_fn):
    model = GAN(
        generator=tiny_unet,
        discriminator=tiny_discriminator,
        g_optimizer=tf.keras.optimizers.Adam(1e-3),
        d_optimizer=tf.keras.optimizers.Adam(1e-3),
        adversarial_loss_fn=tf.keras.losses.BinaryCrossentropy(),
        reconstruction_loss_fn=reconstruction_loss_fn,
        adversarial_loss_weight=1.0,
        reconstruction_loss_weight=10.0,
        label_smoothing=0.0,
    )

    model.compile()
