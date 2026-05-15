"""Focused checkpoint round-trip checks for U-Net and GAN models."""

from collections import Counter
from importlib.util import find_spec
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TF_AVAILABLE = find_spec('tensorflow') is not None

if TF_AVAILABLE:
    import tensorflow as tf

    from starter import load_config
    from src.models.network import GAN, get_discriminator, network
    from src.training.new_train import _instantiate_optimizer
    from src.training.utils import build_checkpoint_custom_objects, load_checkpoint_model, read_checkpoint_info, save_checkpoint_model


@unittest.skipUnless(TF_AVAILABLE, 'TensorFlow is not installed')
class CheckpointSerializationTests(unittest.TestCase):
    """Check that checkpoints keep model state and saved model info."""

    input_shape = (32, 32, 1)

    def _config(self, **overrides):
        cfg = load_config(**overrides)
        training_cfg = dict(cfg['training'])
        network_cfg = dict(training_cfg['network_kwargs'])
        discriminator_cfg = dict(training_cfg['discriminator_kwargs'])
        gan_cfg = dict(training_cfg['gan_kwargs'])
        network_cfg['depth'] = min(int(network_cfg['depth']), 3)
        network_cfg['n_of_initial_channels'] = 8
        discriminator_cfg['depth'] = min(int(discriminator_cfg['depth']), 2)
        discriminator_cfg['n_initial_filters'] = 8
        return cfg, training_cfg, network_cfg, discriminator_cfg, gan_cfg

    def _optimizer_kwargs(self, training_cfg, learning_rate=None):
        kwargs = {}
        if learning_rate is not None:
            kwargs['learning_rate'] = float(learning_rate)
        elif training_cfg.get('learning_rate') is not None:
            kwargs['learning_rate'] = float(training_cfg['learning_rate'])
        if training_cfg.get('beta_1') is not None:
            kwargs['beta_1'] = float(training_cfg['beta_1'])
        return kwargs

    def _serialize(self, value):
        return tf.keras.utils.serialize_keras_object(value)

    def test_unet_checkpoint_saves_model_info(self):
        cfg, training_cfg, network_cfg, _, _ = self._config(
            model_type='unet',
            attention=True,
            scaling='log_min_max',
            loss_name='ssim_loss',
            activation_name='ReLU',
            output_activation='tanh',
        )
        model = network(self.input_shape, **network_cfg)
        model.compile(
            optimizer=_instantiate_optimizer(training_cfg['optimizer'], self._optimizer_kwargs(training_cfg)),
            loss=training_cfg['g_loss_fn'],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = str(Path(tmpdir) / 'unet.keras')
            save_checkpoint_model(
                model,
                checkpoint_path,
                {
                    'model_type': 'UNET',
                    'scaling': training_cfg['scaling'],
                    'config': cfg,
                },
            )
            restored = load_checkpoint_model(
                checkpoint_path,
                custom_objects=build_checkpoint_custom_objects(),
            )
            checkpoint_info = read_checkpoint_info(checkpoint_path)

        self.assertEqual(checkpoint_info['model_type'], 'UNET')
        self.assertEqual(checkpoint_info['scaling'], 'log_min_max')
        self.assertEqual(restored.checkpoint_info['scaling'], 'log_min_max')
        self.assertEqual(self._serialize(model.loss), self._serialize(restored.loss))
        self.assertEqual(self._serialize(model.optimizer), self._serialize(restored.optimizer))
        self.assertGreater(Counter(layer.__class__.__name__ for layer in restored.layers)['Multiply'], 0)
        self.assertEqual(tf.keras.activations.serialize(restored.layers[-1].activation), 'tanh')

    def test_gan_checkpoint_saves_model_info(self):
        cfg, training_cfg, network_cfg, discriminator_cfg, gan_cfg = self._config(
            model_type='gan',
            attention=True,
            scaling='z_scale',
            loss_name='log_cosh_loss',
            activation_name='LeakyReLU',
        )
        generator = network(self.input_shape, **network_cfg)
        discriminator = get_discriminator(self.input_shape, **discriminator_cfg)
        model = GAN(
            generator=generator,
            discriminator=discriminator,
            g_optimizer=_instantiate_optimizer(training_cfg['optimizer'], self._optimizer_kwargs(training_cfg)),
            d_optimizer=_instantiate_optimizer(
                training_cfg['optimizer'],
                self._optimizer_kwargs(training_cfg, learning_rate=gan_cfg['d_learning_rate']),
            ),
            adversarial_loss_fn=gan_cfg['loss_fn'],
            reconstruction_loss_fn=training_cfg['g_loss_fn'],
            adversarial_loss_weight=gan_cfg['adversarial_loss_weight'],
            reconstruction_loss_weight=gan_cfg['reconstruction_loss_weight'],
            label_smoothing=gan_cfg['label_smoothing'],
            name='gan_checkpoint_test',
        )
        _ = model(tf.zeros((1, *self.input_shape)))
        model.compile()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = str(Path(tmpdir) / 'gan.keras')
            save_checkpoint_model(
                model,
                checkpoint_path,
                {
                    'model_type': 'GAN',
                    'scaling': training_cfg['scaling'],
                    'config': cfg,
                },
            )
            restored = load_checkpoint_model(
                checkpoint_path,
                custom_objects=build_checkpoint_custom_objects(),
            )
            checkpoint_info = read_checkpoint_info(checkpoint_path)

        self.assertIsInstance(restored, GAN)
        self.assertEqual(checkpoint_info['model_type'], 'GAN')
        self.assertEqual(checkpoint_info['scaling'], 'z_scale')
        self.assertEqual(restored.checkpoint_info['model_type'], 'GAN')
        self.assertEqual(restored.generator.checkpoint_info['scaling'], 'z_scale')
        self.assertEqual(self._serialize(model.adversarial_loss_fn), self._serialize(restored.adversarial_loss_fn))
        self.assertEqual(self._serialize(model.reconstruction_loss_fn), self._serialize(restored.reconstruction_loss_fn))
        self.assertEqual(self._serialize(model.g_optimizer), self._serialize(restored.g_optimizer))
        self.assertEqual(self._serialize(model.d_optimizer), self._serialize(restored.d_optimizer))
        self.assertGreater(Counter(layer.__class__.__name__ for layer in restored.generator.layers)['Multiply'], 0)


if __name__ == '__main__':
    unittest.main()
