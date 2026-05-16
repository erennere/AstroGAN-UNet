"""Coverage for starter.py helper logic that does not require TensorFlow."""

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starter import (
    _component_tag,
    _decode_data_alias_plain,
    _decode_data_signature,
    _decode_model_alias,
    _decode_models_dir,
    _encode_data_alias_plain,
    _encode_data_signature,
    _encode_model_alias,
    _format_with_known_templates,
    _normalize_paths,
    parse_config_overrides,
)


class StarterHelperTests(unittest.TestCase):
    def test_data_alias_plain_round_trips(self):
        encoded = _encode_data_alias_plain('SV_IR_GRISM1024')

        self.assertNotEqual(encoded, 'SV_IR_GRISM1024')
        self.assertEqual(_decode_data_alias_plain(encoded), 'SV_IR_GRISM1024')
        self.assertEqual(_decode_data_alias_plain(_encode_data_alias_plain('')), '')

    def test_data_signature_round_trips_known_surveys_and_last_names(self):
        encoded = _encode_data_signature(
            filter_surveys=True,
            allowed_survey=['IR', 'GRISM1024'],
            filter_by_last_name=True,
            last_name_filter_value=['smith', 'jones'],
            nsigma=3,
            footprint_radius=12,
            npixels=8,
        )

        decoded = _decode_data_signature(encoded['data_alias_enriched_hex'])

        self.assertEqual(decoded['data_alias_plain'], 'SV_IR_GRISM1024_LNsmith_jones')
        self.assertEqual(decoded['allowed_survey'], ['IR', 'GRISM1024'])
        self.assertEqual(decoded['last_name_filter_value'], ['smith', 'jones'])
        self.assertTrue(decoded['filter_surveys'])
        self.assertTrue(decoded['filter_by_last_name'])
        self.assertEqual(decoded['nsigma'], 3)
        self.assertEqual(decoded['footprint_radius'], 12)
        self.assertEqual(decoded['npixels'], 8)

    def test_model_alias_round_trips_and_decodes_dropout(self):
        data_signature = _encode_data_signature(
            filter_surveys=True,
            allowed_survey=['IR'],
            filter_by_last_name=False,
            last_name_filter_value=[],
            nsigma=2,
            footprint_radius=10,
            npixels=6,
        )

        encoded = _encode_model_alias(
            is_gan='GAN',
            use_attention='ATTN',
            loss_function='SSIM',
            data_alias_enriched_hex=data_signature['data_alias_enriched_hex'],
            scaling_tag='z_scale',
            dropout_tag='0p25',
            activation_tag='relu',
            output_activation_tag='tanh',
            discriminator_activation_tag='leakyrelu',
            discriminator_output_activation_tag='sigmoid',
        )

        decoded = _decode_model_alias(encoded['model_alias_hex'])

        self.assertEqual(decoded['model_type'], 'gan')
        self.assertTrue(decoded['attention'])
        self.assertEqual(decoded['loss_function'], 'SSIM')
        self.assertEqual(decoded['dropout_rate'], 0.25)
        self.assertEqual(decoded['allowed_survey'], ['IR'])
        self.assertEqual(decoded['scaling_tag'], 'z_scale')
        self.assertEqual(decoded['output_activation_tag'], 'tanh')

    def test_decode_models_dir_extracts_all_tags(self):
        data_signature = _encode_data_signature(
            filter_surveys=False,
            allowed_survey=[],
            filter_by_last_name=True,
            last_name_filter_value=['doe'],
            nsigma=4,
            footprint_radius=9,
            npixels=11,
        )
        models_dir = (
            '/tmp/models/UNET/NOATTN/MAE/'
            f"{data_signature['data_alias_enriched_hex']}/min_max/DO0p5/ACTrelu/OUTsigmoid/DACTleakyrelu/DOUTnone"
        )

        decoded = _decode_models_dir(models_dir)

        self.assertEqual(decoded['models_root_dir'], '/tmp/models')
        self.assertEqual(decoded['model_type'], 'unet')
        self.assertFalse(decoded['attention'])
        self.assertEqual(decoded['loss_function'], 'MAE')
        self.assertEqual(decoded['dropout_rate'], 0.5)
        self.assertFalse(decoded['filter_surveys'])
        self.assertTrue(decoded['filter_by_last_name'])
        self.assertEqual(decoded['last_name_filter_value'], ['doe'])
        self.assertEqual(decoded['npixels'], 11)

    def test_parse_config_overrides_supports_named_flags(self):
        parsed = parse_config_overrides(
            argv=[
                'prog',
                '--nsigma', '5',
                '--footprint-radius', '17',
                '--npixels', '12',
                '--model-type', 'gan',
                '--attention', 'true',
                '--scaling', 'log_min_max',
                '--loss-name', 'ssim_loss',
                '--dropout-rate', '0.3',
                '--output-activation', 'tanh',
                '--kernel-initializer', 'he_normal',
                '--activation-name', 'ReLU',
                '--discriminator-activation', 'LeakyReLU',
                '--discriminator-output-activation', 'sigmoid',
                '--filter-surveys', 'yes',
                '--filter-by-last-name', '1',
                '--last-name-filter-value', 'alpha,beta',
            ],
        )

        self.assertEqual(parsed['nsigma'], 5)
        self.assertEqual(parsed['footprint_radius'], 17)
        self.assertEqual(parsed['npixels'], 12)
        self.assertEqual(parsed['model_type'], 'gan')
        self.assertTrue(parsed['attention'])
        self.assertEqual(parsed['scaling'], 'log_min_max')
        self.assertEqual(parsed['loss_name'], 'ssim_loss')
        self.assertEqual(parsed['dropout_rate'], 0.3)
        self.assertEqual(parsed['output_activation'], 'tanh')
        self.assertEqual(parsed['kernel_initializer'], 'he_normal')
        self.assertEqual(parsed['activation_name'], 'ReLU')
        self.assertEqual(parsed['discriminator_activation'], 'LeakyReLU')
        self.assertEqual(parsed['discriminator_output_activation'], 'sigmoid')
        self.assertTrue(parsed['filter_surveys'])
        self.assertTrue(parsed['filter_by_last_name'])
        self.assertEqual(parsed['last_name_filter_value'], ['alpha', 'beta'])

    def test_parse_config_overrides_supports_positional_values_and_null_strings(self):
        parsed = parse_config_overrides(
            argv=[
                'prog',
                '6',
                '14',
                '9',
                'unet',
                'false',
                'min_max',
                'mae',
                '0.15',
                'null',
                'none',
                'relu',
                'sigmoid',
                'none',
                '0',
                'false',
                '',
            ],
        )

        self.assertEqual(parsed['nsigma'], 6)
        self.assertEqual(parsed['footprint_radius'], 14)
        self.assertEqual(parsed['npixels'], 9)
        self.assertEqual(parsed['model_type'], 'unet')
        self.assertFalse(parsed['attention'])
        self.assertEqual(parsed['scaling'], 'min_max')
        self.assertEqual(parsed['loss_name'], 'mae')
        self.assertEqual(parsed['dropout_rate'], 0.15)
        self.assertIsNone(parsed['output_activation'])
        self.assertIsNone(parsed['kernel_initializer'])
        self.assertEqual(parsed['activation_name'], 'relu')
        self.assertEqual(parsed['discriminator_activation'], 'sigmoid')
        self.assertIsNone(parsed['discriminator_output_activation'])
        self.assertFalse(parsed['filter_surveys'])
        self.assertFalse(parsed['filter_by_last_name'])
        self.assertIsNone(parsed['last_name_filter_value'])

    def test_parse_config_overrides_rejects_invalid_booleans(self):
        with self.assertRaisesRegex(ValueError, 'attention'):
            parse_config_overrides(argv=['prog', '--attention', 'sometimes'])

    def test_format_with_known_templates_keeps_unknown_placeholders(self):
        formatted = _format_with_known_templates(
            'models/{name}/{missing}/{value:03d}/{repr!r}',
            {'name': 'demo', 'value': 7, 'repr': 'x'},
        )

        self.assertEqual(formatted, "models/demo/{missing}/007/'x'")

    def test_normalize_paths_formats_recursively_and_absolutizes_path_like_values(self):
        import os
        normalized = _normalize_paths(
            {
                'root': './outputs/{run}',
                'nested': ['../artifacts/{run}', '{run}', 3],
            },
            {'run': 'trial'},
        )

        # Use os.path.join to make the test cross-platform
        expected_root_ending = os.path.join('outputs', 'trial')
        expected_nested_0_ending = os.path.join('artifacts', 'trial')
        self.assertTrue(normalized['root'].endswith(expected_root_ending) or 
                       normalized['root'].replace('\\', '/').endswith('outputs/trial'))
        self.assertTrue(normalized['nested'][0].endswith(expected_nested_0_ending) or 
                       normalized['nested'][0].replace('\\', '/').endswith('artifacts/trial'))
        self.assertEqual(normalized['nested'][1], 'trial')
        self.assertEqual(normalized['nested'][2], 3)

    def test_component_tag_normalizes_none_blank_and_symbols(self):
        self.assertEqual(_component_tag(None), 'none')
        self.assertEqual(_component_tag('  '), 'none')
        self.assertEqual(_component_tag('ReLU++ Value'), 'relu___value')


if __name__ == '__main__':
    unittest.main()
