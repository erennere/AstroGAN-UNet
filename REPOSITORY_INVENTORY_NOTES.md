# Repository Inventory Notes

## Main Pipeline Order

1. `src/data/mast.py`: query MAST metadata, optionally resolve product URLs, and optionally download FITS files. `src/bash/mast_and_create_dataset.sh` can invoke this stage.
2. `src/data/create_dataset.py`: filter metadata, split originals into training, eval, and test sets, crop FITS images, compute crop and original statistics, and write enriched CSV outputs.
3. `src/training/new_train.py`: build datasets, build or restore models, and train either generator-only or GAN mode. `src/bash/train_model_scenarios.sh` orchestrates sweep runs for this stage.
4. `src/evaluation/metrics.py`: discover model directories and checkpoints, reconstruct test data, extract sources, and write multimodel metrics and catalog outputs.
5. `src/evaluation/uncropped_metrics.py`: run full-image sliding-window evaluation and write uncropped metrics, catalogs, and histogram outputs.
6. `src/evaluation/merge_catalogs.py`: merge uncropped reconstructed, noisy, and original catalogs by spatial proximity into a parquet table.
7. `src/visualization/prepare_images.py`: create composite image panels and source-overlay images.
8. `src/visualization/prepare_plots.py`: build photometric summary CSVs and plot grids from evaluation outputs.

## Major Directories

1. `data`: data and metadata artifact root; the current tree contains alias-keyed dataset directories with downloaded metadata, filtered metadata, noise-enriched metadata, cropped-image statistics, and `training_images`, `eval_images`, and `test_images` FITS splits.
2. `histograms`: saved histogram images and grouped histogram folders such as `faber`, `normal_galaxy`, `noscal`, `old`, and `zscale`.
3. `legacy`: older standalone scripts for downloading, splitting, metadata assembly, and image generation.
4. `models`: model artifact root; `config.yaml` explicitly describes this as the output folder hierarchy by model and data regimes, including activation and output settings.
5. `notebooks`: Jupyter notebooks for interactive inspection, metrics work, plotting, and experiments.
6. `src`: active package code.
7. `src/bash`: HPC, SLURM, environment bootstrap, and orchestration scripts.
8. `src/data`: MAST query and dataset preparation code.
9. `src/evaluation`: evaluation, source extraction, uncropped evaluation, and catalog merge code.
10. `src/models`: active network definitions.
11. `src/training`: training loop, callback, math helpers, losses, checkpoint helpers, and dataset generators.
12. `src/visualization`: qualitative image composites and quantitative plotting.
13. `tests`: checkpoint serialization test coverage.

## Python Entrypoints

1. Current main-guarded entrypoints: `src/data/mast.py`, `src/data/create_dataset.py`, `src/training/new_train.py`, `src/evaluation/metrics.py`, `src/evaluation/uncropped_metrics.py`, `src/evaluation/merge_catalogs.py`, `src/visualization/prepare_images.py`, `src/visualization/prepare_plots.py`, and `tests/test_checkpoint_serialization.py`.
2. `src/evaluation/metrics.py` has two main-related guards: one early guard that sets CPU-only execution when launched directly and `CUDA_VISIBLE_DEVICES` is unset, and one bottom guard that runs the evaluation entrypoint.
3. Script-style legacy entry files with top-level executable code and no main guard: `legacy/create_new_image.py`, `legacy/down_data.py`, `legacy/original_data_split.py`, `legacy/original_down_data.py`, `legacy/original_images.py`, and `legacy/train_test_val_split.py`.
4. `starter.py` is not a CLI entrypoint; it is the shared config and alias-orchestration module imported by the active scripts.

## Bash Scripts and Which Python Files They Execute

1. `src/bash/mast_and_create_dataset.sh`: executes `src/data/mast.py` and `src/data/create_dataset.py`.
2. `src/bash/train_model_scenarios.sh`: executes `src/training/new_train.py` with runtime overrides.
3. `src/bash/check_gpu_hpc.sh`: does not execute a repository Python file; it runs an inline Python snippet that checks TensorFlow and PyTorch GPU visibility.
4. `src/bash/setup_hpc_environment.sh`: does not execute a repository Python file; it builds the environment from `environment.yml` and runs verification commands with the environment's Python.
5. `src/bash/setup_hpc_environment_slurm.sh`: does not execute a repository Python file directly; it delegates to `src/bash/setup_hpc_environment.sh`.

## Configuration Systems and Parameter Interfaces

1. `config.yaml` is the runtime configuration source. Its top-level sections are `paths`, `shared_bindings`, `data`, `network`, `discriminator`, `gan`, `training`, `mast`, `create_dataset`, `evaluation`, and `visualization`.
2. `starter.py` provides the runtime interface through `load_config` and `parse_config_overrides`. `load_config` reads YAML, merges overrides, normalizes path templates, resolves runtime symbols, applies shared bindings, builds evaluation payload dicts, and syncs training payload dicts.
3. The shared CLI override surface parsed by `parse_config_overrides` is: `nsigma`, `footprint_radius`, `npixels`, `model_type`, `attention`, `scaling`, `loss_name`, `dropout_rate`, `output_activation`, `kernel_initializer`, `activation_name`, `discriminator_activation`, `discriminator_output_activation`, `filter_surveys`, `filter_by_last_name`, and `last_name_filter_value`. The parser supports named flags and a positional fallback.
4. `starter.py` generates training payloads inside the loaded config: `training.data_kwargs`, `training.network_kwargs`, `training.discriminator_kwargs`, and `training.gan_kwargs`.
5. `starter.py` generates evaluation payloads inside the loaded config: `evaluation.data_kwargs`, `evaluation.model_kwargs`, and `evaluation.kwargs_source`.
6. `starter.py` also propagates shared values across sections. Explicit examples in the code are: `paths.metadata_csv` into `mast.metadata_output` and `create_dataset.metadata_filepath`, `create_dataset.cropped_stats_output_file` into `data.metadata_filepath` and `evaluation.metadata_filepath`, and `data.ps` into `training.patch_size` and `evaluation.patch_size`.
7. `starter.py` defines a reversible alias system for data and model signatures. The data alias encodes `filter_surveys`, `allowed_survey`, `filter_by_last_name`, `last_name_filter_value`, `nsigma`, `footprint_radius`, and `npixels`. The model alias encodes model type, attention, loss, data alias, scaling, dropout, activation, output activation, discriminator activation, and discriminator output activation. `src/evaluation/metrics.py` uses this path structure when it walks model directories.
8. `environment.yml` is the environment specification. It defines a conda environment named `astro-unets` with Python 3.11 and pip-installed TensorFlow with CUDA support, astronomy libraries, image-processing libraries, plotting libraries, async networking libraries, and test and notebook tools.
9. `src/bash/setup_hpc_environment.sh` exposes two script arguments, `--prefix` and `--clean-start`. `src/bash/mast_and_create_dataset.sh` and `src/bash/train_model_scenarios.sh` also expose runtime parameters through `AUN`-prefixed environment variables.
10. `src/training/callback.py` and `src/training/utils.py` define a checkpoint metadata interface by embedding `checkpoint_info.json` inside each saved Keras checkpoint; the stored fields explicitly include `model_type`, `scaling`, and `config`.

## Model Architectures Available

1. Active generator architecture in `src/models/network.py`: a configurable U-Net built by the `network` function.
2. Active attention variant in `src/models/network.py`: the same U-Net can enable attention gates on skip connections through the `attention` parameter and the `attention_gate` helper.
3. Active discriminator architecture in `src/models/network.py`: a configurable CNN discriminator built by `get_discriminator`.
4. Active composite architecture in `src/models/network.py`: the `GAN` class wraps a generator and discriminator and defines adversarial and reconstruction training logic.
5. Legacy architecture in `legacy/create_new_image.py`: a TensorFlow 1.x, `tf.contrib.slim` U-Net-like network with a custom `upsample_and_concat` path and a top-level session-based inference script.

## Training, Evaluation, Merge, and Visualization Stages

1. Training stage: `src/training/new_train.py` builds the model, creates TensorFlow datasets, restores checkpoints, and runs fit; `src/training/callback.py` handles validation, preview image generation, checkpoint saving, and training metrics logging; `src/training/utils.py` provides losses, checkpoint helpers, and dataset construction; `src/training/math_helpers.py` provides scaling and simulation helpers.
2. Evaluation stage: `src/evaluation/metrics.py` handles model discovery, checkpoint selection, test-image generation, reconstruction, source extraction, and multimodel metrics and catalog writing; `src/evaluation/uncropped_metrics.py` handles uncropped, full-image evaluation and writes uncropped result CSVs, catalog CSVs, histogram CSVs, and histogram PNGs.
3. Merge stage: `src/evaluation/merge_catalogs.py` reads uncropped rec, noisy, and original catalog CSVs, matches them with a KDTree-based nearest-neighbor merge, and writes a parquet table.
4. Visualization stage: `src/visualization/prepare_images.py` creates original, noisy, and reconstructed composite panels plus source-overlay images; `src/visualization/prepare_plots.py` reads the merged parquet and uncropped metrics CSV, adds derived photometric columns, writes `all_metrics.csv`, and saves plot grids.

## Parameter Sweep Functionality and Hyperparameter Exploration

1. Dataset-preparation sweep in `src/bash/mast_and_create_dataset.sh`: a three-job table varies `nsigma`, `footprint_radius`, `npixels`, `filter_surveys`, `filter_by_last_name`, and last-name filter values.
2. Training hyperparameter sweep in `src/bash/train_model_scenarios.sh`: the script explicitly builds a Cartesian product of model type with two values, attention with two values, scaling with four values, and loss with four values, for 64 total scenarios.
3. The fixed values in that training sweep are explicit in `src/bash/train_model_scenarios.sh`: `dropout_rate` is `0.2`, `kernel_initializer` is `he_normal`, `activation_name` is `LeakyReLU`, `discriminator_activation` is `LeakyReLU`, and `discriminator_output_activation` is `sigmoid`.
4. The training sweep also applies one explicit conditional rule in `src/bash/train_model_scenarios.sh`: when loss is `ssim_loss`, `output_activation` is `sigmoid`; otherwise it is `null`.
5. Worker-level experiment sharding in `src/bash/train_model_scenarios.sh`: `TOTAL_WORKERS` is `10`, and each worker deterministically takes the scenarios whose indices match its modulo shard.
6. Data-candidate exploration in `src/training/utils.py`: `candidates_based_on_range` sweeps sigma values over base and exponent bins; `candidates_based_on_ratio` sweeps exposure ratios built from `ratio_initial`, `ratio_count`, and `ratio_growth`.
7. Sampling strategy exploration in `src/training/utils.py`: `sample_fn` is registry-driven and can be `sigma_range`, `exposure_ratio`, or `exposure_ratio_v2`; `exposure_ratio_v2` uses quantiles, percentages, and `occurrences_per_col_D` for quantile-allocated selection.
8. Strategy selection is fully config-driven in `src/training/utils.py` and `starter.py`: `candidates_fn`, `post_filter_fn`, `sample_fn`, `sigma_kernel_fn`, `noise_fn`, and `stats_name_fn` are resolved from named registries.
9. Checkpoint exploration in `src/evaluation/metrics.py`: `get_model_by_modulo` selects checkpoints at a regular epoch interval, includes the final checkpoint, and is driven by `evaluation.n` as the modulo argument.

## Scaling, Normalization, Activation Functions, and Loss Function Options

1. Scaling modes explicitly documented in `config.yaml` and implemented in `src/training/math_helpers.py`: none or `null`, `z_scale`, `min_max`, and `log_min_max`.
2. Normalization functions in `src/training/math_helpers.py`: `min_max_normalization` with `inverse_min_max_normalization`, `zscore_normalization` with `inverse_zscore_normalization`, and `adaptive_log_transform_and_normalize` with `inverse_adaptive_log_transform_and_denormalize`.
3. Generator activation surface in `config.yaml` and `src/models/network.py`: `network.func` is explicitly documented with `LeakyReLU` and `ReLU`; runtime resolution in `starter.py` also resolves names from TensorFlow registries.
4. Output activation surface in `config.yaml` and `starter.py`: `null`, `relu`, `sigmoid`, and `tanh` are explicitly documented and encoded in the activation tag map.
5. Discriminator activation surface in `config.yaml`: `discriminator.func` defaults to `LeakyReLU`, and `discriminator.output_activation` defaults to `sigmoid`.
6. Active generator reconstruction losses in `config.yaml`, `src/training/utils.py`, and `src/bash/train_model_scenarios.sh`: `MeanAbsoluteError`, `scale_invariant_mae`, `log_cosh_loss`, and `ssim_loss`.
7. Active GAN adversarial loss in `config.yaml`: `BinaryCrossentropy`.
8. Loss-name encoding in `starter.py`: the loss code map explicitly includes `BCE`, `MSE`, `MAE`, `SSIM`, `SIMAE`, and `LOGCOSH` for model alias encoding.

## Dependencies Between Scripts

1. `starter.py` is the shared config dependency for all current main-guarded repository scripts and for `tests/test_checkpoint_serialization.py`.
2. `src/data/create_dataset.py` depends on `src/data/mast.py` for `download_images` and on `src/training/utils.py` for FITS and directory helpers.
3. `src/training/new_train.py` depends on `src/models/network.py`, `src/training/callback.py`, `src/training/utils.py`, `src/training/math_helpers.py`, and `starter.py`.
4. `src/evaluation/metrics.py` depends on `src/training/new_train.py` for `data_augment_pluggable`, on `src/data/create_dataset.py` for `crop_image_generator`, on `src/training/utils.py` for checkpoint and FITS helpers, on `src/training/math_helpers.py` for scaling and noise helpers, and on `starter.py` for config and model-path decoding.
5. `src/evaluation/uncropped_metrics.py` depends on `src/evaluation/metrics.py`, `src/data/create_dataset.py`, `src/training/utils.py`, and `starter.py`.
6. `src/evaluation/merge_catalogs.py` depends on `starter.py` for config and on `src/training/utils.py` for path helpers, and it consumes the uncropped catalog CSV outputs produced by `src/evaluation/uncropped_metrics.py`.
7. `src/visualization/prepare_images.py` depends on `src/evaluation/metrics.py` for model discovery and source-extraction helpers, on `src/data/create_dataset.py` for `crop_image_generator`, on `src/training/utils.py` for FITS and checkpoint helpers, on `src/training/math_helpers.py` for noise simulation, and on `starter.py` for config.
8. `src/visualization/prepare_plots.py` depends on `starter.py` for config and on `src/training/utils.py` for path helpers, and it consumes the parquet output from `src/evaluation/merge_catalogs.py` plus the uncropped results CSV from `src/evaluation/uncropped_metrics.py`.
9. `src/training/callback.py` and `src/training/utils.py` write checkpoint metadata into each Keras checkpoint; `src/evaluation/metrics.py` and `src/visualization/prepare_images.py` read that metadata to recover scaling and model type.
10. `src/bash/mast_and_create_dataset.sh` is the shell-level dependency bridge between metadata retrieval and dataset creation, and `src/bash/train_model_scenarios.sh` is the shell-level dependency bridge between the scenario matrix and model training.

## Evidence-Limited Notes

1. `models/training_config.json` appears to be an artifact-like config snapshot. No direct runtime reference to that filename was found in the active repository code that was inspected.
2. `legacy` appears separate from the active `src` pipeline and contains older standalone scripts.
3. In the inspected code, `src/evaluation/uncropped_metrics.py` uses `find_best_performing_models(...)` in a way that appears type-ambiguous relative to the inspected implementation in `src/evaluation/metrics.py`; this note is included as an observation only, not as a claim about runtime failure.
4. `legacy/original_images.py` references `update_file` from `src.data.mast`; that symbol was not confirmed during the inspected reads.