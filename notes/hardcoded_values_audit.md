# Hardcoded Values Audit (Scripts)

This table is generated from static literal scanning across script entrypoints and orchestration scripts.

| Lang | File | Line | Hardcoded value | Context |
| --- | --- | ---: | --- | --- |
| bash | src/bash/check_gpu_hpc.sh | 2 | #SBATCH --job-name=check-gpu | #SBATCH --job-name=check-gpu |
| bash | src/bash/check_gpu_hpc.sh | 3 | #SBATCH --output=../logs/check_gpu_%j.out | #SBATCH --output=../logs/check_gpu_%j.out |
| bash | src/bash/check_gpu_hpc.sh | 4 | #SBATCH --error=../errs/check_gpu_%j.err | #SBATCH --error=../errs/check_gpu_%j.err |
| bash | src/bash/check_gpu_hpc.sh | 5 | #SBATCH --partition=gpu-single | #SBATCH --partition=gpu-single |
| bash | src/bash/check_gpu_hpc.sh | 6 | #SBATCH --gres=gpu:1 | #SBATCH --gres=gpu:1 |
| bash | src/bash/check_gpu_hpc.sh | 7 | #SBATCH --mem=2gb | #SBATCH --mem=2gb |
| bash | src/bash/check_gpu_hpc.sh | 8 | #SBATCH --time=00:30:00 | #SBATCH --time=00:30:00 |
| bash | src/bash/check_gpu_hpc.sh | 68 | importlib.import_module("tensorflow") | tf = importlib.import_module("tensorflow") |
| bash | src/bash/check_gpu_hpc.sh | 70 | tf.config.list_physical_devices("GPU") | gpus = tf.config.list_physical_devices("GPU") |
| bash | src/bash/check_gpu_hpc.sh | 77 | importlib.import_module("torch") | torch = importlib.import_module("torch") |
| bash | src/bash/check_gpu_hpc.sh | 86 | len(result["tensorflow_visible_gpus"]) > 0 | tf_ok = len(result["tensorflow_visible_gpus"]) > 0 |
| bash | src/bash/check_gpu_hpc.sh | 87 | result["torch_cuda_available"] and result["torch_device_count"] > 0 | torch_ok = result["torch_cuda_available"] and result["torch_device_count"] > 0 |
| bash | src/bash/mast_and_create_dataset.sh | 2 | #SBATCH --job-name=mast-create-dataset | #SBATCH --job-name=mast-create-dataset |
| bash | src/bash/mast_and_create_dataset.sh | 3 | #SBATCH --output=../logs/mast_and_create_dataset_%j.out | #SBATCH --output=../logs/mast_and_create_dataset_%j.out |
| bash | src/bash/mast_and_create_dataset.sh | 4 | #SBATCH --error=../errs/mast_and_create_dataset_%j.err | #SBATCH --error=../errs/mast_and_create_dataset_%j.err |
| bash | src/bash/mast_and_create_dataset.sh | 5 | #SBATCH --partition=cpu-single | #SBATCH --partition=cpu-single |
| bash | src/bash/mast_and_create_dataset.sh | 6 | #SBATCH --cpus-per-task=16 | #SBATCH --cpus-per-task=16 |
| bash | src/bash/mast_and_create_dataset.sh | 7 | #SBATCH --mem=64gb | #SBATCH --mem=64gb |
| bash | src/bash/mast_and_create_dataset.sh | 8 | #SBATCH --time=96:00:00 | #SBATCH --time=96:00:00 |
| bash | src/bash/mast_and_create_dataset.sh | 9 | #SBATCH --array=1-3 | #SBATCH --array=1-3 |
| bash | src/bash/mast_and_create_dataset.sh | 37 | (                          ""        2      2      2      ) | NSIGMA=(                          ""        2      2      2      ) |
| bash | src/bash/mast_and_create_dataset.sh | 38 | (                       ""        10     10     10     ) | FP_RADIUS=(                       ""        10     10     10     ) |
| bash | src/bash/mast_and_create_dataset.sh | 39 | (                         ""        8      8      8      ) | NPIXELS=(                         ""        8      8      8      ) |
| bash | src/bash/mast_and_create_dataset.sh | 40 | (                  ""        true   true   true   ) | FILTER_SURVEYS=(                  ""        true   true   true   ) |
| bash | src/bash/mast_and_create_dataset.sh | 41 | (             ""        false  true   true   ) | FILTER_BY_LAST_NAME=(             ""        false  true   true   ) |
| bash | src/bash/mast_and_create_dataset.sh | 52 | "$(date +%Y%m%d_%H%M%S)" | LOG_TS="$(date +%Y%m%d_%H%M%S)" |
| bash | src/bash/mast_and_create_dataset.sh | 53 | "${SLURM_JOB_ID:-local}" | LOG_JOB_ID="${SLURM_JOB_ID:-local}" |
| bash | src/bash/mast_and_create_dataset.sh | 54 | "${AUN_LOG_DIR}/mast_and_create_dataset_${LOG_JOB_ID}_${LOG_TS}.log" | LOG_FILE="${AUN_LOG_DIR}/mast_and_create_dataset_${LOG_JOB_ID}_${LOG_TS}.log" |
| bash | src/bash/setup_hpc_environment.sh | 32 | "$(cd .. && pwd)" | PROJECT_ROOT="$(cd .. && pwd)" |
| bash | src/bash/setup_hpc_environment.sh | 33 | "$PROJECT_ROOT/logs" | LOGS_DIR="$PROJECT_ROOT/logs" |
| bash | src/bash/setup_hpc_environment.sh | 36 | "${HOME}/.local/miniconda3" | CONDA_INSTALL_PREFIX="${HOME}/.local/miniconda3" |
| bash | src/bash/setup_hpc_environment.sh | 38 | "${PROJECT_ROOT}/.venv"  # Use .venv for compatibility with existing scripts | ENV_PATH="${PROJECT_ROOT}/.venv"  # Use .venv for compatibility with existing scripts |
| bash | src/bash/setup_hpc_environment.sh | 39 | 0 | CLEAN_START=0 |
| bash | src/bash/setup_hpc_environment.sh | 42 | "latest"  # or use specific version like "24.11.2-0" | MINICONDA_VERSION="latest"  # or use specific version like "24.11.2-0" |
| bash | src/bash/setup_hpc_environment.sh | 44 | "x86_64" | MINICONDA_ARCH="x86_64" |
| bash | src/bash/setup_hpc_environment.sh | 45 | "Miniconda3-${MINICONDA_VERSION}-${MINICONDA_OS}-${MINICONDA_ARCH}.sh" | MINICONDA_INSTALLER="Miniconda3-${MINICONDA_VERSION}-${MINICONDA_OS}-${MINICONDA_ARCH}.sh" |
| bash | src/bash/setup_hpc_environment.sh | 46 | "https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}" | MINICONDA_URL="https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}" |
| bash | src/bash/setup_hpc_environment.sh | 50 | "$LOGS_DIR/setup_hpc_$(date +%Y%m%d_%H%M%S).log" | LOG_FILE="$LOGS_DIR/setup_hpc_$(date +%Y%m%d_%H%M%S).log" |
| bash | src/bash/setup_hpc_environment.sh | 56 | "$2" | CONDA_INSTALL_PREFIX="$2" |
| bash | src/bash/setup_hpc_environment.sh | 60 | 1 | CLEAN_START=1 |
| bash | src/bash/setup_hpc_environment_slurm.sh | 20 | #SBATCH --job-name=setup-astro-hpc | #SBATCH --job-name=setup-astro-hpc |
| bash | src/bash/setup_hpc_environment_slurm.sh | 21 | #SBATCH --output=../logs/setup_astro_hpc_%j.out | #SBATCH --output=../logs/setup_astro_hpc_%j.out |
| bash | src/bash/setup_hpc_environment_slurm.sh | 22 | #SBATCH --error=../errs/setup_astro_hpc_%j.err | #SBATCH --error=../errs/setup_astro_hpc_%j.err |
| bash | src/bash/setup_hpc_environment_slurm.sh | 23 | #SBATCH --partition=cpu-single | #SBATCH --partition=cpu-single |
| bash | src/bash/setup_hpc_environment_slurm.sh | 24 | #SBATCH --nodes=1 | #SBATCH --nodes=1 |
| bash | src/bash/setup_hpc_environment_slurm.sh | 25 | #SBATCH --cpus-per-task=16 | #SBATCH --cpus-per-task=16 |
| bash | src/bash/setup_hpc_environment_slurm.sh | 26 | #SBATCH --mem=16GB | #SBATCH --mem=16GB |
| bash | src/bash/setup_hpc_environment_slurm.sh | 27 | #SBATCH --time=02:00:00 | #SBATCH --time=02:00:00 |
| bash | src/bash/setup_hpc_environment_slurm.sh | 43 | "$(cd .. && pwd)" | PROJECT_ROOT="$(cd .. && pwd)" |
| bash | src/bash/train_model_scenarios.sh | 2 | #SBATCH --job-name=train-scenarios | #SBATCH --job-name=train-scenarios |
| bash | src/bash/train_model_scenarios.sh | 3 | #SBATCH --output=../logs/train_model_scenarios_%j.out | #SBATCH --output=../logs/train_model_scenarios_%j.out |
| bash | src/bash/train_model_scenarios.sh | 4 | #SBATCH --error=../errs/train_model_scenarios_%j.err | #SBATCH --error=../errs/train_model_scenarios_%j.err |
| bash | src/bash/train_model_scenarios.sh | 5 | #SBATCH --partition=gpu-single | #SBATCH --partition=gpu-single |
| bash | src/bash/train_model_scenarios.sh | 6 | #SBATCH --nodes=1 | #SBATCH --nodes=1 |
| bash | src/bash/train_model_scenarios.sh | 7 | #SBATCH --ntasks=1 | #SBATCH --ntasks=1 |
| bash | src/bash/train_model_scenarios.sh | 8 | #SBATCH --cpus-per-task=8 | #SBATCH --cpus-per-task=8 |
| bash | src/bash/train_model_scenarios.sh | 9 | #SBATCH --gres=gpu:A40:1 | #SBATCH --gres=gpu:A40:1 |
| bash | src/bash/train_model_scenarios.sh | 10 | #SBATCH --mem=32gb | #SBATCH --mem=32gb |
| bash | src/bash/train_model_scenarios.sh | 11 | #SBATCH --time=96:00:00 | #SBATCH --time=96:00:00 |
| bash | src/bash/train_model_scenarios.sh | 12 | #SBATCH --array=1-10%10 | #SBATCH --array=1-10%10 |
| bash | src/bash/train_model_scenarios.sh | 17 | 8 | N=8 |
| bash | src/bash/train_model_scenarios.sh | 18 | 10 | TOTAL_WORKERS=10 |
| bash | src/bash/train_model_scenarios.sh | 29 | "${SLURM_CPUS_PER_TASK}" | N="${SLURM_CPUS_PER_TASK}" |
| bash | src/bash/train_model_scenarios.sh | 56 | "$(date +%Y%m%d_%H%M%S)" | LOG_TS="$(date +%Y%m%d_%H%M%S)" |
| bash | src/bash/train_model_scenarios.sh | 57 | "${SLURM_JOB_ID:-local}" | LOG_JOB_ID="${SLURM_JOB_ID:-local}" |
| bash | src/bash/train_model_scenarios.sh | 58 | "${AUN_LOG_DIR}/train_model_scenarios_${LOG_JOB_ID}_${LOG_TS}.log" | LOG_FILE="${AUN_LOG_DIR}/train_model_scenarios_${LOG_JOB_ID}_${LOG_TS}.log" |
| bash | src/bash/train_model_scenarios.sh | 76 | ("false" "true") | ATTENTION_OPTIONS=("false" "true") |
| bash | src/bash/train_model_scenarios.sh | 218 | 1 | ran=1 |
| python | src/data/create_dataset.py | 55 | 2 | mean_val = round(df[exp_column].mean(), 2) |
| python | src/data/create_dataset.py | 56 | 2 | median_val = round(df[exp_column].median(), 2) |
| python | src/data/create_dataset.py | 57 | 2 | std_val = round(df[exp_column].std(), 2) |
| python | src/data/create_dataset.py | 59 | 10 | plt.figure(figsize=(10, 6)) |
| python | src/data/create_dataset.py | 59 | 6 | plt.figure(figsize=(10, 6)) |
| python | src/data/create_dataset.py | 60 | 30 | plt.hist(df[exp_column], bins=30, color='skyblue', edgecolor='black', label=f'N: {len(df)}') |
| python | src/data/create_dataset.py | 66 | 2 | plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val}') |
| python | src/data/create_dataset.py | 67 | 2 | plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val}') |
| python | src/data/create_dataset.py | 68 | 2 | plt.axvline(mean_val + std_val, color='orange', linestyle='dashed', linewidth=2, label=f'Std Dev: {std_val}') |
| python | src/data/create_dataset.py | 69 | 2 | plt.axvline(mean_val - std_val, color='orange', linestyle='dashed', linewidth=2) |
| python | src/data/create_dataset.py | 77 | 256 | def _iter_image_crops(image, ps=256): |
| python | src/data/create_dataset.py | 98 | 1 | yield i, j, image[i * ps:(i + 1) * ps, j * ps:(j + 1) * ps] |
| python | src/data/create_dataset.py | 101 | 256 | def crop_image_generator(image, ps=256): |
| python | src/data/create_dataset.py | 107 | 256 | def crop_image(image, filepath, save_dir, ps=256, crop_name_separator='_', output_extension='.fits', type_of_image='SCI'): |
| python | src/data/create_dataset.py | 107 | '_' | def crop_image(image, filepath, save_dir, ps=256, crop_name_separator='_', output_extension='.fits', type_of_image='SCI'): |
| python | src/data/create_dataset.py | 107 | '.fits' | def crop_image(image, filepath, save_dir, ps=256, crop_name_separator='_', output_extension='.fits', type_of_image='SCI'): |
| python | src/data/create_dataset.py | 110 | 'crop_image: filepath is None; cannot derive patch filename' | logging.warning('crop_image: filepath is None; cannot derive patch filename') |
| python | src/data/create_dataset.py | 114 | 'crop_image: save_dir is None; cannot save cropped files' | logging.warning('crop_image: save_dir is None; cannot save cropped files') |
| python | src/data/create_dataset.py | 118 | 0 | base_name = os.path.basename(filepath).split('.')[0] |
| python | src/data/create_dataset.py | 158 | 3.0 | def calculate_image_stats(data: np.ndarray, sigma: float = 3.0, n_sigma: float = 2.0, |
| python | src/data/create_dataset.py | 158 | 2.0 | def calculate_image_stats(data: np.ndarray, sigma: float = 3.0, n_sigma: float = 2.0, |
| python | src/data/create_dataset.py | 159 | 10 | n_pixels: int = 10, footprint_radius: int = 10, maxiters: int = 10, |
| python | src/data/create_dataset.py | 160 | 5 | step: int = 5, bkg_box_size: int = 64, exclude_percentile: float = 10.0): |
| python | src/data/create_dataset.py | 160 | 64 | step: int = 5, bkg_box_size: int = 64, exclude_percentile: float = 10.0): |
| python | src/data/create_dataset.py | 160 | 10.0 | step: int = 5, bkg_box_size: int = 64, exclude_percentile: float = 10.0): |
| python | src/data/create_dataset.py | 226 | 'make_source_mask' | if not hasattr(segment_img, 'make_source_mask'): |
| python | src/data/create_dataset.py | 227 | 'Segmentation object does not expose make_source_mask; returning' | logging.warning('Segmentation object does not expose make_source_mask; returning') |
| python | src/data/create_dataset.py | 270 | 0 | if step <= 0: |
| python | src/data/create_dataset.py | 271 | 1 | step = 1 |
| python | src/data/create_dataset.py | 273 | 0 | percentiles = np.arange(0, 101, step, dtype=int) |
| python | src/data/create_dataset.py | 273 | 101 | percentiles = np.arange(0, 101, step, dtype=int) |
| python | src/data/create_dataset.py | 274 | 100 | if percentiles[-1] != 100: |
| python | src/data/create_dataset.py | 275 | 100 | percentiles = np.append(percentiles, 100) |
| python | src/data/create_dataset.py | 279 | 100 | cutoff = len(data) * p // 100 |
| python | src/data/create_dataset.py | 282 | 0 | if subset.size == 0: |
| python | src/data/create_dataset.py | 283 | '_mean' | stats[f'{p}_mean'] = np.nan |
| python | src/data/create_dataset.py | 284 | '_median' | stats[f'{p}_median'] = np.nan |
| python | src/data/create_dataset.py | 285 | '_std' | stats[f'{p}_std'] = np.nan |
| python | src/data/create_dataset.py | 286 | '_max' | stats[f'{p}_max'] = np.nan |
| python | src/data/create_dataset.py | 289 | '_mean' | stats[f'{p}_mean'] = np.mean(subset) |
| python | src/data/create_dataset.py | 290 | '_median' | stats[f'{p}_median'] = np.median(subset) |
| python | src/data/create_dataset.py | 291 | '_std' | stats[f'{p}_std'] = np.abs(np.std(subset)) |
| python | src/data/create_dataset.py | 292 | '_max' | stats[f'{p}_max'] = np.max(subset) |
| python | src/data/create_dataset.py | 299 | 1000 | def filter_out_metadata(filepath, col, exp_column, allowed_survey, size=1000, low=100, high=10000, |
| python | src/data/create_dataset.py | 299 | 100 | def filter_out_metadata(filepath, col, exp_column, allowed_survey, size=1000, low=100, high=10000, |
| python | src/data/create_dataset.py | 299 | 10000 | def filter_out_metadata(filepath, col, exp_column, allowed_survey, size=1000, low=100, high=10000, |
| python | src/data/create_dataset.py | 300 | 42 | seed=42, temp_index_column='temp_index', max_iterations=1, filter_surveys=True, |
| python | src/data/create_dataset.py | 300 | 'temp_index' | seed=42, temp_index_column='temp_index', max_iterations=1, filter_surveys=True, |
| python | src/data/create_dataset.py | 357 | 'last_name_col must be specified when filter_by_last_name is True' | logging.warning('last_name_col must be specified when filter_by_last_name is True') |
| python | src/data/create_dataset.py | 365 | 1 | df = df.sample(frac=1, random_state=seed).reset_index(drop=True) |
| python | src/data/create_dataset.py | 366 | 0 | df[temp_index_column] = np.arange(0, len(df)) |
| python | src/data/create_dataset.py | 372 | 0 | iterations = 0 |
| python | src/data/create_dataset.py | 382 | 0 | if additional_size <= 0: |
| python | src/data/create_dataset.py | 385 | 2 | indices = np.random.normal(loc=len(subdf) // 2, scale=len(subdf) // 4, size=additional_size) |
| python | src/data/create_dataset.py | 385 | 4 | indices = np.random.normal(loc=len(subdf) // 2, scale=len(subdf) // 4, size=additional_size) |
| python | src/data/create_dataset.py | 387 | 0 | unique_indices = np.unique(np.clip(unique_indices, 0, len(subdf) - 1).astype(int)) |
| python | src/data/create_dataset.py | 387 | 1 | unique_indices = np.unique(np.clip(unique_indices, 0, len(subdf) - 1).astype(int)) |
| python | src/data/create_dataset.py | 395 | 1 | results.drop([temp_index_column], inplace=True, axis=1) |
| python | src/data/create_dataset.py | 400 | 60 | def test_train_validation_split(dataset_metadata, url_column, split=(60, 20, 20), seed=42): |
| python | src/data/create_dataset.py | 400 | 20 | def test_train_validation_split(dataset_metadata, url_column, split=(60, 20, 20), seed=42): |
| python | src/data/create_dataset.py | 400 | 42 | def test_train_validation_split(dataset_metadata, url_column, split=(60, 20, 20), seed=42): |
| python | src/data/create_dataset.py | 430 | 1 | test_size = total_len * split[1] // 100 |
| python | src/data/create_dataset.py | 430 | 100 | test_size = total_len * split[1] // 100 |
| python | src/data/create_dataset.py | 431 | 2 | val_size = total_len * split[2] // 100 |
| python | src/data/create_dataset.py | 431 | 100 | val_size = total_len * split[2] // 100 |
| python | src/data/create_dataset.py | 438 | 'Train/test/val split: %d / %d / %d (total=%d, split=%s)' | 'Train/test/val split: %d / %d / %d (total=%d, split=%s)', |
| python | src/data/create_dataset.py | 443 | 5 | def download_dataset(metadata, id_column, url_column, save_dir, max_requests=5, reset_after=10): |
| python | src/data/create_dataset.py | 443 | 10 | def download_dataset(metadata, id_column, url_column, save_dir, max_requests=5, reset_after=10): |
| python | src/data/create_dataset.py | 469 | 'Starting async download of %d images to %s (max_requests=%d)' | logging.info('Starting async download of %d images to %s (max_requests=%d)', len(metadata), save_dir, max_requests) |
| python | src/data/create_dataset.py | 478 | '_' | original_filename_column=None, crop_name_separator='_', |
| python | src/data/create_dataset.py | 479 | '_drz.fits' | crop_prefix_parts=1, original_filename_suffix='_drz.fits'): |
| python | src/data/create_dataset.py | 556 | 'mean_bkg' | stats_column_map['mean_bkg']: mean_bkg, |
| python | src/data/create_dataset.py | 557 | 'median_bkg' | stats_column_map['median_bkg']: median_bkg, |
| python | src/data/create_dataset.py | 558 | 'std_bkg' | stats_column_map['std_bkg']: std_bkg, |
| python | src/data/create_dataset.py | 559 | 'max_bkg' | stats_column_map['max_bkg']: max_bkg, |
| python | src/data/create_dataset.py | 560 | 'abs_mean' | stats_column_map['abs_mean']: abs_mean, |
| python | src/data/create_dataset.py | 561 | 'abs_median' | stats_column_map['abs_median']: abs_median, |
| python | src/data/create_dataset.py | 562 | 'mean_src' | stats_column_map['mean_src']: mean_src, |
| python | src/data/create_dataset.py | 563 | 'median_src' | stats_column_map['median_src']: median_src, |
| python | src/data/create_dataset.py | 564 | 'std_src' | stats_column_map['std_src']: std_src, |
| python | src/data/create_dataset.py | 565 | 'max_src' | stats_column_map['max_src']: max_src, |
| python | src/data/create_dataset.py | 602 | 1000 | size=1000, low=100, high=1000, seed=42, split=(60, 20, 20), |
| python | src/data/create_dataset.py | 602 | 100 | size=1000, low=100, high=1000, seed=42, split=(60, 20, 20), |
| python | src/data/create_dataset.py | 602 | 42 | size=1000, low=100, high=1000, seed=42, split=(60, 20, 20), |
| python | src/data/create_dataset.py | 602 | 60 | size=1000, low=100, high=1000, seed=42, split=(60, 20, 20), |
| python | src/data/create_dataset.py | 602 | 20 | size=1000, low=100, high=1000, seed=42, split=(60, 20, 20), |
| python | src/data/create_dataset.py | 603 | 5 | max_requests=5, reset_after=10, type_of_image='SCI', sigma=3, n_sigma=2, |
| python | src/data/create_dataset.py | 603 | 10 | max_requests=5, reset_after=10, type_of_image='SCI', sigma=3, n_sigma=2, |
| python | src/data/create_dataset.py | 603 | 3 | max_requests=5, reset_after=10, type_of_image='SCI', sigma=3, n_sigma=2, |
| python | src/data/create_dataset.py | 603 | 2 | max_requests=5, reset_after=10, type_of_image='SCI', sigma=3, n_sigma=2, |
| python | src/data/create_dataset.py | 604 | 10 | n_pixels=10, footprint_radius=10, maxiters=10, bkg_box_size=64, |
| python | src/data/create_dataset.py | 604 | 64 | n_pixels=10, footprint_radius=10, maxiters=10, bkg_box_size=64, |
| python | src/data/create_dataset.py | 605 | 10.0 | exclude_percentile=10.0, ps=256, max_workers=16, step=5, filter_surveys=True, |
| python | src/data/create_dataset.py | 605 | 256 | exclude_percentile=10.0, ps=256, max_workers=16, step=5, filter_surveys=True, |
| python | src/data/create_dataset.py | 605 | 16 | exclude_percentile=10.0, ps=256, max_workers=16, step=5, filter_surveys=True, |
| python | src/data/create_dataset.py | 605 | 5 | exclude_percentile=10.0, ps=256, max_workers=16, step=5, filter_surveys=True, |
| python | src/data/create_dataset.py | 696 | 'control_flow started (download=%s, cropping=%s, stats_on_crops=%s)' | 'control_flow started (download=%s, cropping=%s, stats_on_crops=%s)', |
| python | src/data/create_dataset.py | 709 | '── Phase 1: filter / download / crop ──' | logging.info('── Phase 1: filter / download / crop ──') |
| python | src/data/create_dataset.py | 800 | 'Collected stats for %d/%d original images in split "%s"' | logging.info('Collected stats for %d/%d original images in split "%s"', len(noise_attributes), len(crops_to_process), save_dir) |
| python | src/data/create_dataset.py | 828 | 'Collected stats for %d/%d cropped patches in split "%s"' | logging.info('Collected stats for %d/%d cropped patches in split "%s"', len(cropped_image_attributes), len(crops_to_process), folder) |
| python | src/data/create_dataset.py | 853 | 'stats_on_crops=True but cropping=False: Phase 2 was skipped so noisy_filtered_metadata_output_file does not exist and cropped stats were not collected. Set cropping=True or run Phase 2 first.' | 'stats_on_crops=True but cropping=False: Phase 2 was skipped so ' |
| python | src/data/create_dataset.py | 909 | 'create_dataset' | dataset_cfg = cfg['create_dataset'] |
| python | src/data/create_dataset.py | 912 | 'dataset_dir' | dataset_dir=dataset_cfg['dataset_dir'], |
| python | src/data/create_dataset.py | 913 | 'metadata_filepath' | metadata_filepath=dataset_cfg['metadata_filepath'], |
| python | src/data/create_dataset.py | 914 | 'survey_column' | survey_column=dataset_cfg['survey_column'], |
| python | src/data/create_dataset.py | 915 | 'exp_column' | exp_column=dataset_cfg['exp_column'], |
| python | src/data/create_dataset.py | 916 | 'id_column' | id_column=dataset_cfg['id_column'], |
| python | src/data/create_dataset.py | 917 | 'url_column' | url_column=dataset_cfg['url_column'], |
| python | src/data/create_dataset.py | 918 | 'allowed_survey' | allowed_survey=dataset_cfg['allowed_survey'], |
| python | src/data/create_dataset.py | 919 | 'filter_surveys' | filter_surveys=dataset_cfg['filter_surveys'], |
| python | src/data/create_dataset.py | 920 | 'split_dirs' | split_dirs=dataset_cfg['split_dirs'], |
| python | src/data/create_dataset.py | 921 | 'originals_subdir' | originals_subdir=dataset_cfg['originals_subdir'], |
| python | src/data/create_dataset.py | 922 | 'masked_images_dirname' | masked_images_dirname=dataset_cfg['masked_images_dirname'], |
| python | src/data/create_dataset.py | 923 | 'file_extension' | file_extension=dataset_cfg['file_extension'], |
| python | src/data/create_dataset.py | 924 | 'filtered_metadata_output_file' | filtered_metadata_output_file=dataset_cfg['filtered_metadata_output_file'], |
| python | src/data/create_dataset.py | 925 | 'noisy_filtered_metadata_output_file' | noisy_filtered_metadata_output_file=dataset_cfg['noisy_filtered_metadata_output_file'], |
| python | src/data/create_dataset.py | 926 | 'cropped_stats_output_file' | cropped_stats_output_file=dataset_cfg['cropped_stats_output_file'], |
| python | src/data/create_dataset.py | 927 | 'url_filename_split_token' | url_filename_split_token=dataset_cfg['url_filename_split_token'], |
| python | src/data/create_dataset.py | 928 | 'crop_name_separator' | crop_name_separator=dataset_cfg['crop_name_separator'], |
| python | src/data/create_dataset.py | 929 | 'crop_prefix_parts' | crop_prefix_parts=dataset_cfg['crop_prefix_parts'], |
| python | src/data/create_dataset.py | 930 | 'original_filename_suffix' | original_filename_suffix=dataset_cfg['original_filename_suffix'], |
| python | src/data/create_dataset.py | 931 | 'stats_column_tokens' | stats_column_tokens=dataset_cfg['stats_column_tokens'], |
| python | src/data/create_dataset.py | 932 | 'temp_index_column' | temp_index_column=dataset_cfg['temp_index_column'], |
| python | src/data/create_dataset.py | 933 | 'filename_column' | filename_column=dataset_cfg['filename_column'], |
| python | src/data/create_dataset.py | 934 | 'original_filename_column' | original_filename_column=dataset_cfg['original_filename_column'], |
| python | src/data/create_dataset.py | 935 | 'location_col' | location_col=dataset_cfg['location_col'], |
| python | src/data/create_dataset.py | 936 | 'masked_filename_prefix' | masked_filename_prefix=dataset_cfg['masked_filename_prefix'], |
| python | src/data/create_dataset.py | 937 | 'stats_column_map' | stats_column_map=dataset_cfg['stats_column_map'], |
| python | src/data/create_dataset.py | 938 | 'original_stats_prefix' | original_stats_prefix=dataset_cfg['original_stats_prefix'], |
| python | src/data/create_dataset.py | 939 | 'max_iterations' | max_iterations=dataset_cfg['max_iterations'], |
| python | src/data/create_dataset.py | 940 | 'nan_value' | nan_value=dataset_cfg['nan_value'], |
| python | src/data/create_dataset.py | 941 | 'posinf_value' | posinf_value=dataset_cfg['posinf_value'], |
| python | src/data/create_dataset.py | 942 | 'neginf_value' | neginf_value=dataset_cfg['neginf_value'], |
| python | src/data/create_dataset.py | 945 | 'stats_on_crops' | stats_on_crops=dataset_cfg['stats_on_crops'], |
| python | src/data/create_dataset.py | 952 | 'max_requests' | max_requests=dataset_cfg['max_requests'], |
| python | src/data/create_dataset.py | 953 | 'reset_after' | reset_after=dataset_cfg['reset_after'], |
| python | src/data/create_dataset.py | 954 | 'type_of_image' | type_of_image=dataset_cfg['type_of_image'], |
| python | src/data/create_dataset.py | 958 | 'footprint_radius' | footprint_radius=dataset_cfg['footprint_radius'], |
| python | src/data/create_dataset.py | 960 | 'bkg_box_size' | bkg_box_size=dataset_cfg['bkg_box_size'], |
| python | src/data/create_dataset.py | 961 | 'exclude_percentile' | exclude_percentile=dataset_cfg['exclude_percentile'], |
| python | src/data/create_dataset.py | 963 | 'max_workers' | max_workers=dataset_cfg['max_workers'], |
| python | src/data/create_dataset.py | 965 | 'filter_by_last_name' | filter_by_last_name=dataset_cfg['filter_by_last_name'], |
| python | src/data/create_dataset.py | 966 | 'last_name_filter_value' | last_name_filter_value=dataset_cfg['last_name_filter_value'], |
| python | src/data/create_dataset.py | 967 | 'last_name_col' | last_name_col=dataset_cfg['last_name_col'], |
| python | src/data/create_dataset.py | 970 | '__main__' | if __name__ == "__main__": |
| python | src/data/mast.py | 41 | 1 | length = 1 |
| python | src/data/mast.py | 42 | 0 | offset = 0 |
| python | src/data/mast.py | 43 | 5000 | limit = 5000 |
| python | src/data/mast.py | 64 | 0 | index = 0 |
| python | src/data/mast.py | 74 | 0 | length = 0 |
| python | src/data/mast.py | 78 | 0 | length = 0 |
| python | src/data/mast.py | 93 | 20 | def plot_histogram(data, bins=20, label=None, xlabel='Exposure Time (s)', ylabel='Frequency', title='Histogram of Exposure', output_filename='histogram.png',loc='upper left', c='b', rotation=60): |
| python | src/data/mast.py | 93 | 'histogram.png' | def plot_histogram(data, bins=20, label=None, xlabel='Exposure Time (s)', ylabel='Frequency', title='Histogram of Exposure', output_filename='histogram.png',loc='upper left', c='b', rotation=60): |
| python | src/data/mast.py | 93 | 60 | def plot_histogram(data, bins=20, label=None, xlabel='Exposure Time (s)', ylabel='Frequency', title='Histogram of Exposure', output_filename='histogram.png',loc='upper left', c='b', rotation=60): |
| python | src/data/mast.py | 133 | 2 | label = f'mean: {round(mean, 2)}, median: {round(median, 2)}\nstd: {round(std, 2)}, var: {round(var, 2)}\n N: {len(data)}' |
| python | src/data/mast.py | 136 | 0.75 | counts, bins_edges, patches = plt.hist(data, bins=bins, density=False, alpha=0.75, color=c, edgecolor='black', label=label) |
| python | src/data/mast.py | 137 | 1 | xticks = np.linspace(bins_edges.min(), bins_edges.max(), bins+1) |
| python | src/data/mast.py | 138 | 1 | diff = (xticks[1] - xticks[0])/2 |
| python | src/data/mast.py | 138 | 0 | diff = (xticks[1] - xticks[0])/2 |
| python | src/data/mast.py | 138 | 2 | diff = (xticks[1] - xticks[0])/2 |
| python | src/data/mast.py | 139 | 1 | plt.xticks(np.linspace(bins_edges.min() + diff, bins_edges.max() + diff, bins+1), rotation=rotation) |
| python | src/data/mast.py | 141 | '--' | plt.axvline(mean, color='r', linestyle='--', label='mean') |
| python | src/data/mast.py | 142 | '--' | plt.axvline(median, color='purple', linestyle='--', label='median') |
| python | src/data/mast.py | 150 | 300 | plt.savefig(output_filename, dpi=300) |
| python | src/data/mast.py | 186 | 15 | session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) |
| python | src/data/mast.py | 188 | 200 | if response.status == 200: |
| python | src/data/mast.py | 200 | 5 | async def download_images(ids, urls, save_dir, max_requests=5, reset_after=10): |
| python | src/data/mast.py | 200 | 10 | async def download_images(ids, urls, save_dir, max_requests=5, reset_after=10): |
| python | src/data/mast.py | 221 | 'Starting bulk download to %s with max_requests=%d reset_after=%d' | logging.info('Starting bulk download to %s with max_requests=%d reset_after=%d', save_dir, max_requests, reset_after) |
| python | src/data/mast.py | 224 | 0 | success_count = 0 |
| python | src/data/mast.py | 225 | 0 | attempted_count = 0 |
| python | src/data/mast.py | 228 | 0 | if i % reset_after == 0: |
| python | src/data/mast.py | 231 | 15 | session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) |
| python | src/data/mast.py | 244 | 'Bulk download finished: %d/%d files downloaded.' | logging.info('Bulk download finished: %d/%d files downloaded.', success_count, attempted_count) |
| python | src/data/mast.py | 249 | 0 | if chunk_size <= 0: |
| python | src/data/mast.py | 250 | 1 | chunk_size = 1 |
| python | src/data/mast.py | 251 | 0 | for i in range(0, len(items), chunk_size): |
| python | src/data/mast.py | 264 | '.fits' | df = df[filename.str.lower().str.endswith(f'{prefer_token}.fits')] |
| python | src/data/mast.py | 277 | 'calib_level' | if 'calib_level' in df.columns: |
| python | src/data/mast.py | 278 | 'calib_level' | calib_numeric = pd.to_numeric(df['calib_level'], errors='coerce') |
| python | src/data/mast.py | 279 | '_calib_level_num' | df = df.assign(_calib_level_num=calib_numeric).sort_values('_calib_level_num', ascending=False, na_position='last') |
| python | src/data/mast.py | 281 | 0 | return df.iloc[0] if not df.empty else None |
| python | src/data/mast.py | 283 | 2 | def _resolve_products_bulk(dataset_ids, prefer_token='drz', max_retries=2, retry_delay=2.0): |
| python | src/data/mast.py | 283 | 2.0 | def _resolve_products_bulk(dataset_ids, prefer_token='drz', max_retries=2, retry_delay=2.0): |
| python | src/data/mast.py | 310 | 1 | attempts = max(1, int(max_retries) + 1) |
| python | src/data/mast.py | 311 | 1 | for attempt in range(1, attempts + 1): |
| python | src/data/mast.py | 314 | 0 | if obs is None or len(obs) == 0: |
| python | src/data/mast.py | 321 | 'to_pandas' | products_df = products.to_pandas() if hasattr(products, 'to_pandas') else pd.DataFrame(products) |
| python | src/data/mast.py | 324 | 'obs_id' | key_col = next((col for col in ['obs_id', 'obsid', 'obsID'] if col in products_df.columns), None) |
| python | src/data/mast.py | 343 | 'https://mast.stsci.edu/api/v0.1/Download/file?uri=' | url = f'https://mast.stsci.edu/api/v0.1/Download/file?uri={data_uri}' |
| python | src/data/mast.py | 357 | 'Bulk resolution attempt %d/%d failed for %d ids: %s. Retrying in %.1fs.' | 'Bulk resolution attempt %d/%d failed for %d ids: %s. Retrying in %.1fs.', |
| python | src/data/mast.py | 378 | 8 | max_workers=8, |
| python | src/data/mast.py | 380 | 500 | chunk_size=500, |
| python | src/data/mast.py | 381 | 2 | max_retries=2, |
| python | src/data/mast.py | 382 | 2.0 | retry_delay=2.0, |
| python | src/data/mast.py | 430 | 'Resolving product URLs for %d dataset ids via bulk Observations.get_product_list.' | logging.info('Resolving product URLs for %d dataset ids via bulk Observations.get_product_list.', len(unresolved_ids)) |
| python | src/data/mast.py | 432 | 1 | chunk_size = max(1, int(chunk_size)) |
| python | src/data/mast.py | 436 | 1 | if total_chunks == 1 or int(max_workers) <= 1: |
| python | src/data/mast.py | 437 | 1 | for idx, id_chunk in enumerate(chunks, start=1): |
| python | src/data/mast.py | 438 | 'Resolving chunk %d/%d (%d ids).' | logging.info('Resolving chunk %d/%d (%d ids).', idx, total_chunks, len(id_chunk)) |
| python | src/data/mast.py | 459 | 1 | for idx, id_chunk in enumerate(chunks, start=1) |
| python | src/data/mast.py | 464 | 'Resolved chunk %d/%d (%d ids).' | logging.info('Resolved chunk %d/%d (%d ids).', idx, total_chunks, chunk_len) |
| python | src/data/mast.py | 468 | 'Chunk %d/%d failed in process pool: %s. Retrying in parent process.' | logging.warning('Chunk %d/%d failed in process pool: %s. Retrying in parent process.', idx, total_chunks, err) |
| python | src/data/mast.py | 483 | 'Product merge complete: %d/%d rows have %s.' | logging.info('Product merge complete: %d/%d rows have %s.', resolved_count, len(merged), url_column) |
| python | src/data/mast.py | 503 | 'main_column' | 'main_column', |
| python | src/data/mast.py | 504 | 'max_requests' | 'max_requests', |
| python | src/data/mast.py | 505 | 'reset_after' | 'reset_after', |
| python | src/data/mast.py | 506 | 'max_workers' | 'max_workers', |
| python | src/data/mast.py | 508 | 'chunk_size' | 'chunk_size', |
| python | src/data/mast.py | 509 | 'resolve_max_retries' | 'resolve_max_retries', |
| python | src/data/mast.py | 510 | 'resolve_retry_delay' | 'resolve_retry_delay', |
| python | src/data/mast.py | 511 | 'metadata_output' | 'metadata_output', |
| python | src/data/mast.py | 512 | 'id_column' | 'id_column', |
| python | src/data/mast.py | 513 | 'url_column' | 'url_column', |
| python | src/data/mast.py | 514 | 'fetch_metadata' | 'fetch_metadata', |
| python | src/data/mast.py | 515 | 'resolve_urls' | 'resolve_urls', |
| python | src/data/mast.py | 522 | 'metadata_output' | metadata_output = os.path.abspath(cfg['metadata_output']) |
| python | src/data/mast.py | 523 | 'Loaded mast config: mission=%s fetch_metadata=%s resolve_urls=%s' | logging.info('Loaded mast config: mission=%s fetch_metadata=%s resolve_urls=%s', |
| python | src/data/mast.py | 524 | 'fetch_metadata' | cfg['mission'], cfg['fetch_metadata'], cfg['resolve_urls']) |
| python | src/data/mast.py | 524 | 'resolve_urls' | cfg['mission'], cfg['fetch_metadata'], cfg['resolve_urls']) |
| python | src/data/mast.py | 527 | 'fetch_metadata' | if cfg['fetch_metadata']: |
| python | src/data/mast.py | 532 | 'main_column' | table.sort_values(by=[cfg['main_column']], ascending=False, inplace=True) |
| python | src/data/mast.py | 537 | 'resolve_urls' | if cfg['resolve_urls']: |
| python | src/data/mast.py | 549 | 'id_column' | id_column=cfg['id_column'], |
| python | src/data/mast.py | 550 | 'url_column' | url_column=cfg['url_column'], |
| python | src/data/mast.py | 551 | 'max_workers' | max_workers=cfg['max_workers'], |
| python | src/data/mast.py | 552 | 'prefer_token' | prefer_token=cfg['prefer_token'], |
| python | src/data/mast.py | 553 | 'chunk_size' | chunk_size=cfg['chunk_size'], |
| python | src/data/mast.py | 554 | 'resolve_max_retries' | max_retries=cfg['resolve_max_retries'], |
| python | src/data/mast.py | 555 | 'resolve_retry_delay' | retry_delay=cfg['resolve_retry_delay'], |
| python | src/data/mast.py | 562 | 'url_column' | if cfg['download'] and cfg['url_column'] in table.columns: |
| python | src/data/mast.py | 563 | 'id_column' | valid = table[[cfg['id_column'], cfg['url_column']]].dropna(subset=[cfg['url_column']]) |
| python | src/data/mast.py | 563 | 'url_column' | valid = table[[cfg['id_column'], cfg['url_column']]].dropna(subset=[cfg['url_column']]) |
| python | src/data/mast.py | 567 | 'id_column' | valid[cfg['id_column']].tolist(), |
| python | src/data/mast.py | 568 | 'url_column' | valid[cfg['url_column']].tolist(), |
| python | src/data/mast.py | 569 | 'save_dir' | cfg['save_dir'], |
| python | src/data/mast.py | 570 | 'max_requests' | max_requests=cfg['max_requests'], |
| python | src/data/mast.py | 571 | 'reset_after' | reset_after=cfg['reset_after'], |
| python | src/data/mast.py | 576 | '__main__' | if __name__ == '__main__': |
| python | src/evaluation/merge_catalogs.py | 16 | 3.0 | def merge_based_on_proximity(df_rec, df_noise, df_org, threshold=3.0): |
| python | src/evaluation/merge_catalogs.py | 44 | 'image_id' | for image_id in df_rec['image_id'].unique(): |
| python | src/evaluation/merge_catalogs.py | 45 | 'image_id' | df_rec_filtered = df_rec[df_rec['image_id'] == image_id] |
| python | src/evaluation/merge_catalogs.py | 46 | 'image_id' | df_noise_filtered = df_noise[df_noise['image_id'] == image_id] |
| python | src/evaluation/merge_catalogs.py | 47 | 'image_id' | df_org_filtered = df_org[df_org['image_id'] == image_id].reset_index(drop=True) |
| python | src/evaluation/merge_catalogs.py | 49 | 'new_exp_time' | exp_times = df_rec_filtered['new_exp_time'].unique() |
| python | src/evaluation/merge_catalogs.py | 51 | 'new_exp_time' | df_rec_second_filtered = df_rec_filtered[df_rec_filtered['new_exp_time'] == new_exp_time].reset_index(drop=True) |
| python | src/evaluation/merge_catalogs.py | 52 | 'new_exp_time' | df_noise_second_filtered = df_noise_filtered[df_noise_filtered['new_exp_time'] == new_exp_time].reset_index(drop=True) |
| python | src/evaluation/merge_catalogs.py | 61 | 0 | if len(org_coords) > 0 and len(rec_coords) > 0: |
| python | src/evaluation/merge_catalogs.py | 63 | 1 | dist, indices_org = tree.query(rec_coords, k=1) |
| python | src/evaluation/merge_catalogs.py | 70 | 'image_id' | row_rec_renamed = row_rec.rename(lambda x: x + '_rec' if x != 'image_id' else x).to_dict() |
| python | src/evaluation/merge_catalogs.py | 70 | '_rec' | row_rec_renamed = row_rec.rename(lambda x: x + '_rec' if x != 'image_id' else x).to_dict() |
| python | src/evaluation/merge_catalogs.py | 72 | 0 | if len(org_coords) > 0 and valid_matches_org[i]: |
| python | src/evaluation/merge_catalogs.py | 73 | 0 | row_org = df_org_filtered.iloc[indices_org[i][0]] |
| python | src/evaluation/merge_catalogs.py | 74 | 'image_id' | row_org_renamed = row_org.rename(lambda x: x + '_org' if x != 'image_id' else x).to_dict() |
| python | src/evaluation/merge_catalogs.py | 74 | '_org' | row_org_renamed = row_org.rename(lambda x: x + '_org' if x != 'image_id' else x).to_dict() |
| python | src/evaluation/merge_catalogs.py | 81 | 'image_id' | if col != 'image_id': |
| python | src/evaluation/merge_catalogs.py | 82 | '_org' | row_rec_renamed[col + '_org'] = np.nan |
| python | src/evaluation/merge_catalogs.py | 89 | 0 | if len(noise_coords) > 0 and len(rec_coords) > 0: |
| python | src/evaluation/merge_catalogs.py | 91 | 1 | dist, indices_noise = tree.query(rec_coords, k=1) |
| python | src/evaluation/merge_catalogs.py | 100 | 0 | if len(noise_coords) > 0 and valid_matches_noise[i]: |
| python | src/evaluation/merge_catalogs.py | 101 | 0 | row_noise = df_noise_second_filtered.iloc[indices_noise[i][0]] |
| python | src/evaluation/merge_catalogs.py | 102 | 'image_id' | row_noise_renamed = row_noise.rename(lambda x: x + '_noise' if x != 'image_id' else x).to_dict() |
| python | src/evaluation/merge_catalogs.py | 102 | '_noise' | row_noise_renamed = row_noise.rename(lambda x: x + '_noise' if x != 'image_id' else x).to_dict() |
| python | src/evaluation/merge_catalogs.py | 107 | 'image_id' | if col != 'image_id': |
| python | src/evaluation/merge_catalogs.py | 108 | '_noise' | row_rec_dict[col + '_noise'] = np.nan |
| python | src/evaluation/merge_catalogs.py | 111 | 0 | matched_org_indices = set(indices_org[valid_matches_org, 0].tolist()) if len(org_coords) > 0 else set() |
| python | src/evaluation/merge_catalogs.py | 112 | 0 | matched_noise_indices = set(indices_noise[valid_matches_noise, 0].tolist()) if len(noise_coords) > 0 else set() |
| python | src/evaluation/merge_catalogs.py | 119 | 'image_id' | new_row = row_org.rename(lambda x: x + '_org' if x != 'image_id' else x).to_dict() |
| python | src/evaluation/merge_catalogs.py | 119 | '_org' | new_row = row_org.rename(lambda x: x + '_org' if x != 'image_id' else x).to_dict() |
| python | src/evaluation/merge_catalogs.py | 121 | 'image_id' | if col != 'image_id': |
| python | src/evaluation/merge_catalogs.py | 122 | '_noise' | new_row[col + '_noise'] = np.nan |
| python | src/evaluation/merge_catalogs.py | 124 | 'image_id' | if col != 'image_id': |
| python | src/evaluation/merge_catalogs.py | 125 | '_rec' | new_row[col + '_rec'] = np.nan |
| python | src/evaluation/merge_catalogs.py | 130 | 'image_id' | new_row = row_noise.rename(lambda x: x + '_noise' if x != 'image_id' else x).to_dict() |
| python | src/evaluation/merge_catalogs.py | 130 | '_noise' | new_row = row_noise.rename(lambda x: x + '_noise' if x != 'image_id' else x).to_dict() |
| python | src/evaluation/merge_catalogs.py | 132 | 'image_id' | if col != 'image_id': |
| python | src/evaluation/merge_catalogs.py | 133 | '_org' | new_row[col + '_org'] = np.nan |
| python | src/evaluation/merge_catalogs.py | 135 | 'image_id' | if col != 'image_id': |
| python | src/evaluation/merge_catalogs.py | 136 | '_rec' | new_row[col + '_rec'] = np.nan |
| python | src/evaluation/merge_catalogs.py | 142 | 3.0 | def process(noise_csv, org_csv, rec_csv, workers, output_parquet, threshold=3.0): |
| python | src/evaluation/merge_catalogs.py | 165 | 'image_id' | set(org_df['image_id'].unique()).union( |
| python | src/evaluation/merge_catalogs.py | 166 | 'image_id' | set(rec_df['image_id'].unique()), |
| python | src/evaluation/merge_catalogs.py | 167 | 'image_id' | set(noise_df['image_id'].unique()), |
| python | src/evaluation/merge_catalogs.py | 170 | 1 | workers = max(int(workers), 1) |
| python | src/evaluation/merge_catalogs.py | 178 | 'image_id' | chunk_org_df = org_df[org_df['image_id'].isin(ids)].reset_index(drop=True) |
| python | src/evaluation/merge_catalogs.py | 179 | 'image_id' | chunk_rec_df = rec_df[rec_df['image_id'].isin(ids)].reset_index(drop=True) |
| python | src/evaluation/merge_catalogs.py | 180 | 'image_id' | chunk_noise_df = noise_df[noise_df['image_id'].isin(ids)].reset_index(drop=True) |
| python | src/evaluation/merge_catalogs.py | 214 | 'uncropped_output_dir' | file_dir = eval_cfg['uncropped_output_dir'] |
| python | src/evaluation/merge_catalogs.py | 215 | 'uncropped_rec_catalog_csv' | rec_filename = eval_cfg['uncropped_rec_catalog_csv'] |
| python | src/evaluation/merge_catalogs.py | 216 | 'uncropped_noisy_catalog_csv' | noise_filename = eval_cfg['uncropped_noisy_catalog_csv'] |
| python | src/evaluation/merge_catalogs.py | 217 | 'uncropped_org_catalog_csv' | org_filename = eval_cfg['uncropped_org_catalog_csv'] |
| python | src/evaluation/merge_catalogs.py | 218 | 'photometrical_data_filename' | photometrical_data_filename = eval_cfg['photometrical_data_filename'] |
| python | src/evaluation/merge_catalogs.py | 225 | 'merge_catalog_workers' | workers = int(eval_cfg['merge_catalog_workers']) |
| python | src/evaluation/merge_catalogs.py | 226 | 'merge_catalog_threshold' | threshold = float(eval_cfg['merge_catalog_threshold']) |
| python | src/evaluation/merge_catalogs.py | 237 | '__main__' | if __name__ == '__main__': |
| python | src/evaluation/merge_catalogs.py | 241 | 'merge_catalogs' | _run_from_config(cfg['merge_catalogs']) |
| python | src/evaluation/metrics.py | 13 | '__main__' | if __name__ == '__main__' and 'CUDA_VISIBLE_DEVICES' not in os.environ: |
| python | src/evaluation/metrics.py | 13 | 'CUDA_VISIBLE_DEVICES' | if __name__ == '__main__' and 'CUDA_VISIBLE_DEVICES' not in os.environ: |
| python | src/evaluation/metrics.py | 14 | 'CUDA_VISIBLE_DEVICES' | os.environ['CUDA_VISIBLE_DEVICES'] = '-1' |
| python | src/evaluation/metrics.py | 39 | 64 | def generate_gaussian_weights(patch_size, sigma=64): |
| python | src/evaluation/metrics.py | 55 | 3 | if patch_size is None or len(patch_size) != 3: |
| python | src/evaluation/metrics.py | 56 | 'generate_gaussian_weights: patch_size is not a tuple of length 3 but %s' | logging.warning('generate_gaussian_weights: patch_size is not a tuple of length 3 but %s', patch_size) |
| python | src/evaluation/metrics.py | 58 | 2 | h, w = patch_size[:2]  # Extract height and width, ignore depth |
| python | src/evaluation/metrics.py | 59 | 2 | ax_h = np.linspace(-(h // 2), h // 2, h) |
| python | src/evaluation/metrics.py | 60 | 2 | ax_w = np.linspace(-(w // 2), w // 2, w) |
| python | src/evaluation/metrics.py | 62 | 2 | weights = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2)) |
| python | src/evaluation/metrics.py | 62 | 2.0 | weights = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2)) |
| python | src/evaluation/metrics.py | 82 | 3 | if patch_size is None or len(patch_size) != 3: |
| python | src/evaluation/metrics.py | 83 | 'generate_distance_weights: patch_size is not a tuple of length 3 but %s' | logging.warning('generate_distance_weights: patch_size is not a tuple of length 3 but %s', patch_size) |
| python | src/evaluation/metrics.py | 85 | 2 | h, w = patch_size[:2] |
| python | src/evaluation/metrics.py | 86 | 1 | ax_h = np.linspace(-1, 1, h) |
| python | src/evaluation/metrics.py | 87 | 1 | ax_w = np.linspace(-1, 1, w) |
| python | src/evaluation/metrics.py | 89 | 1.0 | weights = 1.0 - np.sqrt(xx**2 + yy**2) |
| python | src/evaluation/metrics.py | 89 | 2 | weights = 1.0 - np.sqrt(xx**2 + yy**2) |
| python | src/evaluation/metrics.py | 90 | 0 | return np.clip(weights, 0, None)[..., np.newaxis] |
| python | src/evaluation/metrics.py | 121 | 3 | if patch_size is None or len(patch_size) != 3: |
| python | src/evaluation/metrics.py | 122 | 'pad_image: patch_size must be a 3-element tuple, got %s' | logging.warning('pad_image: patch_size must be a 3-element tuple, got %s', patch_size) |
| python | src/evaluation/metrics.py | 126 | 'pad_image: image is None' | logging.warning('pad_image: image is None') |
| python | src/evaluation/metrics.py | 130 | 0 | pad_h = int(np.ceil(image_size[0]/patch_size[0])*patch_size[0] - image_size[0]) |
| python | src/evaluation/metrics.py | 131 | 1 | pad_w = int(np.ceil(image_size[1]/patch_size[1])*patch_size[1] - image_size[1]) |
| python | src/evaluation/metrics.py | 132 | 2 | pad_h_top = pad_h//2 |
| python | src/evaluation/metrics.py | 134 | 2 | pad_w_left = pad_w//2 |
| python | src/evaluation/metrics.py | 137 | 0 | pad_eq_h = patch_size[0] - stride[0] |
| python | src/evaluation/metrics.py | 138 | 1 | pad_eq_w = patch_size[1] - stride[1] |
| python | src/evaluation/metrics.py | 144 | 0 | constant_values=0 |
| python | src/evaluation/metrics.py | 175 | 'strip_pad: image is None' | logging.warning('strip_pad: image is None') |
| python | src/evaluation/metrics.py | 177 | 2 | if  (pad_eq_h * 2 + pad_h_top + pad_h_bottom) >= image.shape[0]: |
| python | src/evaluation/metrics.py | 177 | 0 | if  (pad_eq_h * 2 + pad_h_top + pad_h_bottom) >= image.shape[0]: |
| python | src/evaluation/metrics.py | 178 | 'strip_pad: image height %s smaller than total vertical padding' | logging.warning('strip_pad: image height %s smaller than total vertical padding', image.shape[0]) |
| python | src/evaluation/metrics.py | 178 | 0 | logging.warning('strip_pad: image height %s smaller than total vertical padding', image.shape[0]) |
| python | src/evaluation/metrics.py | 180 | 2 | if  (pad_eq_w * 2 + pad_w_left + pad_w_right) >= image.shape[1]: |
| python | src/evaluation/metrics.py | 180 | 1 | if  (pad_eq_w * 2 + pad_w_left + pad_w_right) >= image.shape[1]: |
| python | src/evaluation/metrics.py | 181 | 'strip_pad: image width %s smaller than total horizontal padding' | logging.warning('strip_pad: image width %s smaller than total horizontal padding', image.shape[1]) |
| python | src/evaluation/metrics.py | 181 | 1 | logging.warning('strip_pad: image width %s smaller than total horizontal padding', image.shape[1]) |
| python | src/evaluation/metrics.py | 184 | 0 | if pad_eq_h > 0: |
| python | src/evaluation/metrics.py | 186 | 0 | if pad_eq_w > 0: |
| python | src/evaluation/metrics.py | 188 | 0 | if pad_h_top > 0: |
| python | src/evaluation/metrics.py | 190 | 0 | if pad_h_bottom > 0: |
| python | src/evaluation/metrics.py | 192 | 0 | if pad_w_left > 0: |
| python | src/evaluation/metrics.py | 194 | 0 | if pad_w_right > 0: |
| python | src/evaluation/metrics.py | 198 | 256 | def sliding_window_generator(padded_image, patch_size=(256, 256, 1), stride=(128, 128, 1)): |
| python | src/evaluation/metrics.py | 198 | 128 | def sliding_window_generator(padded_image, patch_size=(256, 256, 1), stride=(128, 128, 1)): |
| python | src/evaluation/metrics.py | 220 | 0 | for i in range(0, H - patch_size[0] + 1, stride[0]): |
| python | src/evaluation/metrics.py | 220 | 1 | for i in range(0, H - patch_size[0] + 1, stride[0]): |
| python | src/evaluation/metrics.py | 221 | 0 | for j in range(0, W - patch_size[1] + 1, stride[1]): |
| python | src/evaluation/metrics.py | 221 | 1 | for j in range(0, W - patch_size[1] + 1, stride[1]): |
| python | src/evaluation/metrics.py | 222 | 0 | patch = padded_image[i:i + patch_size[0], j:j + patch_size[1], :] |
| python | src/evaluation/metrics.py | 222 | 1 | patch = padded_image[i:i + patch_size[0], j:j + patch_size[1], :] |
| python | src/evaluation/metrics.py | 225 | 256 | def create_prediction_dataset(image, patch_size=(256, 256, 1), stride=(128, 128, 1), batch_size=128): |
| python | src/evaluation/metrics.py | 225 | 128 | def create_prediction_dataset(image, patch_size=(256, 256, 1), stride=(128, 128, 1), batch_size=128): |
| python | src/evaluation/metrics.py | 250 | 0 | tf.TensorSpec(shape=(patch_size[0], patch_size[1], patch_size[2]), dtype=tf.float32), |
| python | src/evaluation/metrics.py | 250 | 1 | tf.TensorSpec(shape=(patch_size[0], patch_size[1], patch_size[2]), dtype=tf.float32), |
| python | src/evaluation/metrics.py | 250 | 2 | tf.TensorSpec(shape=(patch_size[0], patch_size[1], patch_size[2]), dtype=tf.float32), |
| python | src/evaluation/metrics.py | 251 | 2 | tf.TensorSpec(shape=(2,), dtype=tf.int64) |
| python | src/evaluation/metrics.py | 256 | 256 | def sliding_window_inference(image, model, patch_size=(256, 256, 1), |
| python | src/evaluation/metrics.py | 257 | 128 | stride=(128, 128, 1), weighting='average', batch_size=16, |
| python | src/evaluation/metrics.py | 257 | 'average' | stride=(128, 128, 1), weighting='average', batch_size=16, |
| python | src/evaluation/metrics.py | 257 | 16 | stride=(128, 128, 1), weighting='average', batch_size=16, |
| python | src/evaluation/metrics.py | 258 | 64 | gaussian_sigma=64): |
| python | src/evaluation/metrics.py | 288 | 'sliding_window_inference: patch_size %s exceeds image size %s' | logging.warning('sliding_window_inference: patch_size %s exceeds image size %s', patch_size, image_shape) |
| python | src/evaluation/metrics.py | 291 | 'sliding_window_inference: stride %s exceeds image size %s' | logging.warning('sliding_window_inference: stride %s exceeds image size %s', stride, image_shape) |
| python | src/evaluation/metrics.py | 297 | 'sliding_window_inference: pad_image returned None for image shape %s' | logging.warning('sliding_window_inference: pad_image returned None for image shape %s', image.shape) |
| python | src/evaluation/metrics.py | 301 | "could not unpack 'pad_image': " | logging.warning(f"could not unpack 'pad_image': {err}") |
| python | src/evaluation/metrics.py | 315 | 'average' | elif weighting == 'average': |
| python | src/evaluation/metrics.py | 320 | "sliding_window_inference: unknown weighting '%s', must be 'average', 'gaussian', or 'distance'" | logging.warning("sliding_window_inference: unknown weighting '%s', must be 'average', 'gaussian', or 'distance'", weighting) |
| python | src/evaluation/metrics.py | 328 | 0 | output[i:i + patch_size[0], j:j + patch_size[1], :] += prediction * weights |
| python | src/evaluation/metrics.py | 328 | 1 | output[i:i + patch_size[0], j:j + patch_size[1], :] += prediction * weights |
| python | src/evaluation/metrics.py | 329 | 0 | weight_matrix[i:i + patch_size[0], j:j + patch_size[1], :] += weights |
| python | src/evaluation/metrics.py | 329 | 1 | weight_matrix[i:i + patch_size[0], j:j + patch_size[1], :] += weights |
| python | src/evaluation/metrics.py | 334 | 'sliding_window_inference: strip_pad returned None' | logging.warning('sliding_window_inference: strip_pad returned None') |
| python | src/evaluation/metrics.py | 339 | 1e-12 | return output / (weight_matrix + 1e-12) |
| python | src/evaluation/metrics.py | 367 | 'scale_image: expected a NumPy array, got %s' | logging.warning('scale_image: expected a NumPy array, got %s', type(image)) |
| python | src/evaluation/metrics.py | 369 | 2 | if image.ndim < 2: |
| python | src/evaluation/metrics.py | 370 | 'scale_image: expected at least a 2D image, got shape %s' | logging.warning('scale_image: expected at least a 2D image, got shape %s', image.shape) |
| python | src/evaluation/metrics.py | 373 | 'scale_image: image array must contain numerical values, got dtype %s' | logging.warning('scale_image: image array must contain numerical values, got dtype %s', image.dtype) |
| python | src/evaluation/metrics.py | 381 | 3 | image_2d = image.squeeze() if image.ndim == 3 and image.shape[2] == 1 else image |
| python | src/evaluation/metrics.py | 381 | 2 | image_2d = image.squeeze() if image.ndim == 3 and image.shape[2] == 1 else image |
| python | src/evaluation/metrics.py | 381 | 1 | image_2d = image.squeeze() if image.ndim == 3 and image.shape[2] == 1 else image |
| python | src/evaluation/metrics.py | 382 | 255 | normalized_data = (norm(image_2d) * 255).astype(np.uint8) |
| python | src/evaluation/metrics.py | 426 | 'plot_source_comparison: expected a NumPy array, got %s' | logging.warning('plot_source_comparison: expected a NumPy array, got %s', type(image)) |
| python | src/evaluation/metrics.py | 428 | 2 | if image.ndim < 2: |
| python | src/evaluation/metrics.py | 429 | 'plot_source_comparison: expected at least a 2D image, got shape %s' | logging.warning('plot_source_comparison: expected at least a 2D image, got shape %s', image.shape) |
| python | src/evaluation/metrics.py | 432 | 'plot_source_comparison: image array must contain numerical values, got dtype %s' | logging.warning('plot_source_comparison: image array must contain numerical values, got dtype %s', image.dtype) |
| python | src/evaluation/metrics.py | 443 | 18 | plt.figure(figsize=(18, 6)) |
| python | src/evaluation/metrics.py | 443 | 6 | plt.figure(figsize=(18, 6)) |
| python | src/evaluation/metrics.py | 446 | 1 | plt.subplot(1, 3, 1) |
| python | src/evaluation/metrics.py | 446 | 3 | plt.subplot(1, 3, 1) |
| python | src/evaluation/metrics.py | 456 | 1 | plt.subplot(1, 3, 2) |
| python | src/evaluation/metrics.py | 456 | 3 | plt.subplot(1, 3, 2) |
| python | src/evaluation/metrics.py | 456 | 2 | plt.subplot(1, 3, 2) |
| python | src/evaluation/metrics.py | 461 | 1 | plt.subplot(1, 3, 3) |
| python | src/evaluation/metrics.py | 461 | 3 | plt.subplot(1, 3, 3) |
| python | src/evaluation/metrics.py | 471 | 300 | plt.savefig(filepath, dpi=300, bbox_inches='tight') |
| python | src/evaluation/metrics.py | 519 | 'plot_source_comparison_sep: expected a NumPy array, got %s' | logging.warning('plot_source_comparison_sep: expected a NumPy array, got %s', type(image)) |
| python | src/evaluation/metrics.py | 521 | 2 | if image.ndim < 2: |
| python | src/evaluation/metrics.py | 522 | 'plot_source_comparison_sep: expected at least a 2D image, got shape %s' | logging.warning('plot_source_comparison_sep: expected at least a 2D image, got shape %s', image.shape) |
| python | src/evaluation/metrics.py | 525 | 'plot_source_comparison_sep: image array must contain numerical values, got dtype %s' | logging.warning('plot_source_comparison_sep: image array must contain numerical values, got dtype %s', image.dtype) |
| python | src/evaluation/metrics.py | 536 | 18 | plt.figure(figsize=(18, 6)) |
| python | src/evaluation/metrics.py | 536 | 6 | plt.figure(figsize=(18, 6)) |
| python | src/evaluation/metrics.py | 539 | 1 | ax1 = plt.subplot(1, 3, 1) |
| python | src/evaluation/metrics.py | 539 | 3 | ax1 = plt.subplot(1, 3, 1) |
| python | src/evaluation/metrics.py | 545 | 6 | e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), |
| python | src/evaluation/metrics.py | 546 | 1.5 | edgecolor='blue', facecolor='none', linewidth=1.5) |
| python | src/evaluation/metrics.py | 552 | 6 | e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), |
| python | src/evaluation/metrics.py | 553 | 1.5 | edgecolor='red', facecolor='none', linewidth=1.5) |
| python | src/evaluation/metrics.py | 559 | 1 | ax2 = plt.subplot(1, 3, 2) |
| python | src/evaluation/metrics.py | 559 | 3 | ax2 = plt.subplot(1, 3, 2) |
| python | src/evaluation/metrics.py | 559 | 2 | ax2 = plt.subplot(1, 3, 2) |
| python | src/evaluation/metrics.py | 564 | 1 | ax3 = plt.subplot(1, 3, 3) |
| python | src/evaluation/metrics.py | 564 | 3 | ax3 = plt.subplot(1, 3, 3) |
| python | src/evaluation/metrics.py | 570 | 6 | e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), |
| python | src/evaluation/metrics.py | 571 | 1.5 | edgecolor='green', facecolor='none', linewidth=1.5) |
| python | src/evaluation/metrics.py | 577 | 6 | e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), |
| python | src/evaluation/metrics.py | 578 | 1.5 | edgecolor='red', facecolor='none', linewidth=1.5) |
| python | src/evaluation/metrics.py | 585 | 2 | Line2D([0], [0], color='red', lw=2, label='Matched Sources'), |
| python | src/evaluation/metrics.py | 586 | 2 | Line2D([0], [0], color='blue', lw=2, label='Unmatched Sources (Original)'), |
| python | src/evaluation/metrics.py | 587 | 2 | Line2D([0], [0], color='green', lw=2, label='Unmatched Sources (Reconstructed)') |
| python | src/evaluation/metrics.py | 590 | 0.5 | plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.2), |
| python | src/evaluation/metrics.py | 590 | 0.2 | plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.2), |
| python | src/evaluation/metrics.py | 591 | 3 | ncol=3, frameon=False) |
| python | src/evaluation/metrics.py | 594 | 300 | plt.savefig(filepath, dpi=300, bbox_inches='tight') |
| python | src/evaluation/metrics.py | 600 | 0.01 | def compute_ssim(x, y, alpha=1, beta=1, gamma=1, k1=0.01, k2=0.03, win_size=11, win_sigma=1.5): |
| python | src/evaluation/metrics.py | 600 | 0.03 | def compute_ssim(x, y, alpha=1, beta=1, gamma=1, k1=0.01, k2=0.03, win_size=11, win_sigma=1.5): |
| python | src/evaluation/metrics.py | 600 | 11 | def compute_ssim(x, y, alpha=1, beta=1, gamma=1, k1=0.01, k2=0.03, win_size=11, win_sigma=1.5): |
| python | src/evaluation/metrics.py | 600 | 1.5 | def compute_ssim(x, y, alpha=1, beta=1, gamma=1, k1=0.01, k2=0.03, win_size=11, win_sigma=1.5): |
| python | src/evaluation/metrics.py | 638 | 3 | squeezed = x.ndim == 3 and x.shape[2] == 1 |
| python | src/evaluation/metrics.py | 638 | 2 | squeezed = x.ndim == 3 and x.shape[2] == 1 |
| python | src/evaluation/metrics.py | 638 | 1 | squeezed = x.ndim == 3 and x.shape[2] == 1 |
| python | src/evaluation/metrics.py | 647 | 0 | if L == 0: |
| python | src/evaluation/metrics.py | 650 | 2 | c1 = (k1 * L) ** 2 |
| python | src/evaluation/metrics.py | 651 | 2 | c2 = (k2 * L) ** 2 |
| python | src/evaluation/metrics.py | 652 | 2 | c3 = c2 / 2 |
| python | src/evaluation/metrics.py | 656 | 2 | truncate = (win_size // 2) / win_sigma |
| python | src/evaluation/metrics.py | 661 | 2 | mu_x_sq = mu_x ** 2 |
| python | src/evaluation/metrics.py | 662 | 2 | mu_y_sq = mu_y ** 2 |
| python | src/evaluation/metrics.py | 665 | 2 | sigma_x_sq = np.maximum(filt(x2d ** 2) - mu_x_sq, 0) |
| python | src/evaluation/metrics.py | 665 | 0 | sigma_x_sq = np.maximum(filt(x2d ** 2) - mu_x_sq, 0) |
| python | src/evaluation/metrics.py | 666 | 2 | sigma_y_sq = np.maximum(filt(y2d ** 2) - mu_y_sq, 0) |
| python | src/evaluation/metrics.py | 666 | 0 | sigma_y_sq = np.maximum(filt(y2d ** 2) - mu_y_sq, 0) |
| python | src/evaluation/metrics.py | 673 | 2 | l = (2 * mu_x * mu_y + c1) / (mu_x_sq + mu_y_sq + c1) |
| python | src/evaluation/metrics.py | 674 | 2 | c = (2 * sigma_x * sigma_y + c2) / (sigma_x_sq + sigma_y_sq + c2) |
| python | src/evaluation/metrics.py | 706 | 'calculate_psnr: input images must be NumPy arrays with identical shapes' | logging.warning('calculate_psnr: input images must be NumPy arrays with identical shapes') |
| python | src/evaluation/metrics.py | 710 | 0 | if L == 0: |
| python | src/evaluation/metrics.py | 711 | 'calculate_psnr: original image has zero dynamic range, PSNR is undefined' | logging.warning('calculate_psnr: original image has zero dynamic range, PSNR is undefined') |
| python | src/evaluation/metrics.py | 714 | 2 | mse = np.mean((original - noisy) ** 2) |
| python | src/evaluation/metrics.py | 715 | 0 | if mse == 0: |
| python | src/evaluation/metrics.py | 717 | 10 | psnr = 10 * np.log10((L ** 2) / mse) |
| python | src/evaluation/metrics.py | 717 | 2 | psnr = 10 * np.log10((L ** 2) / mse) |
| python | src/evaluation/metrics.py | 752 | 'aggregate_df: expected a Pandas DataFrame, got %s' | logging.warning('aggregate_df: expected a Pandas DataFrame, got %s', type(df)) |
| python | src/evaluation/metrics.py | 755 | 0 | tp = df.get('TP', pd.Series(0)).sum() |
| python | src/evaluation/metrics.py | 756 | 0 | fp = df.get('FP', pd.Series(0)).sum() |
| python | src/evaluation/metrics.py | 757 | 0 | fn = df.get('FN', pd.Series(0)).sum() |
| python | src/evaluation/metrics.py | 760 | 'MSE_rec' | mean_mse_rec = df.get('MSE_rec', pd.Series(np.nan)).mean() |
| python | src/evaluation/metrics.py | 761 | 'MSE_noisy' | mean_mse_noisy = df.get('MSE_noisy', pd.Series(np.nan)).mean() |
| python | src/evaluation/metrics.py | 762 | 'PSNR_L' | L_rec = df.get('PSNR_L', pd.Series(np.nan)).mean()  # fallback: use stored L if available |
| python | src/evaluation/metrics.py | 765 | 'PSNR_rec' | psnr_rec = df.get('PSNR_rec', pd.Series(np.nan)).mean() |
| python | src/evaluation/metrics.py | 766 | 'PSNR_noisy' | psnr_noisy = df.get('PSNR_noisy', pd.Series(np.nan)).mean() |
| python | src/evaluation/metrics.py | 767 | 0 | if not np.isnan(mean_mse_rec) and mean_mse_rec > 0: |
| python | src/evaluation/metrics.py | 769 | 'PSNR_L' | L = df.get('PSNR_L', pd.Series(np.nan)).median() |
| python | src/evaluation/metrics.py | 770 | 0 | if not np.isnan(L) and L > 0: |
| python | src/evaluation/metrics.py | 771 | 10 | psnr_rec = 10 * np.log10((L ** 2) / mean_mse_rec) |
| python | src/evaluation/metrics.py | 771 | 2 | psnr_rec = 10 * np.log10((L ** 2) / mean_mse_rec) |
| python | src/evaluation/metrics.py | 772 | 0 | if not np.isnan(mean_mse_noisy) and mean_mse_noisy > 0: |
| python | src/evaluation/metrics.py | 773 | 'PSNR_L' | L = df.get('PSNR_L', pd.Series(np.nan)).median() |
| python | src/evaluation/metrics.py | 774 | 0 | if not np.isnan(L) and L > 0: |
| python | src/evaluation/metrics.py | 775 | 10 | psnr_noisy = 10 * np.log10((L ** 2) / mean_mse_noisy) |
| python | src/evaluation/metrics.py | 775 | 2 | psnr_noisy = 10 * np.log10((L ** 2) / mean_mse_noisy) |
| python | src/evaluation/metrics.py | 777 | 'SSIM_rec' | ssim_rec = df.get('SSIM_rec', pd.Series(np.nan)).mean() |
| python | src/evaluation/metrics.py | 778 | 'SSIM_noisy' | ssim_noisy = df.get('SSIM_noisy', pd.Series(np.nan)).mean() |
| python | src/evaluation/metrics.py | 780 | 0 | precision = tp / (tp + fp) if (tp + fp) > 0 else 0 |
| python | src/evaluation/metrics.py | 781 | 0 | recall = tp / (tp + fn) if (tp + fn) > 0 else 0 |
| python | src/evaluation/metrics.py | 782 | 0 | f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0 |
| python | src/evaluation/metrics.py | 782 | 2 | f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0 |
| python | src/evaluation/metrics.py | 785 | 0.0 | nonzero = flux_org != 0.0 |
| python | src/evaluation/metrics.py | 792 | 0 | snr_rec = 0 |
| python | src/evaluation/metrics.py | 796 | 0 | snr_org = 0 |
| python | src/evaluation/metrics.py | 806 | 'SNR_org' | 'SNR_org': snr_org, |
| python | src/evaluation/metrics.py | 807 | 'SNR_rec' | 'SNR_rec': snr_rec, |
| python | src/evaluation/metrics.py | 808 | 'PSNR_rec' | 'PSNR_rec': psnr_rec, |
| python | src/evaluation/metrics.py | 809 | 'PSNR_noisy' | 'PSNR_noisy': psnr_noisy, |
| python | src/evaluation/metrics.py | 810 | 'SSIM_rec' | 'SSIM_rec': ssim_rec, |
| python | src/evaluation/metrics.py | 811 | 'SSIM_noisy' | 'SSIM_noisy': ssim_noisy, |
| python | src/evaluation/metrics.py | 855 | 'footprint_radius' | 'footprint_radius', 'deblend', 'deblend_timeout'] |
| python | src/evaluation/metrics.py | 855 | 'deblend_timeout' | 'footprint_radius', 'deblend', 'deblend_timeout'] |
| python | src/evaluation/metrics.py | 858 | 'detect_sources_in_image: missing required parameters: %s' | logging.warning('detect_sources_in_image: missing required parameters: %s', ', '.join(missing_params)) |
| python | src/evaluation/metrics.py | 860 | 2 | if not isinstance(image, np.ndarray) or image.ndim != 2: |
| python | src/evaluation/metrics.py | 861 | 'detect_sources_in_image: image must be a 2D NumPy array, got shape %s' | logging.warning('detect_sources_in_image: image must be a 2D NumPy array, got shape %s', getattr(image, 'shape', type(image))) |
| python | src/evaluation/metrics.py | 868 | 'bkg_box_size' | bkg = Background2D(image, box_size=kwargs['bkg_box_size'], sigma_clip=sigma_clip,  # type: ignore[arg-type] |
| python | src/evaluation/metrics.py | 872 | 'detect_sources_in_image: error during background estimation: %s' | logging.warning('detect_sources_in_image: error during background estimation: %s', err) |
| python | src/evaluation/metrics.py | 877 | 'detect_sources_in_image: detect_sources returned None, no sources detected' | logging.warning('detect_sources_in_image: detect_sources returned None, no sources detected') |
| python | src/evaluation/metrics.py | 882 | 1 | with ThreadPoolExecutor(max_workers=1) as executor: |
| python | src/evaluation/metrics.py | 888 | 'deblend_timeout' | segm_deblended = future.result(timeout=kwargs['deblend_timeout']) |
| python | src/evaluation/metrics.py | 890 | 'detect_sources_in_image: deblending timed out, using undeblended segmentation' | logging.warning('detect_sources_in_image: deblending timed out, using undeblended segmentation') |
| python | src/evaluation/metrics.py | 893 | 'detect_sources_in_image: error during deblending: %s' | logging.warning('detect_sources_in_image: error during deblending: %s', err) |
| python | src/evaluation/metrics.py | 901 | 'make_source_mask' | if not hasattr(segm_deblended, 'make_source_mask'): |
| python | src/evaluation/metrics.py | 902 | 'Segmentation object does not expose make_source_mask; returning' | logging.warning('Segmentation object does not expose make_source_mask; returning') |
| python | src/evaluation/metrics.py | 905 | 'footprint_radius' | footprint = circular_footprint(radius=kwargs['footprint_radius']) |
| python | src/evaluation/metrics.py | 931 | 0 | if len(catalog) == 0: |
| python | src/evaluation/metrics.py | 932 | 'detect_sources_in_image: no sources found in catalog after deblending' | logging.warning('detect_sources_in_image: no sources found in catalog after deblending') |
| python | src/evaluation/metrics.py | 941 | 'detect_sources_in_image: error during source detection: %s' | logging.warning('detect_sources_in_image: error during source detection: %s', err) |
| python | src/evaluation/metrics.py | 978 | 'org_thresh' | thresh = kwargs['thresh'] if image_flag != 'original' else kwargs['org_thresh'] |
| python | src/evaluation/metrics.py | 979 | 'radius_factor' | radius_factor = kwargs['radius_factor']  # Factor for Kron radius |
| python | src/evaluation/metrics.py | 980 | 'PHOT_FLUXFRAC' | PHOT_FLUXFRAC  = kwargs['PHOT_FLUXFRAC']  # Fraction for flux radius |
| python | src/evaluation/metrics.py | 981 | 'r_min' | r_min = kwargs['r_min']  # Minimum radius for circular apertures |
| python | src/evaluation/metrics.py | 982 | 'elongation_fraction' | elongation_fraction = kwargs['elongation_fraction'] |
| python | src/evaluation/metrics.py | 983 | 'PHOT_AUTOPARAMS' | PHOT_AUTOPARAMS = kwargs['PHOT_AUTOPARAMS'] |
| python | src/evaluation/metrics.py | 986 | 'org_minarea' | minarea = kwargs['minarea'] if image_flag != 'original' else kwargs['org_minarea'] |
| python | src/evaluation/metrics.py | 987 | 'filter_type' | filter_type = kwargs['filter_type']  # Filter treatment type |
| python | src/evaluation/metrics.py | 988 | 'deblend_nthresh' | deblend_nthresh = kwargs['deblend_nthresh']  # Number of thresholds for deblending |
| python | src/evaluation/metrics.py | 989 | 'deblend_cont' | deblend_cont = kwargs['deblend_cont']  # Minimum contrast ratio for deblending |
| python | src/evaluation/metrics.py | 991 | 'clean_param' | clean_param = kwargs['clean_param']  # Cleaning parameter |
| python | src/evaluation/metrics.py | 1017 | 1 | PHOT_AUTOPARAMS*kronrad, subpix=1, err=rms_map) |
| python | src/evaluation/metrics.py | 1022 | 1 | cflux, cfluxerr, cflag = sep.sum_circle(data_sub, x[use_circle], y[use_circle], r_min, subpix=1, err=rms_map) |
| python | src/evaluation/metrics.py | 1028 | 5 | r, rflag = sep.flux_radius(data_sub, x, y, radius_factor * a, PHOT_FLUXFRAC, normflux=flux, subpix=5) |
| python | src/evaluation/metrics.py | 1033 | 3.0 | sep.mask_ellipse(mask, x, y, a, b, theta, r=3.) |
| python | src/evaluation/metrics.py | 1038 | 'kron_radius' | 'kron_radius': kronrad, |
| python | src/evaluation/metrics.py | 1040 | 'flux_err' | 'flux_err': fluxerr, |
| python | src/evaluation/metrics.py | 1041 | 'flux_radius' | 'flux_radius': r, |
| python | src/evaluation/metrics.py | 1042 | 'sep_flag' | 'sep_flag': phot_flag |
| python | src/evaluation/metrics.py | 1046 | 'is_galaxy' | merged_df['is_galaxy'] = (merged_df['a'] / merged_df['b'] >= elongation_fraction).astype(int) |
| python | src/evaluation/metrics.py | 1085 | 'flux_err' | return df['x'].to_numpy(), df['y'].to_numpy(), df['flux'].to_numpy(), df['flux_err'].to_numpy(), mask, df['a'].to_numpy(), df['b'].to_numpy(), df['theta'].to_numpy(), df |
| python | src/evaluation/metrics.py | 1116 | 0 | iou = intersection / union if union != 0 else np.nan |
| python | src/evaluation/metrics.py | 1169 | 'compare_images: all input images must be NumPy arrays: got types %s, %s, %s' | logging.warning('compare_images: all input images must be NumPy arrays: got types %s, %s, %s', type(image_org), type(noisy_image), type(image_reconstructed)) |
| python | src/evaluation/metrics.py | 1172 | 'compare_images: all input images must have the same shape: got shapes %s, %s, %s' | logging.warning('compare_images: all input images must have the same shape: got shapes %s, %s, %s', image_org.shape, noisy_image.shape, image_reconstructed.shape) |
| python | src/evaluation/metrics.py | 1178 | '_' | filepath = os.path.join(output_dir, f'{image_id}_{round(exp_time)}_{round(new_exp_time)}.png') |
| python | src/evaluation/metrics.py | 1178 | '.png' | filepath = os.path.join(output_dir, f'{image_id}_{round(exp_time)}_{round(new_exp_time)}.png') |
| python | src/evaluation/metrics.py | 1200 | 0 | orig_idx_rec = np.where(valid_rec)[0]          # maps clean→original index space |
| python | src/evaluation/metrics.py | 1204 | 0 | orig_idx_org = np.where(valid_org)[0]          # maps clean→original index space |
| python | src/evaluation/metrics.py | 1215 | 'image_id' | org_df['image_id'] = [image_id]*len(org_df) |
| python | src/evaluation/metrics.py | 1216 | 'image_id' | rec_df['image_id'] = [image_id]*len(rec_df) |
| python | src/evaluation/metrics.py | 1217 | 'image_id' | noisy_df['image_id'] = [image_id]*len(noisy_df) |
| python | src/evaluation/metrics.py | 1218 | 'exp_time' | org_df['exp_time'] = [exp_time]*len(org_df) |
| python | src/evaluation/metrics.py | 1219 | 'exp_time' | noisy_df['exp_time'] = [exp_time]*len(noisy_df) |
| python | src/evaluation/metrics.py | 1220 | 'new_exp_time' | noisy_df['new_exp_time'] = [new_exp_time]*len(noisy_df) |
| python | src/evaluation/metrics.py | 1221 | 'exp_time' | rec_df['exp_time'] = [exp_time]*len(rec_df) |
| python | src/evaluation/metrics.py | 1222 | 'new_exp_time' | rec_df['new_exp_time'] = [new_exp_time]*len(rec_df) |
| python | src/evaluation/metrics.py | 1229 | 'distance_threshold' | matches = tree_rec.query(np.column_stack((x_org_clean, y_org_clean)), distance_upper_bound=kwargs['distance_threshold']) |
| python | src/evaluation/metrics.py | 1231 | 0 | matched_clean_org = np.where(matches[0] < kwargs['distance_threshold'])[0] |
| python | src/evaluation/metrics.py | 1231 | 'distance_threshold' | matched_clean_org = np.where(matches[0] < kwargs['distance_threshold'])[0] |
| python | src/evaluation/metrics.py | 1232 | 1 | matched_clean_rec = matches[1][matched_clean_org] |
| python | src/evaluation/metrics.py | 1239 | 'compare_images: error finding mutual sources: %s' | logging.warning('compare_images: error finding mutual sources: %s', err) |
| python | src/evaluation/metrics.py | 1249 | 'win_size' | kwargs['k2'], kwargs['win_size'], kwargs['win_sigma']) |
| python | src/evaluation/metrics.py | 1249 | 'win_sigma' | kwargs['k2'], kwargs['win_size'], kwargs['win_sigma']) |
| python | src/evaluation/metrics.py | 1253 | 'win_size' | kwargs['k2'], kwargs['win_size'], kwargs['win_sigma']) |
| python | src/evaluation/metrics.py | 1253 | 'win_sigma' | kwargs['k2'], kwargs['win_size'], kwargs['win_sigma']) |
| python | src/evaluation/metrics.py | 1257 | 'compare_images: error calculating metrics: %s' | logging.warning('compare_images: error calculating metrics: %s', err) |
| python | src/evaluation/metrics.py | 1263 | 0 | precision = tp / (tp + fp) if (tp + fp) > 0 else 0 |
| python | src/evaluation/metrics.py | 1264 | 0 | recall = tp / (tp + fn) if (tp + fn) > 0 else 0 |
| python | src/evaluation/metrics.py | 1265 | 0 | f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0 |
| python | src/evaluation/metrics.py | 1265 | 2 | f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0 |
| python | src/evaluation/metrics.py | 1267 | 0.0 | rec_error_mask = flux_error_rec[matched_indices_image_rec] != 0.0 |
| python | src/evaluation/metrics.py | 1268 | 0.0 | org_error_mask = flux_error_org[matched_indices_image_org] != 0.0 |
| python | src/evaluation/metrics.py | 1280 | 0.0 | nonzero_flux = flux_org != 0.0 |
| python | src/evaluation/metrics.py | 1287 | 0 | snr_rec = 0 |
| python | src/evaluation/metrics.py | 1291 | 0 | snr_org = 0 |
| python | src/evaluation/metrics.py | 1295 | 'image_id' | 'image_id': image_id, |
| python | src/evaluation/metrics.py | 1296 | 'org_exp_time' | 'org_exp_time': exp_time, |
| python | src/evaluation/metrics.py | 1297 | 'new_exp_time' | 'new_exp_time': new_exp_time, |
| python | src/evaluation/metrics.py | 1305 | 'SNR_org' | 'SNR_org': snr_org, |
| python | src/evaluation/metrics.py | 1306 | 'SNR_rec' | 'SNR_rec': snr_rec, |
| python | src/evaluation/metrics.py | 1307 | 'PSNR_rec' | 'PSNR_rec': psnr, |
| python | src/evaluation/metrics.py | 1308 | 'PSNR_noisy' | 'PSNR_noisy': noisy_psnr, |
| python | src/evaluation/metrics.py | 1309 | 'MSE_rec' | 'MSE_rec': mse if isinstance(mse, float) else np.nan, |
| python | src/evaluation/metrics.py | 1310 | 'MSE_noisy' | 'MSE_noisy': noisy_mse if isinstance(noisy_mse, float) else np.nan, |
| python | src/evaluation/metrics.py | 1311 | 'PSNR_L' | 'PSNR_L': float(psnr_L), |
| python | src/evaluation/metrics.py | 1312 | 'SSIM_rec' | 'SSIM_rec': mean_ssmi, |
| python | src/evaluation/metrics.py | 1313 | 'SSIM_noisy' | 'SSIM_noisy': mean_noisy_ssmi, |
| python | src/evaluation/metrics.py | 1334 | 'compare_images: error saving source comparison plot: %s' | logging.warning('compare_images: error saving source comparison plot: %s', err) |
| python | src/evaluation/metrics.py | 1361 | 'log_min_max' | if scaling == 'log_min_max' or 'log_min_max' in model_dir: |
| python | src/evaluation/metrics.py | 1363 | 'min_max' | elif scaling == 'min_max' or 'min_max' in model_dir: |
| python | src/evaluation/metrics.py | 1365 | 'z_scale' | elif scaling == 'z_scale' or 'z_scale' in model_dir: |
| python | src/evaluation/metrics.py | 1408 | 'is_gan' | 'is_gan', |
| python | src/evaluation/metrics.py | 1409 | 'use_attention' | 'use_attention', |
| python | src/evaluation/metrics.py | 1410 | 'loss_function' | 'loss_function', |
| python | src/evaluation/metrics.py | 1411 | 'data_alias_enriched_hex' | 'data_alias_enriched_hex', |
| python | src/evaluation/metrics.py | 1412 | 'scaling_tag' | 'scaling_tag', |
| python | src/evaluation/metrics.py | 1413 | 'dropout_tag' | 'dropout_tag', |
| python | src/evaluation/metrics.py | 1414 | 'activation_tag' | 'activation_tag', |
| python | src/evaluation/metrics.py | 1415 | 'output_activation_tag' | 'output_activation_tag', |
| python | src/evaluation/metrics.py | 1416 | 'discriminator_activation_tag' | 'discriminator_activation_tag', |
| python | src/evaluation/metrics.py | 1417 | 'discriminator_output_activation_tag' | 'discriminator_output_activation_tag', |
| python | src/evaluation/metrics.py | 1422 | 0 | base = os.path.basename(model_file).split('.')[0] |
| python | src/evaluation/metrics.py | 1423 | 'final_model' | if base == 'final_model': |
| python | src/evaluation/metrics.py | 1425 | '_' | for part in reversed(base.split('_')): |
| python | src/evaluation/metrics.py | 1451 | 'model_prototype must be a non-empty string.' | raise ValueError('model_prototype must be a non-empty string.') |
| python | src/evaluation/metrics.py | 1482 | 'model_dir' | base_info['model_dir'] = rel_model_dir |
| python | src/evaluation/metrics.py | 1483 | 'checkpoints_dir' | base_info['checkpoints_dir'] = root |
| python | src/evaluation/metrics.py | 1490 | 'best_model' | row['prefix'] = 'best_model' if basename.startswith('best_model') else 'model' |
| python | src/evaluation/metrics.py | 1496 | '_prefix_order' | model_info['_prefix_order'] = (model_info['prefix'] != 'best_model').astype(int) |
| python | src/evaluation/metrics.py | 1496 | 'best_model' | model_info['_prefix_order'] = (model_info['prefix'] != 'best_model').astype(int) |
| python | src/evaluation/metrics.py | 1498 | '_prefix_order' | ['_prefix_order', 'epoch'], ascending=[True, False] |
| python | src/evaluation/metrics.py | 1499 | '_prefix_order' | ).drop(columns=['_prefix_order']).reset_index(drop=True) |
| python | src/evaluation/metrics.py | 1517 | 42 | random.seed(42)  # Set a fixed seed for reproducibility |
| python | src/evaluation/metrics.py | 1524 | 'find_best_performing_models: selected %d model directories' | logging.info('find_best_performing_models: selected %d model directories', len(discovered_models)) |
| python | src/evaluation/metrics.py | 1554 | 64 | def _reconstruct_patch(noisy_patch, scales, model, use_mosaic, patch_size, stride, weighting, batch_size, gaussian_sigma=64): |
| python | src/evaluation/metrics.py | 1591 | 0 | scaled_image = args[0] |
| python | src/evaluation/metrics.py | 1604 | '_reconstruct_patch: error during reconstruction: %s' | logging.warning('_reconstruct_patch: error during reconstruction: %s', err) |
| python | src/evaluation/metrics.py | 1607 | '_reconstruct_patch: error in scaling: %s' | logging.warning('_reconstruct_patch: error in scaling: %s', e) |
| python | src/evaluation/metrics.py | 1621 | '_reconstruct_patch: error during reconstruction: %s' | logging.warning('_reconstruct_patch: error during reconstruction: %s', err) |
| python | src/evaluation/metrics.py | 1669 | 0 | if model_index == 0: |
| python | src/evaluation/metrics.py | 1696 | 0 | if model_index == 0: |
| python | src/evaluation/metrics.py | 1719 | 256 | patch_size=(256, 256, 1), stride=(128, 128, 1), weighting='gaussian', batch_size=16, |
| python | src/evaluation/metrics.py | 1719 | 128 | patch_size=(256, 256, 1), stride=(128, 128, 1), weighting='gaussian', batch_size=16, |
| python | src/evaluation/metrics.py | 1719 | 16 | patch_size=(256, 256, 1), stride=(128, 128, 1), weighting='gaussian', batch_size=16, |
| python | src/evaluation/metrics.py | 1720 | 0.1 | frac=0.1, model_index=0, use_mosaic=True, |
| python | src/evaluation/metrics.py | 1721 | 64 | gaussian_sigma=64, |
| python | src/evaluation/metrics.py | 1724 | 'sci_actual_duration' | location_col='location', exp_time_col='sci_actual_duration', new_exp_time_col='new_exp_time', sigma_key='combined_sigma', |
| python | src/evaluation/metrics.py | 1724 | 'new_exp_time' | location_col='location', exp_time_col='sci_actual_duration', new_exp_time_col='new_exp_time', sigma_key='combined_sigma', |
| python | src/evaluation/metrics.py | 1724 | 'combined_sigma' | location_col='location', exp_time_col='sci_actual_duration', new_exp_time_col='new_exp_time', sigma_key='combined_sigma', |
| python | src/evaluation/metrics.py | 1726 | './metrics_updated/combined_images' | combined_images_dir='./metrics_updated/combined_images', png_dir='./metrics_updated/pngs', |
| python | src/evaluation/metrics.py | 1726 | './metrics_updated/pngs' | combined_images_dir='./metrics_updated/combined_images', png_dir='./metrics_updated/pngs', |
| python | src/evaluation/metrics.py | 1727 | './metrics_updated/original_images' | org_dir='./metrics_updated/original_images', noisy_dir='./metrics_updated/noisy_images', |
| python | src/evaluation/metrics.py | 1727 | './metrics_updated/noisy_images' | org_dir='./metrics_updated/original_images', noisy_dir='./metrics_updated/noisy_images', |
| python | src/evaluation/metrics.py | 1728 | './metrics_updated/reconstructed_images' | rec_dir='./metrics_updated/reconstructed_images', model_alias_hex=''): |
| python | src/evaluation/metrics.py | 1791 | 0 | epoch = os.path.basename(model_file).split('.')[0].split('_')[-1] |
| python | src/evaluation/metrics.py | 1791 | '_' | epoch = os.path.basename(model_file).split('.')[0].split('_')[-1] |
| python | src/evaluation/metrics.py | 1794 | 'process_single_model: DataFrame missing required columns: %s' | logging.warning('process_single_model: DataFrame missing required columns: %s', required_columns) |
| python | src/evaluation/metrics.py | 1803 | 'process_single_model: failed to load model %s: %s' | logging.error('process_single_model: failed to load model %s: %s', model_file, err) |
| python | src/evaluation/metrics.py | 1827 | 'process_single_model: error reading image %s: %s' | logging.warning('process_single_model: error reading image %s: %s', image_filepath, e) |
| python | src/evaluation/metrics.py | 1832 | 'process_single_model: open_fits returned unexpected type %s for %s' | logging.warning('process_single_model: open_fits returned unexpected type %s for %s', type(image), image_filepath) |
| python | src/evaluation/metrics.py | 1838 | 1 | org_name = os.path.basename(image_filepath).rsplit('.', 1)[0] |
| python | src/evaluation/metrics.py | 1838 | 0 | org_name = os.path.basename(image_filepath).rsplit('.', 1)[0] |
| python | src/evaluation/metrics.py | 1844 | '_' | neginf=neginf_value), f'{org_name}_{i}', i) |
| python | src/evaluation/metrics.py | 1845 | 0 | for i, crop in enumerate(crop_image_generator(image, ps=patch_size[0])) |
| python | src/evaluation/metrics.py | 1861 | 'process_single_model: reconstruction failed for %s, skipping' | logging.warning('process_single_model: reconstruction failed for %s, skipping', image_id) |
| python | src/evaluation/metrics.py | 1871 | 'process_single_model: error creating output directories: %s' | logging.warning('process_single_model: error creating output directories: %s', e) |
| python | src/evaluation/metrics.py | 1875 | '_org' | for index, (img, suffix) in enumerate([(base_image, "_org"), |
| python | src/evaluation/metrics.py | 1876 | '_rec' | (reconstructed_image, "_rec"), |
| python | src/evaluation/metrics.py | 1877 | '_noisy' | (noisy_image, "_noisy")]): |
| python | src/evaluation/metrics.py | 1879 | 0 | if index == 0: |
| python | src/evaluation/metrics.py | 1880 | '_' | filename = f"{org_name}{suffix}_{round(org_exp_time)}.fits" |
| python | src/evaluation/metrics.py | 1880 | '.fits' | filename = f"{org_name}{suffix}_{round(org_exp_time)}.fits" |
| python | src/evaluation/metrics.py | 1883 | 1 | elif index == 1: |
| python | src/evaluation/metrics.py | 1884 | '_' | filename = f"{org_name}{suffix}_{epoch}_{round(org_exp_time)}_{round(new_exp_time)}.fits" |
| python | src/evaluation/metrics.py | 1884 | '.fits' | filename = f"{org_name}{suffix}_{epoch}_{round(org_exp_time)}_{round(new_exp_time)}.fits" |
| python | src/evaluation/metrics.py | 1888 | '_' | filename = f"{org_name}{suffix}_{round(org_exp_time)}_{round(new_exp_time)}.fits" |
| python | src/evaluation/metrics.py | 1888 | '.fits' | filename = f"{org_name}{suffix}_{round(org_exp_time)}_{round(new_exp_time)}.fits" |
| python | src/evaluation/metrics.py | 1893 | 0 | scaled_img.save(os.path.join(png_dir, f"{filename.split('.')[0]}.png")) |
| python | src/evaluation/metrics.py | 1893 | '.png' | scaled_img.save(os.path.join(png_dir, f"{filename.split('.')[0]}.png")) |
| python | src/evaluation/metrics.py | 1895 | 'process_single_model: error scaling/saving PNG for %s%s: %s' | logging.warning('process_single_model: error scaling/saving PNG for %s%s: %s', org_name, suffix, e) |
| python | src/evaluation/metrics.py | 1897 | 'process_single_model: error saving images for %s%s: %s' | logging.warning('process_single_model: error saving images for %s%s: %s', org_name, suffix, e) |
| python | src/evaluation/metrics.py | 1923 | 'process_single_model: aggregate_df failed: %s' | logging.warning('process_single_model: aggregate_df failed: %s', err) |
| python | src/evaluation/metrics.py | 1928 | 8 | workers=8, frac=0.1, parallel=True, patch_size=(256, 256, 1), stride=(128, 128, 1), |
| python | src/evaluation/metrics.py | 1928 | 0.1 | workers=8, frac=0.1, parallel=True, patch_size=(256, 256, 1), stride=(128, 128, 1), |
| python | src/evaluation/metrics.py | 1928 | 256 | workers=8, frac=0.1, parallel=True, patch_size=(256, 256, 1), stride=(128, 128, 1), |
| python | src/evaluation/metrics.py | 1928 | 128 | workers=8, frac=0.1, parallel=True, patch_size=(256, 256, 1), stride=(128, 128, 1), |
| python | src/evaluation/metrics.py | 1929 | 16 | weighting='gaussian', batch_size=16, use_mosaic=True, |
| python | src/evaluation/metrics.py | 1930 | 64 | gaussian_sigma=64, |
| python | src/evaluation/metrics.py | 1933 | 'sci_actual_duration' | location_col='location', exp_time_col='sci_actual_duration', new_exp_time_col='new_exp_time', sigma_key='combined_sigma', |
| python | src/evaluation/metrics.py | 1933 | 'new_exp_time' | location_col='location', exp_time_col='sci_actual_duration', new_exp_time_col='new_exp_time', sigma_key='combined_sigma', |
| python | src/evaluation/metrics.py | 1933 | 'combined_sigma' | location_col='location', exp_time_col='sci_actual_duration', new_exp_time_col='new_exp_time', sigma_key='combined_sigma', |
| python | src/evaluation/metrics.py | 1935 | './metrics_updated/combined_images' | combined_images_dir='./metrics_updated/combined_images', png_dir='./metrics_updated/pngs', |
| python | src/evaluation/metrics.py | 1935 | './metrics_updated/pngs' | combined_images_dir='./metrics_updated/combined_images', png_dir='./metrics_updated/pngs', |
| python | src/evaluation/metrics.py | 1936 | './metrics_updated/original_images' | org_dir='./metrics_updated/original_images', noisy_dir='./metrics_updated/noisy_images', |
| python | src/evaluation/metrics.py | 1936 | './metrics_updated/noisy_images' | org_dir='./metrics_updated/original_images', noisy_dir='./metrics_updated/noisy_images', |
| python | src/evaluation/metrics.py | 1937 | './metrics_updated/reconstructed_images' | rec_dir='./metrics_updated/reconstructed_images', |
| python | src/evaluation/metrics.py | 1994 | 5 | if job is None or len(job) != 5: |
| python | src/evaluation/metrics.py | 1995 | 'process_models: invalid job tuple (expected 5-element tuple, got %s)' | logging.warning('process_models: invalid job tuple (expected 5-element tuple, got %s)', job) |
| python | src/evaluation/metrics.py | 2001 | 'process_models requires "' | raise ValueError(f'process_models requires "{name}" as a non-empty string template.') |
| python | src/evaluation/metrics.py | 2004 | 'all_metrics_csv' | all_metrics_csv_s = _require_str_path(all_metrics_csv, 'all_metrics_csv') |
| python | src/evaluation/metrics.py | 2005 | 'aggregated_metrics_csv' | aggregated_metrics_csv_s = _require_str_path(aggregated_metrics_csv, 'aggregated_metrics_csv') |
| python | src/evaluation/metrics.py | 2006 | 'org_catalog_csv' | org_catalog_csv_s = _require_str_path(org_catalog_csv, 'org_catalog_csv') |
| python | src/evaluation/metrics.py | 2007 | 'noisy_catalog_csv' | noisy_catalog_csv_s = _require_str_path(noisy_catalog_csv, 'noisy_catalog_csv') |
| python | src/evaluation/metrics.py | 2008 | 'rec_catalog_csv' | rec_catalog_csv_s = _require_str_path(rec_catalog_csv, 'rec_catalog_csv') |
| python | src/evaluation/metrics.py | 2010 | 'model_alias_hex' | all_metrics_csv = all_metrics_csv_s.replace('*', model_tags['model_alias_hex']) |
| python | src/evaluation/metrics.py | 2011 | 'model_alias_hex' | aggregated_metrics_csv = aggregated_metrics_csv_s.replace('*', model_tags['model_alias_hex']) |
| python | src/evaluation/metrics.py | 2012 | 'model_alias_hex' | org_catalog_csv = org_catalog_csv_s.replace('*', model_tags['model_alias_hex']) |
| python | src/evaluation/metrics.py | 2013 | 'model_alias_hex' | noisy_catalog_csv = noisy_catalog_csv_s.replace('*', model_tags['model_alias_hex']) |
| python | src/evaluation/metrics.py | 2014 | 'model_alias_hex' | rec_catalog_csv = rec_catalog_csv_s.replace('*', model_tags['model_alias_hex']) |
| python | src/evaluation/metrics.py | 2048 | 'model_alias_hex' | model_tags['model_alias_hex']) |
| python | src/evaluation/metrics.py | 2055 | 'process_models: error processing model future: %s' | logging.warning('process_models: error processing model future: %s', err) |
| python | src/evaluation/metrics.py | 2068 | 'model_alias_hex' | model_tags['model_alias_hex'])) |
| python | src/evaluation/metrics.py | 2107 | './metrics_updated/all_metrics_*.csv' | all_metrics_csv='./metrics_updated/all_metrics_*.csv', |
| python | src/evaluation/metrics.py | 2108 | './metrics_updated/aggregated_metrics_*.csv' | aggregated_metrics_csv='./metrics_updated/aggregated_metrics_*.csv', |
| python | src/evaluation/metrics.py | 2109 | './metrics_updated/org_catalog_*.csv' | org_catalog_csv='./metrics_updated/org_catalog_*.csv', |
| python | src/evaluation/metrics.py | 2110 | './metrics_updated/noisy_catalog_*.csv' | noisy_catalog_csv='./metrics_updated/noisy_catalog_*.csv', |
| python | src/evaluation/metrics.py | 2111 | './metrics_updated/rec_catalog_*.csv' | rec_catalog_csv='./metrics_updated/rec_catalog_*.csv', |
| python | src/evaluation/metrics.py | 2169 | 'kwargs_data' | test_images_df = get_test_images(None, data_kwargs["kwargs_data"], scaling=scaling) |
| python | src/evaluation/metrics.py | 2174 | 'get_test_images returned None, aborting.' | logging.warning("get_test_images returned None, aborting.") |
| python | src/evaluation/metrics.py | 2271 | 0 | if not isinstance(x, int) or x <= 0: |
| python | src/evaluation/metrics.py | 2278 | '_' | num_part = base.split('_')[-1].split('.')[0] |
| python | src/evaluation/metrics.py | 2278 | 0 | num_part = base.split('_')[-1].split('.')[0] |
| python | src/evaluation/metrics.py | 2284 | 1 | valid_files.sort(key=lambda item: item[1], reverse=True) |
| python | src/evaluation/metrics.py | 2287 | 0 | top_files = [f[0] for f in valid_files[:x]] |
| python | src/evaluation/metrics.py | 2291 | 'get_top_best_models: only %d valid files found, requested %d' | logging.warning('get_top_best_models: only %d valid files found, requested %d', len(top_files), x) |
| python | src/evaluation/metrics.py | 2294 | 75 | def get_model_by_modulo(file_list, prototype, modulo=75): |
| python | src/evaluation/metrics.py | 2323 | 0 | basename = os.path.basename(file).split('.')[0] |
| python | src/evaluation/metrics.py | 2324 | '_' | parts = basename.split('_') |
| python | src/evaluation/metrics.py | 2334 | 0 | if epoch % modulo == 0 and epoch <= 550: |
| python | src/evaluation/metrics.py | 2334 | 550 | if epoch % modulo == 0 and epoch <= 550: |
| python | src/evaluation/metrics.py | 2336 | 1 | elif index == len(my_dict) - 1: |
| python | src/evaluation/metrics.py | 2340 | '__main__' | if __name__ == '__main__': |
| python | src/evaluation/metrics.py | 2344 | 1 | index = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].lstrip('-').isdigit() else 0 |
| python | src/evaluation/metrics.py | 2345 | 2 | concurrent_workers = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].lstrip('-').isdigit() else 1 |
| python | src/evaluation/metrics.py | 2346 | 3 | overrides = parse_config_overrides(start_index=3)  # sys.argv[1]=index, sys.argv[2]=concurrent_workers, flags start at 3 |
| python | src/evaluation/metrics.py | 2350 | 'models_dir' | main(eval_cfg['models_dir'], |
| python | src/evaluation/metrics.py | 2351 | 'data_kwargs' | eval_cfg['data_kwargs'], eval_cfg['model_kwargs'], eval_cfg['kwargs_source'], |
| python | src/evaluation/metrics.py | 2351 | 'model_kwargs' | eval_cfg['data_kwargs'], eval_cfg['model_kwargs'], eval_cfg['kwargs_source'], |
| python | src/evaluation/metrics.py | 2351 | 'kwargs_source' | eval_cfg['data_kwargs'], eval_cfg['model_kwargs'], eval_cfg['kwargs_source'], |
| python | src/evaluation/metrics.py | 2352 | 'total_workers' | eval_cfg['total_workers'], eval_cfg['max_workers'], eval_cfg['frac'], |
| python | src/evaluation/metrics.py | 2352 | 'max_workers' | eval_cfg['total_workers'], eval_cfg['max_workers'], eval_cfg['frac'], |
| python | src/evaluation/metrics.py | 2353 | 'model_prototype' | condition, get_model_by_modulo, eval_cfg['n'], eval_cfg['model_prototype'], |
| python | src/evaluation/metrics.py | 2354 | 'all_metrics_csv' | all_metrics_csv=eval_cfg['all_metrics_csv'], |
| python | src/evaluation/metrics.py | 2355 | 'aggregated_metrics_csv' | aggregated_metrics_csv=eval_cfg['aggregated_metrics_csv'], |
| python | src/evaluation/metrics.py | 2356 | 'org_catalog_csv' | org_catalog_csv=eval_cfg['org_catalog_csv'], |
| python | src/evaluation/metrics.py | 2357 | 'noisy_catalog_csv' | noisy_catalog_csv=eval_cfg['noisy_catalog_csv'], |
| python | src/evaluation/metrics.py | 2358 | 'rec_catalog_csv' | rec_catalog_csv=eval_cfg['rec_catalog_csv'], |
| python | src/evaluation/metrics.py | 2359 | 'parallel' | parallel=eval_cfg['parallel'], parallel_epoch=eval_cfg['parallel_epoch'], |
| python | src/evaluation/metrics.py | 2359 | 'parallel_epoch' | parallel=eval_cfg['parallel'], parallel_epoch=eval_cfg['parallel_epoch'], |
| python | src/evaluation/uncropped_metrics.py | 48 | 'filter_by_last_name' | if kwargs_eval['filter_by_last_name']: |
| python | src/evaluation/uncropped_metrics.py | 49 | 'last_name_filter_value' | last_names = kwargs_eval['last_name_filter_value'] |
| python | src/evaluation/uncropped_metrics.py | 50 | 'last_name_col' | selected_df = df[df[kwargs_eval['last_name_col']].isin(last_names)] |
| python | src/evaluation/uncropped_metrics.py | 59 | 'new_exp_time' | filtered_data = data_df[data_df['new_exp_time'] >= low] |
| python | src/evaluation/uncropped_metrics.py | 90 | 1 | all_hists = {exp_ratio: [np.zeros(len(bins)-1), np.zeros(len(bins)-1), np.zeros(len(bins)-1)] |
| python | src/evaluation/uncropped_metrics.py | 91 | 'exp_ratio' | for exp_ratio in sub_df['exp_ratio'].unique()} |
| python | src/evaluation/uncropped_metrics.py | 100 | 'process_subdf: failed to load model %s: %s' | logging.error('process_subdf: failed to load model %s: %s', model_filepath, err) |
| python | src/evaluation/uncropped_metrics.py | 105 | 'uncropped_patch_size' | patch_size = tuple(kwargs['uncropped_patch_size']) |
| python | src/evaluation/uncropped_metrics.py | 106 | 'uncropped_stride' | stride = tuple(kwargs['uncropped_stride']) |
| python | src/evaluation/uncropped_metrics.py | 107 | 'uncropped_weighting' | weighting = kwargs['uncropped_weighting'] |
| python | src/evaluation/uncropped_metrics.py | 108 | 'uncropped_batch_size' | batch_size_inf = kwargs['uncropped_batch_size'] |
| python | src/evaluation/uncropped_metrics.py | 109 | 'nan_value' | nan_value = kwargs['nan_value'] |
| python | src/evaluation/uncropped_metrics.py | 110 | 'posinf_value' | posinf_value = kwargs['posinf_value'] |
| python | src/evaluation/uncropped_metrics.py | 111 | 'neginf_value' | neginf_value = kwargs['neginf_value'] |
| python | src/evaluation/uncropped_metrics.py | 119 | 'sigma_key' | noisy_sigma = row[kwargs['sigma_key']] |
| python | src/evaluation/uncropped_metrics.py | 120 | 'exp_time' | exp_time    = row['exp_time'] |
| python | src/evaluation/uncropped_metrics.py | 121 | 'new_exp_time' | new_exp_time = row['new_exp_time'] |
| python | src/evaluation/uncropped_metrics.py | 123 | 'exp_ratio' | exp_ratio  = row['exp_ratio'] |
| python | src/evaluation/uncropped_metrics.py | 124 | 'type_of_image' | image = open_fits(location, type_of_image=kwargs['type_of_image']) |
| python | src/evaluation/uncropped_metrics.py | 128 | 0 | i = 0 |
| python | src/evaluation/uncropped_metrics.py | 130 | 0 | for cropped_image in crop_image_generator(image, ps=patch_size[0]): |
| python | src/evaluation/uncropped_metrics.py | 132 | 'noise_fn' | noisy_image = kwargs['noise_fn'](cropped_image, row, noisy_sigma) |
| python | src/evaluation/uncropped_metrics.py | 137 | 'uncropped_use_mosaic' | use_mosaic=kwargs['uncropped_use_mosaic'], patch_size=patch_size, stride=stride, |
| python | src/evaluation/uncropped_metrics.py | 147 | 0 | all_hists[exp_ratio][0] += np.histogram(cropped_image.flatten(), bins=bins)[0] |
| python | src/evaluation/uncropped_metrics.py | 148 | 1 | all_hists[exp_ratio][1] += np.histogram(noisy_image.flatten(), bins=bins)[0] |
| python | src/evaluation/uncropped_metrics.py | 148 | 0 | all_hists[exp_ratio][1] += np.histogram(noisy_image.flatten(), bins=bins)[0] |
| python | src/evaluation/uncropped_metrics.py | 149 | 2 | all_hists[exp_ratio][2] += np.histogram(rec_image.flatten(), bins=bins)[0] |
| python | src/evaluation/uncropped_metrics.py | 149 | 0 | all_hists[exp_ratio][2] += np.histogram(rec_image.flatten(), bins=bins)[0] |
| python | src/evaluation/uncropped_metrics.py | 150 | '_' | result = compare_images(cropped_image, noisy_image, rec_image, f'{image_id}_{str(i)}', exp_time, new_exp_time, output_dir, kwargs, save_eval_images) |
| python | src/evaluation/uncropped_metrics.py | 153 | 'flux_rec' | stats['flux_rec'] = flux_rec |
| python | src/evaluation/uncropped_metrics.py | 154 | 'flux_org' | stats['flux_org'] = flux_org |
| python | src/evaluation/uncropped_metrics.py | 155 | 'flux_error_rec' | stats['flux_error_rec'] = flux_error_rec |
| python | src/evaluation/uncropped_metrics.py | 156 | 'flux_error_org' | stats['flux_error_org'] = flux_error_org |
| python | src/evaluation/uncropped_metrics.py | 160 | '_' | if f'{image_id}_{str(i)}' not in org_set: |
| python | src/evaluation/uncropped_metrics.py | 162 | '_' | org_set.add(f'{image_id}_{str(i)}') |
| python | src/evaluation/uncropped_metrics.py | 180 | 1 | for i in range(min_exp, max_exp+1): |
| python | src/evaluation/uncropped_metrics.py | 181 | 1 | for j in range(1,10): |
| python | src/evaluation/uncropped_metrics.py | 181 | 10 | for j in range(1,10): |
| python | src/evaluation/uncropped_metrics.py | 182 | 10.0 | data.append(j*10.**i) |
| python | src/evaluation/uncropped_metrics.py | 217 | 1 | chunk_size = max(1, len(gal_df) // workers) |
| python | src/evaluation/uncropped_metrics.py | 218 | 0 | sub_dfs = [gal_df.iloc[i:i + chunk_size] for i in range(0, len(gal_df), chunk_size)] |
| python | src/evaluation/uncropped_metrics.py | 238 | 'image_id' | temp = org_df[~org_df['image_id'].isin(org_set)] |
| python | src/evaluation/uncropped_metrics.py | 239 | 'image_id' | org_set.update(temp['image_id'].tolist()) |
| python | src/evaluation/uncropped_metrics.py | 247 | 'results_csv' | ensure_parent_dir_exists(output_paths['results_csv']) |
| python | src/evaluation/uncropped_metrics.py | 248 | 'results_csv' | results.to_csv(output_paths['results_csv'], index=False) |
| python | src/evaluation/uncropped_metrics.py | 254 | 'org_catalog_csv' | ensure_parent_dir_exists(output_paths['org_catalog_csv']) |
| python | src/evaluation/uncropped_metrics.py | 255 | 'noisy_catalog_csv' | ensure_parent_dir_exists(output_paths['noisy_catalog_csv']) |
| python | src/evaluation/uncropped_metrics.py | 256 | 'rec_catalog_csv' | ensure_parent_dir_exists(output_paths['rec_catalog_csv']) |
| python | src/evaluation/uncropped_metrics.py | 257 | 'org_catalog_csv' | org_dfs.to_csv(output_paths['org_catalog_csv'], index=False) |
| python | src/evaluation/uncropped_metrics.py | 258 | 'noisy_catalog_csv' | noisy_dfs.to_csv(output_paths['noisy_catalog_csv'], index=False) |
| python | src/evaluation/uncropped_metrics.py | 259 | 'rec_catalog_csv' | rec_dfs.to_csv(output_paths['rec_catalog_csv'], index=False) |
| python | src/evaluation/uncropped_metrics.py | 261 | 1 | final_hists = {exp_ratio: [np.zeros(len(bins) - 1), np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)] |
| python | src/evaluation/uncropped_metrics.py | 262 | 'exp_ratio' | for exp_ratio in gal_df['exp_ratio'].unique()} |
| python | src/evaluation/uncropped_metrics.py | 265 | 0 | final_hists[exp_ratio][0] += hists[0]  # Original |
| python | src/evaluation/uncropped_metrics.py | 266 | 1 | final_hists[exp_ratio][1] += hists[1]  # Noisy |
| python | src/evaluation/uncropped_metrics.py | 267 | 2 | final_hists[exp_ratio][2] += hists[2]  # Reconstructed |
| python | src/evaluation/uncropped_metrics.py | 277 | 8 | plt.figure(figsize=(8, 5)) |
| python | src/evaluation/uncropped_metrics.py | 277 | 5 | plt.figure(figsize=(8, 5)) |
| python | src/evaluation/uncropped_metrics.py | 284 | 0.5 | alpha=0.5, |
| python | src/evaluation/uncropped_metrics.py | 289 | 0 | lower_bound = bins[0] |
| python | src/evaluation/uncropped_metrics.py | 293 | 'lower_bound' | 'lower_bound': lower_bound, |
| python | src/evaluation/uncropped_metrics.py | 294 | 'upper_bound' | 'upper_bound': upper_bound, |
| python | src/evaluation/uncropped_metrics.py | 295 | 'exp_ratio' | 'exp_ratio': exp_ratio, |
| python | src/evaluation/uncropped_metrics.py | 305 | 'hist_png_template' | hist_png = output_paths['hist_png_template'].format(exp_ratio=int(exp_ratio), output_dir=output_dir) |
| python | src/evaluation/uncropped_metrics.py | 307 | 300 | plt.savefig(hist_png, dpi=300) |
| python | src/evaluation/uncropped_metrics.py | 311 | 'hist_data_csv' | ensure_parent_dir_exists(output_paths['hist_data_csv']) |
| python | src/evaluation/uncropped_metrics.py | 312 | 'hist_data_csv' | hist_data.to_csv(output_paths['hist_data_csv'], index=False) |
| python | src/evaluation/uncropped_metrics.py | 314 | '__main__' | if __name__ == '__main__': |
| python | src/evaluation/uncropped_metrics.py | 318 | 'uncropped_metrics' | eval_cfg = cfg['uncropped_metrics'] |
| python | src/evaluation/uncropped_metrics.py | 319 | 'data_kwargs' | data_cfg = dict(eval_cfg['data_kwargs']['kwargs_data']) |
| python | src/evaluation/uncropped_metrics.py | 319 | 'kwargs_data' | data_cfg = dict(eval_cfg['data_kwargs']['kwargs_data']) |
| python | src/evaluation/uncropped_metrics.py | 321 | 'metadata_filepath' | metadata_filepath = eval_cfg['metadata_filepath'] |
| python | src/evaluation/uncropped_metrics.py | 322 | 'uncropped_output_dir' | uncropped_output_dir = eval_cfg['uncropped_output_dir'] |
| python | src/evaluation/uncropped_metrics.py | 323 | 'uncropped_combined_images_dir' | uncropped_combined_images_dir = eval_cfg['uncropped_combined_images_dir'] |
| python | src/evaluation/uncropped_metrics.py | 325 | 'results_csv' | 'results_csv': str(Path(uncropped_output_dir) / eval_cfg['uncropped_results_csv']), |
| python | src/evaluation/uncropped_metrics.py | 325 | 'uncropped_results_csv' | 'results_csv': str(Path(uncropped_output_dir) / eval_cfg['uncropped_results_csv']), |
| python | src/evaluation/uncropped_metrics.py | 326 | 'org_catalog_csv' | 'org_catalog_csv': str(Path(uncropped_output_dir) / eval_cfg['uncropped_org_catalog_csv']), |
| python | src/evaluation/uncropped_metrics.py | 326 | 'uncropped_org_catalog_csv' | 'org_catalog_csv': str(Path(uncropped_output_dir) / eval_cfg['uncropped_org_catalog_csv']), |
| python | src/evaluation/uncropped_metrics.py | 327 | 'noisy_catalog_csv' | 'noisy_catalog_csv': str(Path(uncropped_output_dir) / eval_cfg['uncropped_noisy_catalog_csv']), |
| python | src/evaluation/uncropped_metrics.py | 327 | 'uncropped_noisy_catalog_csv' | 'noisy_catalog_csv': str(Path(uncropped_output_dir) / eval_cfg['uncropped_noisy_catalog_csv']), |
| python | src/evaluation/uncropped_metrics.py | 328 | 'rec_catalog_csv' | 'rec_catalog_csv': str(Path(uncropped_output_dir) / eval_cfg['uncropped_rec_catalog_csv']), |
| python | src/evaluation/uncropped_metrics.py | 328 | 'uncropped_rec_catalog_csv' | 'rec_catalog_csv': str(Path(uncropped_output_dir) / eval_cfg['uncropped_rec_catalog_csv']), |
| python | src/evaluation/uncropped_metrics.py | 329 | 'hist_data_csv' | 'hist_data_csv': str(Path(uncropped_output_dir) / eval_cfg['uncropped_hist_data_csv']), |
| python | src/evaluation/uncropped_metrics.py | 329 | 'uncropped_hist_data_csv' | 'hist_data_csv': str(Path(uncropped_output_dir) / eval_cfg['uncropped_hist_data_csv']), |
| python | src/evaluation/uncropped_metrics.py | 330 | 'hist_png_template' | 'hist_png_template': str(Path(uncropped_output_dir) / eval_cfg['uncropped_hist_png_template']), |
| python | src/evaluation/uncropped_metrics.py | 330 | 'uncropped_hist_png_template' | 'hist_png_template': str(Path(uncropped_output_dir) / eval_cfg['uncropped_hist_png_template']), |
| python | src/evaluation/uncropped_metrics.py | 333 | 'uncropped_n' | N               = eval_cfg['uncropped_n'] |
| python | src/evaluation/uncropped_metrics.py | 334 | 'uncropped_sampled_data_csv' | output_filepath = eval_cfg['uncropped_sampled_data_csv'] |
| python | src/evaluation/uncropped_metrics.py | 336 | 'models_dir' | eval_cfg['models_dir'], |
| python | src/evaluation/uncropped_metrics.py | 337 | 0 | condition = lambda df: np.zeros(len(df)) == 0, |
| python | src/evaluation/uncropped_metrics.py | 339 | 'model_prototype' | model_prototype=eval_cfg['model_prototype'], |
| python | src/evaluation/uncropped_metrics.py | 340 | 1 | n=1, |
| python | src/evaluation/uncropped_metrics.py | 341 | 0 | index=0, |
| python | src/evaluation/uncropped_metrics.py | 342 | 1 | concurrent_workers=1 |
| python | src/evaluation/uncropped_metrics.py | 353 | 'kwargs_source' | N, model_filepath, output_filepath, uncropped_combined_images_dir, eval_cfg['kwargs_source'], |
| python | src/evaluation/uncropped_metrics.py | 354 | 'uncropped_workers' | workers=eval_cfg['uncropped_workers'], |
| python | src/evaluation/uncropped_metrics.py | 355 | 'hist_min_exp' | min_exp=eval_cfg['hist_min_exp'], |
| python | src/evaluation/uncropped_metrics.py | 356 | 'hist_max_exp' | max_exp=eval_cfg['hist_max_exp'], |
| python | src/evaluation/uncropped_metrics.py | 358 | 'uncropped_save_images' | save_eval_images=eval_cfg['uncropped_save_images'] |
| python | src/training/callback.py | 30 | 'validation_loss.txt' | validation_loss_filename='validation_loss.txt', |
| python | src/training/callback.py | 31 | 'training_metrics.txt' | training_metrics_filename='training_metrics.txt', |
| python | src/training/callback.py | 105 | 0 | self.lr = 0 |
| python | src/training/callback.py | 117 | 'checkpoint_filename_pattern' | checkpoint_filename_pattern = self.config['training']['checkpoint_filename_pattern'] |
| python | src/training/callback.py | 119 | 'config["training"]["checkpoint_filename_pattern"] must be a non-empty string.' | raise ValueError('config["training"]["checkpoint_filename_pattern"] must be a non-empty string.') |
| python | src/training/callback.py | 121 | 'type_of_image' | self.type_of_image = self.config['data']['type_of_image'] |
| python | src/training/callback.py | 123 | 1 | (self.validation_dataset_size + self.batch_size - 1) // self.batch_size |
| python | src/training/callback.py | 124 | 0 | if self.batch_size > 0 else 0 |
| python | src/training/callback.py | 132 | 'log_min_max' | 'log_min_max': (inverse_adaptive_log_transform_and_denormalize, 3, 3), |
| python | src/training/callback.py | 132 | 3 | 'log_min_max': (inverse_adaptive_log_transform_and_denormalize, 3, 3), |
| python | src/training/callback.py | 133 | 'z_scale' | 'z_scale':     (inverse_zscore_normalization,                    2, 2), |
| python | src/training/callback.py | 133 | 2 | 'z_scale':     (inverse_zscore_normalization,                    2, 2), |
| python | src/training/callback.py | 134 | 'min_max' | 'min_max':     (inverse_min_max_normalization,                   2, 2), |
| python | src/training/callback.py | 134 | 2 | 'min_max':     (inverse_min_max_normalization,                   2, 2), |
| python | src/training/callback.py | 137 | 'Callback initialized: dataset_size=%s, start_epoch=%s, save_freq=%s, eval_save_percentage=%s, ds_save_percentage=%s, scaling=%s' | 'Callback initialized: dataset_size=%s, start_epoch=%s, save_freq=%s, eval_save_percentage=%s, ds_save_percentage=%s, scaling=%s', |
| python | src/training/callback.py | 168 | 'Training started: dataset_size=%s, planned_epochs=%s, start_epoch=%s, save_freq=%s, use_gan=%s.' | 'Training started: dataset_size=%s, planned_epochs=%s, start_epoch=%s, save_freq=%s, use_gan=%s.', |
| python | src/training/callback.py | 183 | 1 | logging.info('Epoch %d begin.', self.current_epoch + 1) |
| python | src/training/callback.py | 210 | 'd_loss' | train_loss = logs.get('d_loss', 0) if logs else 0 |
| python | src/training/callback.py | 210 | 0 | train_loss = logs.get('d_loss', 0) if logs else 0 |
| python | src/training/callback.py | 211 | 'g_loss' | g_loss = logs.get('g_loss', 0) if logs else 0 |
| python | src/training/callback.py | 211 | 0 | g_loss = logs.get('g_loss', 0) if logs else 0 |
| python | src/training/callback.py | 212 | 0 | train_loss = train_loss / self.batch_size if self.batch_size > 0 else train_loss |
| python | src/training/callback.py | 214 | 0 | train_loss = logs.get('loss', 0) if logs else 0 |
| python | src/training/callback.py | 219 | 1 | if self.current_epoch % max(int(self.save_freq), 1) == 0: |
| python | src/training/callback.py | 219 | 0 | if self.current_epoch % max(int(self.save_freq), 1) == 0: |
| python | src/training/callback.py | 220 | '/results_from_epochs/' | result_path = f"{os.path.dirname(self.checkpoints_dir)}/results_from_epochs/{self.current_epoch:04d}" |
| python | src/training/callback.py | 224 | 0.1 | target_count = int(self.dataset_size * max(float(self.ds_save_percentage), 0.1) // 100) |
| python | src/training/callback.py | 224 | 100 | target_count = int(self.dataset_size * max(float(self.ds_save_percentage), 0.1) // 100) |
| python | src/training/callback.py | 225 | 0 | temp = 0 |
| python | src/training/callback.py | 226 | 0 | if n_batches <= 0: |
| python | src/training/callback.py | 239 | 1 | for x_trains, y_trains, args in self.dataset.skip(index).take(1): |
| python | src/training/callback.py | 245 | 0 | filepath_str = decoded[0] if decoded else "sample.fits" |
| python | src/training/callback.py | 245 | 'sample.fits' | filepath_str = decoded[0] if decoded else "sample.fits" |
| python | src/training/callback.py | 263 | '.fits' | save_fits(prd_img,   filename.replace('.fits', '_output.fits'),  result_path, type_of_image=self.type_of_image) |
| python | src/training/callback.py | 263 | '_output.fits' | save_fits(prd_img,   filename.replace('.fits', '_output.fits'),  result_path, type_of_image=self.type_of_image) |
| python | src/training/callback.py | 264 | '.fits' | save_fits(noise_img, filename.replace('.fits', '_noise.fits'),   result_path, type_of_image=self.type_of_image) |
| python | src/training/callback.py | 264 | '_noise.fits' | save_fits(noise_img, filename.replace('.fits', '_noise.fits'),   result_path, type_of_image=self.type_of_image) |
| python | src/training/callback.py | 273 | 0 | if self.validation_total_batches <= 0: |
| python | src/training/callback.py | 275 | 0 | selected_validation_size = 0 |
| python | src/training/callback.py | 276 | 0 | n_batches_to_use = 0 |
| python | src/training/callback.py | 277 | 0 | validation_data = self.validation_dataset.take(0) |
| python | src/training/callback.py | 279 | 1.0 | eval_pct = max(float(self.eval_save_percentage), 1.0) |
| python | src/training/callback.py | 280 | 100 | selected_validation_size = int(self.validation_dataset_size * eval_pct // 100) |
| python | src/training/callback.py | 281 | 1 | selected_validation_size = min(max(selected_validation_size, 1), self.validation_dataset_size) |
| python | src/training/callback.py | 284 | 1 | max(1, (selected_validation_size + self.batch_size - 1) // self.batch_size), |
| python | src/training/callback.py | 294 | 0 | validation_loss = 0 |
| python | src/training/callback.py | 295 | 0 | num_batches = 0 |
| python | src/training/callback.py | 299 | 'reconstruction_loss_fn' | reconstruction_loss_fn = getattr(self.model, 'reconstruction_loss_fn', None) |
| python | src/training/callback.py | 304 | 'compiled_loss' | elif hasattr(self.model, 'compiled_loss') and self.model.compiled_loss is not None: |
| python | src/training/callback.py | 306 | 'loss_fn' | elif hasattr(self.model, 'loss_fn'): |
| python | src/training/callback.py | 312 | 0 | avg_validation_loss = validation_loss / (num_batches if num_batches > 0 else 1) |
| python | src/training/callback.py | 320 | 'best_model' | best_model_filename = build_checkpoint_filename('best_model', self.current_epoch, self.checkpoint_filename_pattern) |
| python | src/training/callback.py | 325 | 'model_type' | 'model_type': 'GAN' if self.use_gan else 'UNET', |
| python | src/training/callback.py | 325 | 'GAN' | 'model_type': 'GAN' if self.use_gan else 'UNET', |
| python | src/training/callback.py | 325 | 'UNET' | 'model_type': 'GAN' if self.use_gan else 'UNET', |
| python | src/training/callback.py | 343 | 1 | if self.current_epoch % max(int(self.save_freq), 1) == 0: |
| python | src/training/callback.py | 343 | 0 | if self.current_epoch % max(int(self.save_freq), 1) == 0: |
| python | src/training/callback.py | 352 | 'model_type' | 'model_type': 'GAN' if self.use_gan else 'UNET', |
| python | src/training/callback.py | 352 | 'GAN' | 'model_type': 'GAN' if self.use_gan else 'UNET', |
| python | src/training/callback.py | 352 | 'UNET' | 'model_type': 'GAN' if self.use_gan else 'UNET', |
| python | src/training/callback.py | 364 | 'train_loss' | 'train_loss': train_loss, |
| python | src/training/callback.py | 365 | 'g_loss' | 'g_loss' : g_loss, |
| python | src/training/callback.py | 366 | 'validation_loss' | 'validation_loss' : avg_validation_loss, |
| python | src/training/callback.py | 367 | 'epoch_time' | 'epoch_time': epoch_time, |
| python | src/training/callback.py | 383 | 'Epoch %d end: d_loss=%.6f g_loss=%.6f val_loss=%.6f epoch_time=%.2fs' | 'Epoch %d end: d_loss=%.6f g_loss=%.6f val_loss=%.6f epoch_time=%.2fs', |
| python | src/training/callback.py | 404 | '/' | final_model_path = f'{checkpoint_path}/{build_checkpoint_filename("final_model", self.current_epoch, self.checkpoint_filename_pattern)}' |
| python | src/training/callback.py | 404 | 'final_model' | final_model_path = f'{checkpoint_path}/{build_checkpoint_filename("final_model", self.current_epoch, self.checkpoint_filename_pattern)}' |
| python | src/training/callback.py | 409 | 'model_type' | 'model_type': 'GAN' if self.use_gan else 'UNET', |
| python | src/training/callback.py | 409 | 'GAN' | 'model_type': 'GAN' if self.use_gan else 'UNET', |
| python | src/training/callback.py | 409 | 'UNET' | 'model_type': 'GAN' if self.use_gan else 'UNET', |
| python | src/training/callback.py | 428 | 'Train_loss: ' | f"Train_loss: {metrics.get('train_loss', 0):.4f}" + "\t" + |
| python | src/training/callback.py | 428 | 'train_loss' | f"Train_loss: {metrics.get('train_loss', 0):.4f}" + "\t" + |
| python | src/training/callback.py | 428 | 0 | f"Train_loss: {metrics.get('train_loss', 0):.4f}" + "\t" + |
| python | src/training/callback.py | 429 | 'Val_loss: ' | f"Val_loss: {metrics.get('validation_loss', 0):.4f}" + "\t" + |
| python | src/training/callback.py | 429 | 'validation_loss' | f"Val_loss: {metrics.get('validation_loss', 0):.4f}" + "\t" + |
| python | src/training/callback.py | 429 | 0 | f"Val_loss: {metrics.get('validation_loss', 0):.4f}" + "\t" + |
| python | src/training/callback.py | 430 | 'epoch_time' | f"Time: {metrics.get('epoch_time', 0):.2f}\n") |
| python | src/training/callback.py | 430 | 0 | f"Time: {metrics.get('epoch_time', 0):.2f}\n") |
| python | src/training/math_helpers.py | 82 | 0 | if stop <= start or count <= 0: |
| python | src/training/math_helpers.py | 85 | 1 | lower_bound = start if start == 1 else start + 1 |
| python | src/training/math_helpers.py | 89 | 1 | sample_count = min(int(count), stop - lower_bound + 1) |
| python | src/training/math_helpers.py | 90 | 0 | if sample_count <= 0: |
| python | src/training/math_helpers.py | 116 | 0 | positive_values = finite_abs_values[finite_abs_values > 0] |
| python | src/training/math_helpers.py | 117 | 0 | if positive_values.size == 0: |
| python | src/training/math_helpers.py | 145 | 1 | for exponent_value in range(min_exponent, max_exponent + 1): |
| python | src/training/math_helpers.py | 146 | 1 | for base_value in range(1, 10): |
| python | src/training/math_helpers.py | 146 | 10 | for base_value in range(1, 10): |
| python | src/training/math_helpers.py | 147 | 10.0 | bin_start = base_value * 10.0 ** exponent_value |
| python | src/training/math_helpers.py | 148 | 1 | bin_end = (base_value + 1) * 10.0 ** exponent_value |
| python | src/training/math_helpers.py | 148 | 10.0 | bin_end = (base_value + 1) * 10.0 ** exponent_value |
| python | src/training/math_helpers.py | 183 | 1 | for exponent_value in range(min_exponent, max_exponent + 1): |
| python | src/training/math_helpers.py | 184 | 10.0 | bin_start = 10.0 ** exponent_value |
| python | src/training/math_helpers.py | 185 | 10.0 | bin_end = 10.0 ** (exponent_value + 1) |
| python | src/training/math_helpers.py | 185 | 1 | bin_end = 10.0 ** (exponent_value + 1) |
| python | src/training/math_helpers.py | 199 | 3 | def _apply_range(range_dataframe, row, use_base=True, n_sigma=3): |
| python | src/training/math_helpers.py | 219 | 'combined_sigma' | combined_sigma = abs(_row_get(row, 'combined_sigma')) |
| python | src/training/math_helpers.py | 220 | 'sigma_exponent' | matching_mask = range_dataframe['exponent'].eq(_row_get(row, 'sigma_exponent')) |
| python | src/training/math_helpers.py | 222 | 'sigma_base' | matching_mask &= range_dataframe['base'].eq(_row_get(row, 'sigma_base')) |
| python | src/training/math_helpers.py | 228 | 0 | mean_value, std_value = selected_rows.iloc[0] |
| python | src/training/math_helpers.py | 256 | 'exp_time' | return sigma_kernel_from_fit(row, fit_data, 't', 'a', 'dt', 'da', 'exp_time') |
| python | src/training/math_helpers.py | 286 | 0 | t = fit_data[t_key].values[0] |
| python | src/training/math_helpers.py | 287 | 0 | a = fit_data[a_key].values[0] |
| python | src/training/math_helpers.py | 288 | 0 | dt = np.random.uniform(-fit_data[dt_key].values[0], fit_data[dt_key].values[0]) |
| python | src/training/math_helpers.py | 289 | 0 | da = np.random.uniform(-fit_data[da_key].values[0], fit_data[da_key].values[0]) |
| python | src/training/math_helpers.py | 351 | 'initial_ratio must be numeric, got ' | raise ValueError(f'initial_ratio must be numeric, got {initial_ratio!r}') from err |
| python | src/training/math_helpers.py | 353 | 'ratio_count' | ratio_count_int = parse_required_int(ratio_count, 'ratio_count') |
| python | src/training/math_helpers.py | 358 | 'growth_factor must be numeric, got ' | raise ValueError(f'growth_factor must be numeric, got {growth_factor!r}') from err |
| python | src/training/math_helpers.py | 360 | 0 | if ratio_count_int <= 0: |
| python | src/training/math_helpers.py | 393 | 0.0 | if exposure_ratio <= 0.0 or original_exposure <= 0.0: |
| python | src/training/math_helpers.py | 453 | 'full_median_bkg' | return create_simulated_image_gaussian(file, _row_get(row, 'full_median_bkg'), new_sigma) |
| python | src/training/math_helpers.py | 476 | 'exp_time' | return create_simulated_image_poisson(file, _row_get(row, 'exp_time'), _row_get(row, 'exp_ratio')) |
| python | src/training/math_helpers.py | 476 | 'exp_ratio' | return create_simulated_image_poisson(file, _row_get(row, 'exp_time'), _row_get(row, 'exp_ratio')) |
| python | src/training/math_helpers.py | 499 | '.fits' | return os.path.basename(filename).split('.fits')[0] + f"_{suffix_value:.5f}".replace('.', '_') + ".fits" |
| python | src/training/math_helpers.py | 499 | 0 | return os.path.basename(filename).split('.fits')[0] + f"_{suffix_value:.5f}".replace('.', '_') + ".fits" |
| python | src/training/math_helpers.py | 499 | '_' | return os.path.basename(filename).split('.fits')[0] + f"_{suffix_value:.5f}".replace('.', '_') + ".fits" |
| python | src/training/math_helpers.py | 519 | '.fits' | return f"{os.path.basename(filename).split('.fits')[0]}_{round(_row_get(row, 'exp_time'))}_{round(_row_get(row, 'new_exp_time'))}" + ".fits" |
| python | src/training/math_helpers.py | 519 | 0 | return f"{os.path.basename(filename).split('.fits')[0]}_{round(_row_get(row, 'exp_time'))}_{round(_row_get(row, 'new_exp_time'))}" + ".fits" |
| python | src/training/math_helpers.py | 519 | '_' | return f"{os.path.basename(filename).split('.fits')[0]}_{round(_row_get(row, 'exp_time'))}_{round(_row_get(row, 'new_exp_time'))}" + ".fits" |
| python | src/training/math_helpers.py | 519 | 'exp_time' | return f"{os.path.basename(filename).split('.fits')[0]}_{round(_row_get(row, 'exp_time'))}_{round(_row_get(row, 'new_exp_time'))}" + ".fits" |
| python | src/training/math_helpers.py | 519 | 'new_exp_time' | return f"{os.path.basename(filename).split('.fits')[0]}_{round(_row_get(row, 'exp_time'))}_{round(_row_get(row, 'new_exp_time'))}" + ".fits" |
| python | src/training/math_helpers.py | 539 | '.fits' | return f"{os.path.basename(filename).split('.fits')[0]}_{round(_row_get(row, 'exp_ratio'), 2)}".replace('.', '-') + ".fits" |
| python | src/training/math_helpers.py | 539 | 0 | return f"{os.path.basename(filename).split('.fits')[0]}_{round(_row_get(row, 'exp_ratio'), 2)}".replace('.', '-') + ".fits" |
| python | src/training/math_helpers.py | 539 | '_' | return f"{os.path.basename(filename).split('.fits')[0]}_{round(_row_get(row, 'exp_ratio'), 2)}".replace('.', '-') + ".fits" |
| python | src/training/math_helpers.py | 539 | 'exp_ratio' | return f"{os.path.basename(filename).split('.fits')[0]}_{round(_row_get(row, 'exp_ratio'), 2)}".replace('.', '-') + ".fits" |
| python | src/training/math_helpers.py | 539 | 2 | return f"{os.path.basename(filename).split('.fits')[0]}_{round(_row_get(row, 'exp_ratio'), 2)}".replace('.', '-') + ".fits" |
| python | src/training/math_helpers.py | 561 | 1e-06 | if not np.isfinite(min_value) or not np.isfinite(max_value) or value_range <= 1e-6: |
| python | src/training/math_helpers.py | 586 | 1e-06 | if not np.isfinite(min_value) or not np.isfinite(max_value) or (max_value - min_value) <= 1e-6: |
| python | src/training/math_helpers.py | 607 | 1e-06 | if not np.isfinite(std_value) or std_value <= 1e-6: |
| python | src/training/math_helpers.py | 632 | 1e-06 | if not np.isfinite(mean_value) or not np.isfinite(std_value) or std_value <= 1e-6: |
| python | src/training/math_helpers.py | 656 | 0 | shift_value = -min_data_value + 1 if min_data_value < 0 else 1 |
| python | src/training/math_helpers.py | 656 | 1 | shift_value = -min_data_value + 1 if min_data_value < 0 else 1 |
| python | src/training/math_helpers.py | 663 | 1e-06 | if (max_log_value - min_log_value) <= 1e-6: |
| python | src/training/math_helpers.py | 703 | 80.0 | log_domain_data = np.clip(log_domain_data, a_min=None, a_max=80.0) |
| python | src/training/math_helpers.py | 740 | 'log_min_max' | 'log_min_max': adaptive_log_transform_and_normalize, |
| python | src/training/math_helpers.py | 741 | 'z_scale' | 'z_scale': zscore_normalization, |
| python | src/training/math_helpers.py | 742 | 'min_max' | 'min_max': min_max_normalization, |
| python | src/training/new_train.py | 11 | 'TF_ENABLE_ONEDNN_OPTS' | os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0') |
| python | src/training/new_train.py | 62 | '__name__' | getattr(optimizer_factory, '__name__', str(optimizer_factory)), |
| python | src/training/new_train.py | 72 | 0.5 | def poisson_noise_with_extra_components(img_data, ratio=0.5, exp_time=1, ron=3, dk=7, save=False, |
| python | src/training/new_train.py | 72 | 3 | def poisson_noise_with_extra_components(img_data, ratio=0.5, exp_time=1, ron=3, dk=7, save=False, |
| python | src/training/new_train.py | 72 | 7 | def poisson_noise_with_extra_components(img_data, ratio=0.5, exp_time=1, ron=3, dk=7, save=False, |
| python | src/training/new_train.py | 109 | 0 | if img_data is None or img_data.shape[0] < 2: |
| python | src/training/new_train.py | 109 | 2 | if img_data is None or img_data.shape[0] < 2: |
| python | src/training/new_train.py | 111 | 2 | height, width = img_data.shape[:2] |
| python | src/training/new_train.py | 113 | 0.0 | if simulated_time == 0.: |
| python | src/training/new_train.py | 118 | 0 | dark_current_noise = np.random.normal(0, np.sqrt(dk * simulated_time / (60 * 60)), (height, width)) |
| python | src/training/new_train.py | 118 | 60 | dark_current_noise = np.random.normal(0, np.sqrt(dk * simulated_time / (60 * 60)), (height, width)) |
| python | src/training/new_train.py | 119 | 0 | readout_noise = np.random.normal(0, ron, (height, width)) |
| python | src/training/new_train.py | 122 | 0.0 | noisy_img[img_data == 0.0] = 0.0 |
| python | src/training/new_train.py | 127 | 256 | def black_level(H, W, image, ps=256, steps=100): |
| python | src/training/new_train.py | 127 | 100 | def black_level(H, W, image, ps=256, steps=100): |
| python | src/training/new_train.py | 155 | 0 | if H < ps or W < ps or ps == 0: |
| python | src/training/new_train.py | 157 | 0 | if ps == 0: |
| python | src/training/new_train.py | 161 | 0 | xx = np.random.randint(0, H - ps, steps) |
| python | src/training/new_train.py | 162 | 0 | yy = np.random.randint(0, W - ps, steps) |
| python | src/training/new_train.py | 165 | 0 | best_idx = 0 |
| python | src/training/new_train.py | 168 | 0.0 | zero_counts = np.sum(patches == 0., axis=(1, 2)) |
| python | src/training/new_train.py | 168 | 2 | zero_counts = np.sum(patches == 0., axis=(1, 2)) |
| python | src/training/new_train.py | 246 | 'type_of_image' | result = open_fits(filepath, ratio_, type_of_image=kwargs_data['type_of_image'], |
| python | src/training/new_train.py | 253 | 0 | gt_patch = black_level(image.shape[0], image.shape[1], image, ps=kwargs_data['ps'], steps=kwargs_data['steps']) |
| python | src/training/new_train.py | 253 | 1 | gt_patch = black_level(image.shape[0], image.shape[1], image, ps=kwargs_data['ps'], steps=kwargs_data['steps']) |
| python | src/training/new_train.py | 255 | 'exp_time' | patch_pair_kwargs['exp_time'] = exp_time |
| python | src/training/new_train.py | 259 | ' due to error in prepare_patch_pair processing.' | logging.info(f"Skipping file {filepath} due to error in prepare_patch_pair processing.") |
| python | src/training/new_train.py | 284 | 8 | max_workers=8, |
| python | src/training/new_train.py | 330 | 'prepare_data: sampled_data has no "location" column; yielding nothing.' | logging.warning('prepare_data: sampled_data has no "location" column; yielding nothing.') |
| python | src/training/new_train.py | 334 | 'sigma_kernel_fn' | 'sigma_kernel_fn': sigma_kernel_fn, |
| python | src/training/new_train.py | 335 | 'noise_fn' | 'noise_fn': noise_fn, |
| python | src/training/new_train.py | 336 | 'stats_name_fn' | 'stats_name_fn': stats_name_fn, |
| python | src/training/new_train.py | 343 | 0 | if target_size <= 0: |
| python | src/training/new_train.py | 344 | 'prepare_data: sampled_data is empty; yielding nothing.' | logging.info('prepare_data: sampled_data is empty; yielding nothing.') |
| python | src/training/new_train.py | 347 | 1 | max_workers = max(1, int(max_workers)) |
| python | src/training/new_train.py | 364 | 'prepare_data: could not open file %s; skipping group.' | logging.debug('prepare_data: could not open file %s; skipping group.', filepath) |
| python | src/training/new_train.py | 376 | '_' | sigma_alias = f"{sigma_kernel:.5f}".replace('.', '_') |
| python | src/training/new_train.py | 388 | 'prepare_data: scaling returned None for %s; skipping.' | logging.debug('prepare_data: scaling returned None for %s; skipping.', filepath) |
| python | src/training/new_train.py | 394 | 0 | count = 0 |
| python | src/training/new_train.py | 397 | 'prepare_data: processing %d rows (training=%s, scaling=%s)' | 'prepare_data: processing %d rows (training=%s, scaling=%s)', |
| python | src/training/new_train.py | 404 | 100 | if count % 100 == 0: |
| python | src/training/new_train.py | 404 | 0 | if count % 100 == 0: |
| python | src/training/new_train.py | 405 | 'prepare_data: yielded %d / %d samples.' | logging.debug('prepare_data: yielded %d / %d samples.', count, target_size) |
| python | src/training/new_train.py | 413 | 'prepare_data: info_cached_df has no "location" column; using sampled_data for retries.' | logging.warning('prepare_data: info_cached_df has no "location" column; using sampled_data for retries.') |
| python | src/training/new_train.py | 420 | 0 | retry_pool = retry_pool[retry_pool['location'] != retry_candidate.iloc[0]['location']] |
| python | src/training/new_train.py | 428 | 100 | if count % 100 == 0: |
| python | src/training/new_train.py | 428 | 0 | if count % 100 == 0: |
| python | src/training/new_train.py | 429 | 'prepare_data: yielded %d / %d samples.' | logging.debug('prepare_data: yielded %d / %d samples.', count, target_size) |
| python | src/training/new_train.py | 432 | 0 | attempts = 0 |
| python | src/training/new_train.py | 433 | 10 | max_attempts = max(target_size * 10, 1) |
| python | src/training/new_train.py | 433 | 1 | max_attempts = max(target_size * 10, 1) |
| python | src/training/new_train.py | 443 | 100 | if count % 100 == 0: |
| python | src/training/new_train.py | 443 | 0 | if count % 100 == 0: |
| python | src/training/new_train.py | 444 | 'prepare_data: yielded %d / %d samples.' | logging.debug('prepare_data: yielded %d / %d samples.', count, target_size) |
| python | src/training/new_train.py | 448 | 'prepare_data: exhausted retry candidates after yielding %d / %d samples.' | 'prepare_data: exhausted retry candidates after yielding %d / %d samples.', |
| python | src/training/new_train.py | 480 | 'metadata_filepath' | metadata_filepath = kwargs_data['metadata_filepath'] |
| python | src/training/new_train.py | 481 | 'exposure_col' | exposure_col = kwargs_data['exposure_col'] |
| python | src/training/new_train.py | 488 | 'val_samples' | val_samples = parse_required_int(kwargs_data['val_samples'], 'data.val_samples') |
| python | src/training/new_train.py | 488 | 'data.val_samples' | val_samples = parse_required_int(kwargs_data['val_samples'], 'data.val_samples') |
| python | src/training/new_train.py | 490 | 'candidates_fn' | candidates_fn = kwargs_data['candidates_fn'] |
| python | src/training/new_train.py | 491 | 'post_filter_fn' | post_filter_fn = kwargs_data['post_filter_fn'] |
| python | src/training/new_train.py | 492 | 'sample_fn' | sample_fn = kwargs_data['sample_fn'] |
| python | src/training/new_train.py | 493 | 'sigma_kernel_fn' | sigma_kernel_fn = kwargs_data['sigma_kernel_fn'] |
| python | src/training/new_train.py | 494 | 'noise_fn' | noise_fn = kwargs_data['noise_fn'] |
| python | src/training/new_train.py | 495 | 'stats_name_fn' | stats_name_fn = kwargs_data['stats_name_fn'] |
| python | src/training/new_train.py | 497 | 'cache_raw_metadata' | cache_raw_metadata = kwargs_data['cache_raw_metadata'] |
| python | src/training/new_train.py | 498 | 'sub_sample_train' | sub_sample_train = kwargs_data['sub_sample_train'] |
| python | src/training/new_train.py | 499 | 'sub_sample_eval' | sub_sample_eval = kwargs_data['sub_sample_eval'] |
| python | src/training/new_train.py | 500 | 'nan_value' | preprocess_nan_value = kwargs_data['nan_value'] |
| python | src/training/new_train.py | 501 | 'posinf_value' | preprocess_posinf_value = kwargs_data['posinf_value'] |
| python | src/training/new_train.py | 502 | 'neginf_value' | preprocess_neginf_value = kwargs_data['neginf_value'] |
| python | src/training/new_train.py | 503 | 'sigma_key' | sigma_key = kwargs_data['sigma_key'] |
| python | src/training/new_train.py | 504 | 'max_workers' | max_workers = kwargs_data['max_workers'] |
| python | src/training/new_train.py | 507 | 'sigma_kernel_requires_fit_data' | if kwargs_data.get('sigma_kernel_requires_fit_data', False): |
| python | src/training/new_train.py | 508 | 'fit_data_filepath' | fit_data_filepath = kwargs_data['fit_data_filepath'] |
| python | src/training/new_train.py | 521 | 'data_augment_pluggable: using cached test metadata (%d rows).' | logging.info('data_augment_pluggable: using cached test metadata (%d rows).', len(MEM_CACHED_TEST)) |
| python | src/training/new_train.py | 524 | 'data_augment_pluggable: using cached training metadata (%d rows).' | logging.info('data_augment_pluggable: using cached training metadata (%d rows).', len(MEM_CACHED)) |
| python | src/training/new_train.py | 527 | 'data_augment_pluggable: using cached eval metadata (%d rows).' | logging.info('data_augment_pluggable: using cached eval metadata (%d rows).', len(MEM_CACHED_EVAL)) |
| python | src/training/new_train.py | 532 | 'data_augment_pluggable: loaded metadata CSV (%d rows) from %s' | logging.info('data_augment_pluggable: loaded metadata CSV (%d rows) from %s', len(df), metadata_filepath) |
| python | src/training/new_train.py | 534 | 'data_augment_pluggable: %d rows after exposure filter (times=%s, low=%s, high=%s)' | logging.info('data_augment_pluggable: %d rows after exposure filter (times=%s, low=%s, high=%s)', len(df), times, low, high) |
| python | src/training/new_train.py | 541 | 'test_samples' | x = min(parse_required_int(kwargs_data['test_samples'], 'data.test_samples'), len(info)) if test else min(samples if training else val_samples, len(info)) |
| python | src/training/new_train.py | 541 | 'data.test_samples' | x = min(parse_required_int(kwargs_data['test_samples'], 'data.test_samples'), len(info)) if test else min(samples if training else val_samples, len(info)) |
| python | src/training/new_train.py | 543 | 'data_augment_pluggable: %d candidates after post-filter; sampling %d (training=%s).' | logging.info('data_augment_pluggable: %d candidates after post-filter; sampling %d (training=%s).', len(info), x, training) |
| python | src/training/new_train.py | 560 | 'data_augment_pluggable: %d rows selected by sample_fn.' | logging.info('data_augment_pluggable: %d rows selected by sample_fn.', len(sampled_data)) |
| python | src/training/new_train.py | 564 | 'test_cache_filepath' | ensure_parent_dir_exists(kwargs_data['test_cache_filepath']) |
| python | src/training/new_train.py | 565 | 'test_cache_filepath' | sampled_data.reset_index(drop=True).to_csv(kwargs_data['test_cache_filepath'], index=False) |
| python | src/training/new_train.py | 566 | 'data_augment_pluggable: test metadata cached to %s.' | logging.info('data_augment_pluggable: test metadata cached to %s.', kwargs_data['test_cache_filepath']) |
| python | src/training/new_train.py | 566 | 'test_cache_filepath' | logging.info('data_augment_pluggable: test metadata cached to %s.', kwargs_data['test_cache_filepath']) |
| python | src/training/new_train.py | 569 | 'training_cache_filepath' | ensure_parent_dir_exists(kwargs_data['training_cache_filepath']) |
| python | src/training/new_train.py | 570 | 'training_cache_filepath' | sampled_data.reset_index(drop=True).to_csv(kwargs_data['training_cache_filepath'], index=False) |
| python | src/training/new_train.py | 571 | 'data_augment_pluggable: training metadata cached to %s.' | logging.info('data_augment_pluggable: training metadata cached to %s.', kwargs_data['training_cache_filepath']) |
| python | src/training/new_train.py | 571 | 'training_cache_filepath' | logging.info('data_augment_pluggable: training metadata cached to %s.', kwargs_data['training_cache_filepath']) |
| python | src/training/new_train.py | 574 | 'eval_cache_filepath' | ensure_parent_dir_exists(kwargs_data['eval_cache_filepath']) |
| python | src/training/new_train.py | 575 | 'eval_cache_filepath' | sampled_data.reset_index(drop=True).to_csv(kwargs_data['eval_cache_filepath'], index=False) |
| python | src/training/new_train.py | 576 | 'data_augment_pluggable: eval metadata cached to %s.' | logging.info('data_augment_pluggable: eval metadata cached to %s.', kwargs_data['eval_cache_filepath']) |
| python | src/training/new_train.py | 576 | 'eval_cache_filepath' | logging.info('data_augment_pluggable: eval metadata cached to %s.', kwargs_data['eval_cache_filepath']) |
| python | src/training/new_train.py | 578 | 'info_filepath' | ensure_parent_dir_exists(kwargs_data['info_filepath']) |
| python | src/training/new_train.py | 579 | 'info_filepath' | info.to_csv(kwargs_data['info_filepath'], index=False) |
| python | src/training/new_train.py | 585 | 'data_augment_pluggable: sub-sampled to %d rows (frac=%s).' | logging.info('data_augment_pluggable: sub-sampled to %d rows (frac=%s).', len(sampled_data), frac) |
| python | src/training/new_train.py | 593 | '_location_order' | .sort_values('_location_order', kind='stable') |
| python | src/training/new_train.py | 594 | '_location_order' | .drop(columns=['_location_order']) |
| python | src/training/new_train.py | 605 | 'type_of_image' | type_of_image=kwargs_data['type_of_image'], |
| python | src/training/new_train.py | 615 | 32 | def train_network(input_shape, n_epochs, kwargs_data, kwargs_network, data_generator, batch_size=32, |
| python | src/training/new_train.py | 616 | 0.0001 | optimizer=tf.keras.optimizers.Adam, change_learning_rate=[(0, 1e-4), (2000, 1e-5)], G_loss_fn: Any = tf.keras.losses.MeanAbsoluteError(), |
| python | src/training/new_train.py | 616 | 2000 | optimizer=tf.keras.optimizers.Adam, change_learning_rate=[(0, 1e-4), (2000, 1e-5)], G_loss_fn: Any = tf.keras.losses.MeanAbsoluteError(), |
| python | src/training/new_train.py | 616 | 1e-05 | optimizer=tf.keras.optimizers.Adam, change_learning_rate=[(0, 1e-4), (2000, 1e-5)], G_loss_fn: Any = tf.keras.losses.MeanAbsoluteError(), |
| python | src/training/new_train.py | 619 | 500 | save_freq=500, eval_save_percentage=20, ds_save_percentage=30, scaling=None, |
| python | src/training/new_train.py | 619 | 20 | save_freq=500, eval_save_percentage=20, ds_save_percentage=30, scaling=None, |
| python | src/training/new_train.py | 619 | 30 | save_freq=500, eval_save_percentage=20, ds_save_percentage=30, scaling=None, |
| python | src/training/new_train.py | 621 | './results' | training_results_dir='./results', training_metrics_csv_path='./training_history.csv', |
| python | src/training/new_train.py | 621 | './training_history.csv' | training_results_dir='./results', training_metrics_csv_path='./training_history.csv', |
| python | src/training/new_train.py | 622 | './training_history.json' | training_history_json_path='./training_history.json', |
| python | src/training/new_train.py | 623 | 'validation_loss.txt' | validation_loss_filename='validation_loss.txt', |
| python | src/training/new_train.py | 624 | 'training_metrics.txt' | training_metrics_filename='training_metrics.txt', |
| python | src/training/new_train.py | 677 | 'Starting train_network for %s epochs (scaling=%s, use_gan=%s).' | logging.info('Starting train_network for %s epochs (scaling=%s, use_gan=%s).', n_epochs, scaling, use_gan) |
| python | src/training/new_train.py | 681 | "The 'G_loss_fn' must be callable." | raise ValueError("The 'G_loss_fn' must be callable.") |
| python | src/training/new_train.py | 683 | 0 | if len(tf.config.list_physical_devices('GPU')) == 0 and batch_size > 2: |
| python | src/training/new_train.py | 683 | 2 | if len(tf.config.list_physical_devices('GPU')) == 0 and batch_size > 2: |
| python | src/training/new_train.py | 684 | 'No GPU detected; reducing batch_size from %d to %d for CPU stability.' | logging.warning('No GPU detected; reducing batch_size from %d to %d for CPU stability.', batch_size, 2) |
| python | src/training/new_train.py | 684 | 2 | logging.warning('No GPU detected; reducing batch_size from %d to %d for CPU stability.', batch_size, 2) |
| python | src/training/new_train.py | 685 | 2 | batch_size = 2 |
| python | src/training/new_train.py | 688 | 'training_path' | training_path = kwargs_data['training_path'] |
| python | src/training/new_train.py | 689 | 'eval_path' | eval_path = kwargs_data['eval_path'] |
| python | src/training/new_train.py | 690 | 'results_path' | results_path = kwargs_data['results_path'] |
| python | src/training/new_train.py | 693 | '.fits' | train_data = [os.path.join(training_path, f) for f in os.listdir(training_path) if f.endswith('.fits')] |
| python | src/training/new_train.py | 697 | '.fits' | eval_data = [os.path.join(eval_path, f) for f in os.listdir(eval_path) if f.endswith('.fits')] |
| python | src/training/new_train.py | 703 | 0 | start_epoch = 0 |
| python | src/training/new_train.py | 708 | 'learning_rate' | optimizer_kwargs['learning_rate'] = float(learning_rate) |
| python | src/training/new_train.py | 710 | 'beta_1' | optimizer_kwargs['beta_1'] = float(beta_1) |
| python | src/training/new_train.py | 714 | 'checkpoint_custom_epoch' | checkpoint_custom_epoch = kwargs_data['checkpoint_custom_epoch'] |
| python | src/training/new_train.py | 715 | 'checkpoint_restore_kwargs' | checkpoint_restore_kwargs = dict(kwargs_data['checkpoint_restore_kwargs']) |
| python | src/training/new_train.py | 716 | 'custom_objects' | checkpoint_restore_kwargs['custom_objects'] = build_checkpoint_custom_objects() |
| python | src/training/new_train.py | 733 | 0 | start_epoch = 0 |
| python | src/training/new_train.py | 749 | 'd_learning_rate' | if 'd_learning_rate' in gan_kwargs: |
| python | src/training/new_train.py | 750 | 'learning_rate' | discriminator_optimizer_kwargs['learning_rate'] = float(gan_kwargs['d_learning_rate']) |
| python | src/training/new_train.py | 750 | 'd_learning_rate' | discriminator_optimizer_kwargs['learning_rate'] = float(gan_kwargs['d_learning_rate']) |
| python | src/training/new_train.py | 751 | 'learning_rate' | elif 'learning_rate' in generator_optimizer_kwargs: |
| python | src/training/new_train.py | 752 | 'learning_rate' | discriminator_optimizer_kwargs['learning_rate'] = generator_optimizer_kwargs['learning_rate'] |
| python | src/training/new_train.py | 753 | 'beta_1' | if 'beta_1' in generator_optimizer_kwargs: |
| python | src/training/new_train.py | 754 | 'beta_1' | discriminator_optimizer_kwargs['beta_1'] = generator_optimizer_kwargs['beta_1'] |
| python | src/training/new_train.py | 760 | 'loss_fn' | adversarial_loss_fn=gan_kwargs['loss_fn'], |
| python | src/training/new_train.py | 762 | 'adversarial_loss_weight' | adversarial_loss_weight=gan_kwargs['adversarial_loss_weight'], |
| python | src/training/new_train.py | 763 | 'reconstruction_loss_weight' | reconstruction_loss_weight=gan_kwargs['reconstruction_loss_weight'], |
| python | src/training/new_train.py | 764 | 'label_smoothing' | label_smoothing=gan_kwargs['label_smoothing'], |
| python | src/training/new_train.py | 785 | 'Training dataset created with batch_size=%d.' | logging.info('Training dataset created with batch_size=%d.', batch_size) |
| python | src/training/new_train.py | 791 | 'Validation dataset created with batch_size=%d.' | logging.info('Validation dataset created with batch_size=%d.', batch_size) |
| python | src/training/new_train.py | 793 | 0 | train_dataset_size = 0 |
| python | src/training/new_train.py | 794 | 0 | train_batches = 0 |
| python | src/training/new_train.py | 797 | 0 | train_dataset_size += int(x_batch.shape[0]) |
| python | src/training/new_train.py | 799 | 0 | validation_dataset_size = 0 |
| python | src/training/new_train.py | 800 | 0 | valid_batches = 0 |
| python | src/training/new_train.py | 803 | 0 | validation_dataset_size += int(x_batch.shape[0]) |
| python | src/training/new_train.py | 805 | 0 | if train_batches <= 0: |
| python | src/training/new_train.py | 806 | 'Training dataset produced zero batches. Check data filtering/sampling settings.' | raise ValueError('Training dataset produced zero batches. Check data filtering/sampling settings.') |
| python | src/training/new_train.py | 807 | 0 | if valid_batches <= 0: |
| python | src/training/new_train.py | 808 | 'Validation dataset produced zero batches. Check data filtering/sampling settings.' | raise ValueError('Validation dataset produced zero batches. Check data filtering/sampling settings.') |
| python | src/training/new_train.py | 836 | 'GAN' | logging.info('Starting %s.fit for %d epochs (effective=%d).', 'GAN' if use_gan else 'network', n_epochs, n_epochs - start_epoch) |
| python | src/training/new_train.py | 860 | 'new_train' | train_cfg = config_values['new_train'] |
| python | src/training/new_train.py | 862 | 'data_kwargs' | data_config = dict(training_config['data_kwargs']) |
| python | src/training/new_train.py | 863 | 'network_kwargs' | network_config = dict(training_config['network_kwargs']) |
| python | src/training/new_train.py | 864 | 'discriminator_kwargs' | discriminator_config = dict(training_config['discriminator_kwargs']) |
| python | src/training/new_train.py | 865 | 'gan_kwargs' | gan_config = dict(training_config['gan_kwargs']) |
| python | src/training/new_train.py | 867 | 'training_path' | logging.info('Resolved paths: training=%s eval=%s models=%s', data_config['training_path'], data_config['eval_path'], data_config['results_path']) |
| python | src/training/new_train.py | 867 | 'eval_path' | logging.info('Resolved paths: training=%s eval=%s models=%s', data_config['training_path'], data_config['eval_path'], data_config['results_path']) |
| python | src/training/new_train.py | 867 | 'results_path' | logging.info('Resolved paths: training=%s eval=%s models=%s', data_config['training_path'], data_config['eval_path'], data_config['results_path']) |
| python | src/training/new_train.py | 871 | 'patch_size' | tuple(training_config['patch_size']), |
| python | src/training/new_train.py | 872 | 'n_epochs' | training_config['n_epochs'], |
| python | src/training/new_train.py | 875 | 'data_generator' | data_generator=training_config['data_generator'], |
| python | src/training/new_train.py | 876 | 'batch_size' | batch_size=training_config['batch_size'], |
| python | src/training/new_train.py | 878 | 'change_learning_rate' | change_learning_rate=training_config['change_learning_rate'], |
| python | src/training/new_train.py | 879 | 'g_loss_fn' | G_loss_fn=training_config['g_loss_fn'], |
| python | src/training/new_train.py | 880 | 'learning_rate' | learning_rate=training_config['learning_rate'], |
| python | src/training/new_train.py | 881 | 'beta_1' | beta_1=training_config['beta_1'], |
| python | src/training/new_train.py | 882 | 'start_from_best' | start_from_best=training_config['start_from_best'], |
| python | src/training/new_train.py | 883 | 'start_from_last' | start_from_last=training_config['start_from_last'], |
| python | src/training/new_train.py | 884 | 'save_freq' | save_freq=training_config['save_freq'], |
| python | src/training/new_train.py | 885 | 'eval_save_percentage' | eval_save_percentage=training_config['eval_save_percentage'], |
| python | src/training/new_train.py | 886 | 'ds_save_percentage' | ds_save_percentage=training_config['ds_save_percentage'], |
| python | src/training/new_train.py | 890 | 'use_gan' | use_gan=training_config['use_gan'], |
| python | src/training/new_train.py | 891 | 'training_results_dir' | training_results_dir=training_config['training_results_dir'], |
| python | src/training/new_train.py | 892 | 'training_metrics_csv_path' | training_metrics_csv_path=training_config['training_metrics_csv_path'], |
| python | src/training/new_train.py | 893 | 'training_history_json_path' | training_history_json_path=training_config['training_history_json_path'], |
| python | src/training/new_train.py | 894 | 'validation_loss_filename' | validation_loss_filename=training_config['validation_loss_filename'], |
| python | src/training/new_train.py | 895 | 'training_metrics_filename' | training_metrics_filename=training_config['training_metrics_filename'], |
| python | src/training/new_train.py | 899 | '__main__' | if __name__ == "__main__": |
| python | src/training/utils.py | 40 | '^([A-Za-z]):[\\\\/](.*)$' | windows_abs = re.match(r'^([A-Za-z]):[\\/](.*)$', filepath) |
| python | src/training/utils.py | 42 | 1 | drive = windows_abs.group(1).lower() |
| python | src/training/utils.py | 43 | 2 | rest = windows_abs.group(2).replace('\\', '/').lstrip('/') |
| python | src/training/utils.py | 43 | '\\' | rest = windows_abs.group(2).replace('\\', '/').lstrip('/') |
| python | src/training/utils.py | 43 | '/' | rest = windows_abs.group(2).replace('\\', '/').lstrip('/') |
| python | src/training/utils.py | 44 | '/mnt/' | return f'/mnt/{drive}/{rest}' |
| python | src/training/utils.py | 44 | '/' | return f'/mnt/{drive}/{rest}' |
| python | src/training/utils.py | 46 | '\\' | if '\\' in filepath: |
| python | src/training/utils.py | 47 | '\\' | return filepath.replace('\\', '/') |
| python | src/training/utils.py | 47 | '/' | return filepath.replace('\\', '/') |
| python | src/training/utils.py | 229 | 'checkpoint_info.json' | CHECKPOINT_INFO_FILENAME = 'checkpoint_info.json' |
| python | src/training/utils.py | 249 | 'filename_pattern must be a string.' | raise TypeError('filename_pattern must be a string.') |
| python | src/training/utils.py | 250 | '\\{epoch[^}]*\\}' | if '{prefix}' not in filename_pattern or re.search(r'\{epoch[^}]*\}', filename_pattern) is None: |
| python | src/training/utils.py | 251 | "filename_pattern must include '{prefix}' and '{epoch...}'." | raise ValueError("filename_pattern must include '{prefix}' and '{epoch...}'.") |
| python | src/training/utils.py | 262 | 2 | json.dumps(checkpoint_info, indent=2, default=str), |
| python | src/training/utils.py | 318 | 'loader_kwargs must be a dict.' | raise TypeError('loader_kwargs must be a dict.') |
| python | src/training/utils.py | 375 | 'filename_pattern must be a string.' | raise TypeError('filename_pattern must be a string.') |
| python | src/training/utils.py | 377 | '__PREFIX__' | normalised = filename_pattern.replace('{prefix}', '__PREFIX__') |
| python | src/training/utils.py | 378 | '\\{epoch[^}]*\\}' | normalised = re.sub(r'\{epoch[^}]*\}', '__EPOCH__', normalised) |
| python | src/training/utils.py | 378 | '__EPOCH__' | normalised = re.sub(r'\{epoch[^}]*\}', '__EPOCH__', normalised) |
| python | src/training/utils.py | 380 | '__PREFIX__' | if '__PREFIX__' not in normalised or '__EPOCH__' not in normalised: |
| python | src/training/utils.py | 380 | '__EPOCH__' | if '__PREFIX__' not in normalised or '__EPOCH__' not in normalised: |
| python | src/training/utils.py | 381 | "filename_pattern must include '{prefix}' and '{epoch...}'." | raise ValueError("filename_pattern must include '{prefix}' and '{epoch...}'.") |
| python | src/training/utils.py | 383 | '__PREFIX__' | regex_pattern = '^' + re.escape(normalised).replace('__PREFIX__', re.escape(checkpoint_prefix)).replace('__EPOCH__', r'(\d+)') + '$' |
| python | src/training/utils.py | 383 | '__EPOCH__' | regex_pattern = '^' + re.escape(normalised).replace('__PREFIX__', re.escape(checkpoint_prefix)).replace('__EPOCH__', r'(\d+)') + '$' |
| python | src/training/utils.py | 383 | '(\\d+)' | regex_pattern = '^' + re.escape(normalised).replace('__PREFIX__', re.escape(checkpoint_prefix)).replace('__EPOCH__', r'(\d+)') + '$' |
| python | src/training/utils.py | 390 | 1 | discovered_epochs.append(int(match.group(1))) |
| python | src/training/utils.py | 393 | 0 | start_epoch = 0 |
| python | src/training/utils.py | 395 | 'restore_kwargs must be a dict.' | raise TypeError('restore_kwargs must be a dict.') |
| python | src/training/utils.py | 396 | 'filename_pattern' | if 'filename_pattern' not in restore_kwargs: |
| python | src/training/utils.py | 397 | "restore_kwargs must include 'filename_pattern'." | raise ValueError("restore_kwargs must include 'filename_pattern'.") |
| python | src/training/utils.py | 399 | 'filename_pattern' | active_filename_pattern = restore_kwargs['filename_pattern'] |
| python | src/training/utils.py | 400 | 'filename_pattern' | active_loader_kwargs = {k: v for k, v in restore_kwargs.items() if k != 'filename_pattern'} |
| python | src/training/utils.py | 412 | 'best_model' | best_epochs = _extract_checkpoint_epochs(checkpoint_dir, 'best_model', active_filename_pattern) |
| python | src/training/utils.py | 417 | 'best_model' | restored = restore_model(checkpoint_dir, 'best_model', custom_epoch, active_filename_pattern, active_loader_kwargs) |
| python | src/training/utils.py | 428 | 'best_model' | restored = restore_model(checkpoint_dir, 'best_model', epoch_value, active_filename_pattern, active_loader_kwargs) |
| python | src/training/utils.py | 445 | 'GAN' | 'GAN': GAN, |
| python | src/training/utils.py | 446 | 'scale_invariant_mae' | 'scale_invariant_mae': scale_invariant_mae, |
| python | src/training/utils.py | 447 | 'log_cosh_loss' | 'log_cosh_loss': log_cosh_loss, |
| python | src/training/utils.py | 448 | 'ssim_loss' | 'ssim_loss': ssim_loss, |
| python | src/training/utils.py | 472 | 2 | tf.reduce_max(y_true, axis=(1, 2, 3), keepdims=True) |
| python | src/training/utils.py | 472 | 3 | tf.reduce_max(y_true, axis=(1, 2, 3), keepdims=True) |
| python | src/training/utils.py | 473 | 2 | - tf.reduce_min(y_true, axis=(1, 2, 3), keepdims=True), |
| python | src/training/utils.py | 473 | 3 | - tf.reduce_min(y_true, axis=(1, 2, 3), keepdims=True), |
| python | src/training/utils.py | 474 | 1e-06 | tf.constant(1e-6, dtype=y_true.dtype), |
| python | src/training/utils.py | 476 | 2 | mae_per_sample = tf.reduce_mean(tf.abs(y_true - y_pred), axis=(1, 2, 3)) |
| python | src/training/utils.py | 476 | 3 | mae_per_sample = tf.reduce_mean(tf.abs(y_true - y_pred), axis=(1, 2, 3)) |
| python | src/training/utils.py | 497 | 2 | tf.reduce_max(y_true, axis=(1, 2, 3), keepdims=True) |
| python | src/training/utils.py | 497 | 3 | tf.reduce_max(y_true, axis=(1, 2, 3), keepdims=True) |
| python | src/training/utils.py | 498 | 2 | - tf.reduce_min(y_true, axis=(1, 2, 3), keepdims=True) |
| python | src/training/utils.py | 498 | 3 | - tf.reduce_min(y_true, axis=(1, 2, 3), keepdims=True) |
| python | src/training/utils.py | 499 | 1e-10 | + 1e-10 |
| python | src/training/utils.py | 502 | 2.0 | log_two = tf.math.log(tf.constant(2.0, dtype=residual.dtype)) |
| python | src/training/utils.py | 504 | 2.0 | residual + tf.nn.softplus(-2.0 * residual) - log_two, |
| python | src/training/utils.py | 505 | 2 | axis=(1, 2, 3), |
| python | src/training/utils.py | 505 | 3 | axis=(1, 2, 3), |
| python | src/training/utils.py | 510 | 11 | def ssim_loss(y_true, y_pred, win_size=11, win_sigma=1.5, k1=0.01, k2=0.03): |
| python | src/training/utils.py | 510 | 1.5 | def ssim_loss(y_true, y_pred, win_size=11, win_sigma=1.5, k1=0.01, k2=0.03): |
| python | src/training/utils.py | 510 | 0.01 | def ssim_loss(y_true, y_pred, win_size=11, win_sigma=1.5, k1=0.01, k2=0.03): |
| python | src/training/utils.py | 510 | 0.03 | def ssim_loss(y_true, y_pred, win_size=11, win_sigma=1.5, k1=0.01, k2=0.03): |
| python | src/training/utils.py | 542 | 2 | coords = tf.cast(tf.range(win_size) - win_size // 2, tf.float32) |
| python | src/training/utils.py | 543 | 0.5 | g1d = tf.exp(-0.5 * (coords / win_sigma) ** 2) |
| python | src/training/utils.py | 543 | 2 | g1d = tf.exp(-0.5 * (coords / win_sigma) ** 2) |
| python | src/training/utils.py | 550 | 1 | return tf.nn.conv2d(t, kernel, strides=1, padding='SAME') |
| python | src/training/utils.py | 559 | 0.0 | sigma_x_sq = tf.maximum(mu_xx, 0.0) |
| python | src/training/utils.py | 560 | 0.0 | sigma_y_sq = tf.maximum(mu_yy, 0.0) |
| python | src/training/utils.py | 564 | 2 | L  = (tf.reduce_max(y_true, axis=(1, 2, 3), keepdims=True) |
| python | src/training/utils.py | 564 | 3 | L  = (tf.reduce_max(y_true, axis=(1, 2, 3), keepdims=True) |
| python | src/training/utils.py | 565 | 2 | - tf.reduce_min(y_true, axis=(1, 2, 3), keepdims=True)) |
| python | src/training/utils.py | 565 | 3 | - tf.reduce_min(y_true, axis=(1, 2, 3), keepdims=True)) |
| python | src/training/utils.py | 566 | 2 | c1 = (k1 * L) ** 2 |
| python | src/training/utils.py | 567 | 2 | c2 = (k2 * L) ** 2 |
| python | src/training/utils.py | 570 | 2.0 | numerator   = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2) |
| python | src/training/utils.py | 571 | 2 | denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2) |
| python | src/training/utils.py | 572 | 1e-10 | ssim_map    = numerator / (denominator + 1e-10) |
| python | src/training/utils.py | 574 | 1.0 | return 1.0 - tf.reduce_mean(ssim_map) |
| python | src/training/utils.py | 578 | 32 | def create_tf_dataset(images, sample_generator, generator_kwargs, batch_size=32, scaling=None, augment=True): |
| python | src/training/utils.py | 602 | 1 | metadata_shape = 1 |
| python | src/training/utils.py | 603 | 'log_min_max' | if scaling in ['log_min_max']: |
| python | src/training/utils.py | 604 | 7 | metadata_shape = 7 |
| python | src/training/utils.py | 605 | 'z_scale' | elif scaling in ['z_scale', 'min_max']: |
| python | src/training/utils.py | 605 | 'min_max' | elif scaling in ['z_scale', 'min_max']: |
| python | src/training/utils.py | 606 | 5 | metadata_shape = 5 |
| python | src/training/utils.py | 623 | 0 | rotation_steps = tf.random.uniform([], 0, 4, dtype=tf.int32) |
| python | src/training/utils.py | 623 | 4 | rotation_steps = tf.random.uniform([], 0, 4, dtype=tf.int32) |
| python | src/training/utils.py | 627 | 0 | flip_left_right = tf.random.uniform([], 0, 1) > 0.5 |
| python | src/training/utils.py | 627 | 1 | flip_left_right = tf.random.uniform([], 0, 1) > 0.5 |
| python | src/training/utils.py | 627 | 0.5 | flip_left_right = tf.random.uniform([], 0, 1) > 0.5 |
| python | src/training/utils.py | 628 | 0 | flip_up_down = tf.random.uniform([], 0, 1) > 0.5 |
| python | src/training/utils.py | 628 | 1 | flip_up_down = tf.random.uniform([], 0, 1) > 0.5 |
| python | src/training/utils.py | 628 | 0.5 | flip_up_down = tf.random.uniform([], 0, 1) > 0.5 |
| python | src/training/utils.py | 646 | 0.5 | def is_relevant_crop(stats, delta=0.5): |
| python | src/training/utils.py | 663 | 'full_abs_mean' | abs_uncropped = stats['full_abs_mean'] |
| python | src/training/utils.py | 664 | 'crop_abs_mean' | abs_crop = stats['crop_abs_mean'] |
| python | src/training/utils.py | 706 | 1 | data['valid'] = data.apply(lambda row: is_relevant_crop(row, delta=delta), axis=1) |
| python | src/training/utils.py | 720 | 'Must provide either (base, exponent) or noise_ratio for sorting.' | raise ValueError('Must provide either (base, exponent) or noise_ratio for sorting.') |
| python | src/training/utils.py | 722 | 'temp_index' | data.loc[:, 'temp_index'] = np.arange(0, len(data)) |
| python | src/training/utils.py | 722 | 0 | data.loc[:, 'temp_index'] = np.arange(0, len(data)) |
| python | src/training/utils.py | 724 | 0 | already_in = 0 |
| python | src/training/utils.py | 736 | 'temp_index' | data = data[~data['temp_index'].isin(result['temp_index'])] |
| python | src/training/utils.py | 740 | 1 | return pd.concat(dfs, ignore_index=True).iloc[:old_n_samples].sample(frac=1).reset_index(drop=True) |
| python | src/training/utils.py | 788 | 0 | if n_samples < 0: |
| python | src/training/utils.py | 789 | 'n_samples must be non-negative.' | raise ValueError('n_samples must be non-negative.') |
| python | src/training/utils.py | 790 | 0 | if occurrences_per_col_D < 0: |
| python | src/training/utils.py | 791 | 'occurrences_per_col_D must be non-negative.' | raise ValueError('occurrences_per_col_D must be non-negative.') |
| python | src/training/utils.py | 792 | 0 | if len(quantiles) == 0: |
| python | src/training/utils.py | 797 | 0 | previous_quantile = 0 |
| python | src/training/utils.py | 799 | 100 | if quantile <= previous_quantile or quantile > 100: |
| python | src/training/utils.py | 802 | 100 | if quantiles[-1] != 100: |
| python | src/training/utils.py | 806 | 0 | if total_percentage <= 0: |
| python | src/training/utils.py | 809 | 0 | if data.empty or n_samples == 0: |
| python | src/training/utils.py | 857 | 0 | if occurrences_per_col_D > 0: |
| python | src/training/utils.py | 863 | 0 | current_group_count = selected_group_counts.get(group_value, 0) |
| python | src/training/utils.py | 868 | 1 | selected_group_counts[group_value] = current_group_count + 1 |
| python | src/training/utils.py | 871 | 0 | if remaining_needed <= 0: |
| python | src/training/utils.py | 883 | 0 | if leftover > 0: |
| python | src/training/utils.py | 894 | 0 | lower_quantile = 0 |
| python | src/training/utils.py | 896 | 100.0 | start = int(np.floor((lower_quantile / 100.0) * remainder_count)) |
| python | src/training/utils.py | 897 | 100.0 | end = int(np.floor((upper_quantile / 100.0) * remainder_count)) |
| python | src/training/utils.py | 909 | 0 | selected_group_counts[group_value] = selected_group_counts.get(group_value, 0) + 1 |
| python | src/training/utils.py | 909 | 1 | selected_group_counts[group_value] = selected_group_counts.get(group_value, 0) + 1 |
| python | src/training/utils.py | 922 | 0 | selected_group_counts[group_value] = selected_group_counts.get(group_value, 0) + 1 |
| python | src/training/utils.py | 922 | 1 | selected_group_counts[group_value] = selected_group_counts.get(group_value, 0) + 1 |
| python | src/training/utils.py | 946 | 0.0 | if exposure_ratio <= 0.0: |
| python | src/training/utils.py | 953 | 4 | def _augment_samples_based_on_range(sigma, lowest_power=-4, highest_power=5, n_samples_per_magnitude=3): |
| python | src/training/utils.py | 953 | 5 | def _augment_samples_based_on_range(sigma, lowest_power=-4, highest_power=5, n_samples_per_magnitude=3): |
| python | src/training/utils.py | 953 | 3 | def _augment_samples_based_on_range(sigma, lowest_power=-4, highest_power=5, n_samples_per_magnitude=3): |
| python | src/training/utils.py | 979 | 0.0 | if is_not_nan(sigma) and sigma != 0.0: |
| python | src/training/utils.py | 981 | 10.0 | base = np.floor(sigma / (10.0**exponent)) |
| python | src/training/utils.py | 983 | 1 | for x in range(exponent, highest_power + 1): |
| python | src/training/utils.py | 985 | 9 | stop = 9 |
| python | src/training/utils.py | 987 | 10.0 | sigma_value = b * 10.0**x |
| python | src/training/utils.py | 1015 | 'name_col' | name_col = kwargs_data['name_col'] |
| python | src/training/utils.py | 1016 | 'location_col' | location_col = kwargs_data['location_col'] |
| python | src/training/utils.py | 1018 | 'exposure_col' | exposure_col = kwargs_data['exposure_col'] |
| python | src/training/utils.py | 1019 | 'lowest_power' | lowest_power = kwargs_data['lowest_power'] |
| python | src/training/utils.py | 1020 | 'highest_power' | highest_power = kwargs_data['highest_power'] |
| python | src/training/utils.py | 1021 | 'n_samples_per_magnitude' | n_samples_per_magnitude = kwargs_data['n_samples_per_magnitude'] |
| python | src/training/utils.py | 1023 | 'stats_column_map' | scm = kwargs_data['stats_column_map'] |
| python | src/training/utils.py | 1024 | 'original_stats_prefix' | prefix = kwargs_data['original_stats_prefix'] |
| python | src/training/utils.py | 1027 | 'std_bkg' | sigma = row[scm['std_bkg']] |
| python | src/training/utils.py | 1029 | 'std_bkg' | sigma = row[org_scm['std_bkg']] |
| python | src/training/utils.py | 1048 | 2 | diff_sigma = np.sqrt(np.abs(new_sigma**2 - sigma**2)) |
| python | src/training/utils.py | 1049 | 0 | new_exp_time = exposure_value * (new_sigma / sigma)**2 if sigma != 0 else 0.0 |
| python | src/training/utils.py | 1049 | 2 | new_exp_time = exposure_value * (new_sigma / sigma)**2 if sigma != 0 else 0.0 |
| python | src/training/utils.py | 1051 | 'exponent_diff' | 'name': row[name_col], 'base': base, 'exponent': exponent, 'exponent_diff': exponent_difference, |
| python | src/training/utils.py | 1052 | 'combined_sigma' | 'combined_sigma': new_sigma, 'org_sigma': sigma, 'diff_sigma': diff_sigma, |
| python | src/training/utils.py | 1052 | 'org_sigma' | 'combined_sigma': new_sigma, 'org_sigma': sigma, 'diff_sigma': diff_sigma, |
| python | src/training/utils.py | 1052 | 'diff_sigma' | 'combined_sigma': new_sigma, 'org_sigma': sigma, 'diff_sigma': diff_sigma, |
| python | src/training/utils.py | 1054 | 'crop_abs_mean' | 'crop_abs_mean': row[scm['abs_mean']], 'full_abs_mean': row[org_scm['abs_mean']], |
| python | src/training/utils.py | 1054 | 'full_abs_mean' | 'crop_abs_mean': row[scm['abs_mean']], 'full_abs_mean': row[org_scm['abs_mean']], |
| python | src/training/utils.py | 1054 | 'abs_mean' | 'crop_abs_mean': row[scm['abs_mean']], 'full_abs_mean': row[org_scm['abs_mean']], |
| python | src/training/utils.py | 1055 | 'crop_abs_median' | 'crop_abs_median': row[scm['abs_median']], 'full_abs_median': row[org_scm['abs_median']], |
| python | src/training/utils.py | 1055 | 'full_abs_median' | 'crop_abs_median': row[scm['abs_median']], 'full_abs_median': row[org_scm['abs_median']], |
| python | src/training/utils.py | 1055 | 'abs_median' | 'crop_abs_median': row[scm['abs_median']], 'full_abs_median': row[org_scm['abs_median']], |
| python | src/training/utils.py | 1056 | 'crop_median_bkg' | 'crop_median_bkg':row[scm['median_bkg']], 'full_median_bkg': row[org_scm['median_bkg']], |
| python | src/training/utils.py | 1056 | 'full_median_bkg' | 'crop_median_bkg':row[scm['median_bkg']], 'full_median_bkg': row[org_scm['median_bkg']], |
| python | src/training/utils.py | 1056 | 'median_bkg' | 'crop_median_bkg':row[scm['median_bkg']], 'full_median_bkg': row[org_scm['median_bkg']], |
| python | src/training/utils.py | 1057 | 'crop_mean_bkg' | 'crop_mean_bkg': row[scm['mean_bkg']], 'full_mean_bkg': row[org_scm['mean_bkg']], |
| python | src/training/utils.py | 1057 | 'full_mean_bkg' | 'crop_mean_bkg': row[scm['mean_bkg']], 'full_mean_bkg': row[org_scm['mean_bkg']], |
| python | src/training/utils.py | 1057 | 'mean_bkg' | 'crop_mean_bkg': row[scm['mean_bkg']], 'full_mean_bkg': row[org_scm['mean_bkg']], |
| python | src/training/utils.py | 1058 | 'crop_max_bkg' | 'crop_max_bkg': row[scm['max_bkg']], 'full_max_bkg': row[org_scm['max_bkg']], |
| python | src/training/utils.py | 1058 | 'full_max_bkg' | 'crop_max_bkg': row[scm['max_bkg']], 'full_max_bkg': row[org_scm['max_bkg']], |
| python | src/training/utils.py | 1058 | 'max_bkg' | 'crop_max_bkg': row[scm['max_bkg']], 'full_max_bkg': row[org_scm['max_bkg']], |
| python | src/training/utils.py | 1059 | 'crop_median_src' | 'crop_median_src': row[scm['median_src']], 'full_median_src': row[org_scm['median_src']], |
| python | src/training/utils.py | 1059 | 'full_median_src' | 'crop_median_src': row[scm['median_src']], 'full_median_src': row[org_scm['median_src']], |
| python | src/training/utils.py | 1059 | 'median_src' | 'crop_median_src': row[scm['median_src']], 'full_median_src': row[org_scm['median_src']], |
| python | src/training/utils.py | 1060 | 'crop_mean_src' | 'crop_mean_src': row[scm['mean_src']], 'full_mean_src': row[org_scm['mean_src']], |
| python | src/training/utils.py | 1060 | 'full_mean_src' | 'crop_mean_src': row[scm['mean_src']], 'full_mean_src': row[org_scm['mean_src']], |
| python | src/training/utils.py | 1060 | 'mean_src' | 'crop_mean_src': row[scm['mean_src']], 'full_mean_src': row[org_scm['mean_src']], |
| python | src/training/utils.py | 1061 | 'crop_max_src' | 'crop_max_src': row[scm['max_src']], 'full_max_src': row[org_scm['max_src']], |
| python | src/training/utils.py | 1061 | 'full_max_src' | 'crop_max_src': row[scm['max_src']], 'full_max_src': row[org_scm['max_src']], |
| python | src/training/utils.py | 1061 | 'max_src' | 'crop_max_src': row[scm['max_src']], 'full_max_src': row[org_scm['max_src']], |
| python | src/training/utils.py | 1062 | 'sm_mean_NSR' | 'sm_mean_NSR': row[scm['mean_src']]/new_sigma if new_sigma > 0 else 0.0, 'org_mean_NSR': row[scm['mean_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1062 | 'org_mean_NSR' | 'sm_mean_NSR': row[scm['mean_src']]/new_sigma if new_sigma > 0 else 0.0, 'org_mean_NSR': row[scm['mean_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1062 | 0 | 'sm_mean_NSR': row[scm['mean_src']]/new_sigma if new_sigma > 0 else 0.0, 'org_mean_NSR': row[scm['mean_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1062 | 'mean_src' | 'sm_mean_NSR': row[scm['mean_src']]/new_sigma if new_sigma > 0 else 0.0, 'org_mean_NSR': row[scm['mean_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1063 | 'sm_median_NSR' | 'sm_median_NSR': row[scm['median_src']]/new_sigma if new_sigma > 0 else 0.0, 'org_median_NSR': row[scm['median_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1063 | 'org_median_NSR' | 'sm_median_NSR': row[scm['median_src']]/new_sigma if new_sigma > 0 else 0.0, 'org_median_NSR': row[scm['median_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1063 | 0 | 'sm_median_NSR': row[scm['median_src']]/new_sigma if new_sigma > 0 else 0.0, 'org_median_NSR': row[scm['median_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1063 | 'median_src' | 'sm_median_NSR': row[scm['median_src']]/new_sigma if new_sigma > 0 else 0.0, 'org_median_NSR': row[scm['median_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1064 | 'sm_peak_NSR' | 'sm_peak_NSR': row[scm['max_src']]/new_sigma if diff_sigma > 0 else 0.0, 'org_peak_NSR': row[scm['max_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1064 | 'org_peak_NSR' | 'sm_peak_NSR': row[scm['max_src']]/new_sigma if diff_sigma > 0 else 0.0, 'org_peak_NSR': row[scm['max_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1064 | 0 | 'sm_peak_NSR': row[scm['max_src']]/new_sigma if diff_sigma > 0 else 0.0, 'org_peak_NSR': row[scm['max_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1064 | 'max_src' | 'sm_peak_NSR': row[scm['max_src']]/new_sigma if diff_sigma > 0 else 0.0, 'org_peak_NSR': row[scm['max_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1065 | 'exp_time' | 'exp_time': exposure_value, 'new_exp_time': new_exp_time, 'exp_ratio': exposure_value/new_exp_time if new_exp_time > 0 else 0.0 |
| python | src/training/utils.py | 1065 | 'new_exp_time' | 'exp_time': exposure_value, 'new_exp_time': new_exp_time, 'exp_ratio': exposure_value/new_exp_time if new_exp_time > 0 else 0.0 |
| python | src/training/utils.py | 1065 | 'exp_ratio' | 'exp_time': exposure_value, 'new_exp_time': new_exp_time, 'exp_ratio': exposure_value/new_exp_time if new_exp_time > 0 else 0.0 |
| python | src/training/utils.py | 1065 | 0 | 'exp_time': exposure_value, 'new_exp_time': new_exp_time, 'exp_ratio': exposure_value/new_exp_time if new_exp_time > 0 else 0.0 |
| python | src/training/utils.py | 1092 | 'name_col' | name_col = kwargs_data['name_col'] |
| python | src/training/utils.py | 1093 | 'location_col' | location_col = kwargs_data['location_col'] |
| python | src/training/utils.py | 1095 | 'exposure_col' | exposure_col = kwargs_data['exposure_col'] |
| python | src/training/utils.py | 1097 | 'stats_column_map' | scm = kwargs_data['stats_column_map'] |
| python | src/training/utils.py | 1098 | 'original_stats_prefix' | prefix = kwargs_data['original_stats_prefix'] |
| python | src/training/utils.py | 1101 | 'ratio_initial' | ratio_initial = kwargs_data['ratio_initial'] |
| python | src/training/utils.py | 1102 | 'ratio_count' | ratio_count = kwargs_data['ratio_count'] |
| python | src/training/utils.py | 1103 | 'ratio_growth' | ratio_growth = kwargs_data['ratio_growth'] |
| python | src/training/utils.py | 1106 | 'std_bkg' | sigma = row[scm['std_bkg']] |
| python | src/training/utils.py | 1108 | 'std_bkg' | sigma = row[org_scm['std_bkg']] |
| python | src/training/utils.py | 1119 | 2 | diff_sigma = np.sqrt(np.abs(new_sigma**2 - sigma**2)) |
| python | src/training/utils.py | 1122 | 'combined_sigma' | 'combined_sigma': new_sigma, 'org_sigma': sigma, 'diff_sigma': diff_sigma, |
| python | src/training/utils.py | 1122 | 'org_sigma' | 'combined_sigma': new_sigma, 'org_sigma': sigma, 'diff_sigma': diff_sigma, |
| python | src/training/utils.py | 1122 | 'diff_sigma' | 'combined_sigma': new_sigma, 'org_sigma': sigma, 'diff_sigma': diff_sigma, |
| python | src/training/utils.py | 1124 | 'crop_abs_mean' | 'crop_abs_mean': row[scm['abs_mean']], 'full_abs_mean': row[org_scm['abs_mean']], |
| python | src/training/utils.py | 1124 | 'full_abs_mean' | 'crop_abs_mean': row[scm['abs_mean']], 'full_abs_mean': row[org_scm['abs_mean']], |
| python | src/training/utils.py | 1124 | 'abs_mean' | 'crop_abs_mean': row[scm['abs_mean']], 'full_abs_mean': row[org_scm['abs_mean']], |
| python | src/training/utils.py | 1125 | 'crop_abs_median' | 'crop_abs_median': row[scm['abs_median']], 'full_abs_median': row[org_scm['abs_median']], |
| python | src/training/utils.py | 1125 | 'full_abs_median' | 'crop_abs_median': row[scm['abs_median']], 'full_abs_median': row[org_scm['abs_median']], |
| python | src/training/utils.py | 1125 | 'abs_median' | 'crop_abs_median': row[scm['abs_median']], 'full_abs_median': row[org_scm['abs_median']], |
| python | src/training/utils.py | 1126 | 'crop_median_bkg' | 'crop_median_bkg':row[scm['median_bkg']], 'full_median_bkg': row[org_scm['median_bkg']], |
| python | src/training/utils.py | 1126 | 'full_median_bkg' | 'crop_median_bkg':row[scm['median_bkg']], 'full_median_bkg': row[org_scm['median_bkg']], |
| python | src/training/utils.py | 1126 | 'median_bkg' | 'crop_median_bkg':row[scm['median_bkg']], 'full_median_bkg': row[org_scm['median_bkg']], |
| python | src/training/utils.py | 1127 | 'crop_mean_bkg' | 'crop_mean_bkg': row[scm['mean_bkg']], 'full_mean_bkg': row[org_scm['mean_bkg']], |
| python | src/training/utils.py | 1127 | 'full_mean_bkg' | 'crop_mean_bkg': row[scm['mean_bkg']], 'full_mean_bkg': row[org_scm['mean_bkg']], |
| python | src/training/utils.py | 1127 | 'mean_bkg' | 'crop_mean_bkg': row[scm['mean_bkg']], 'full_mean_bkg': row[org_scm['mean_bkg']], |
| python | src/training/utils.py | 1128 | 'crop_max_bkg' | 'crop_max_bkg': row[scm['max_bkg']], 'full_max_bkg': row[org_scm['max_bkg']], |
| python | src/training/utils.py | 1128 | 'full_max_bkg' | 'crop_max_bkg': row[scm['max_bkg']], 'full_max_bkg': row[org_scm['max_bkg']], |
| python | src/training/utils.py | 1128 | 'max_bkg' | 'crop_max_bkg': row[scm['max_bkg']], 'full_max_bkg': row[org_scm['max_bkg']], |
| python | src/training/utils.py | 1129 | 'crop_median_src' | 'crop_median_src': row[scm['median_src']], 'full_median_src': row[org_scm['median_src']], |
| python | src/training/utils.py | 1129 | 'full_median_src' | 'crop_median_src': row[scm['median_src']], 'full_median_src': row[org_scm['median_src']], |
| python | src/training/utils.py | 1129 | 'median_src' | 'crop_median_src': row[scm['median_src']], 'full_median_src': row[org_scm['median_src']], |
| python | src/training/utils.py | 1130 | 'crop_mean_src' | 'crop_mean_src': row[scm['mean_src']], 'full_mean_src': row[org_scm['mean_src']], |
| python | src/training/utils.py | 1130 | 'full_mean_src' | 'crop_mean_src': row[scm['mean_src']], 'full_mean_src': row[org_scm['mean_src']], |
| python | src/training/utils.py | 1130 | 'mean_src' | 'crop_mean_src': row[scm['mean_src']], 'full_mean_src': row[org_scm['mean_src']], |
| python | src/training/utils.py | 1131 | 'crop_max_src' | 'crop_max_src': row[scm['max_src']], 'full_max_src': row[org_scm['max_src']], |
| python | src/training/utils.py | 1131 | 'full_max_src' | 'crop_max_src': row[scm['max_src']], 'full_max_src': row[org_scm['max_src']], |
| python | src/training/utils.py | 1131 | 'max_src' | 'crop_max_src': row[scm['max_src']], 'full_max_src': row[org_scm['max_src']], |
| python | src/training/utils.py | 1132 | 'sm_mean_NSR' | 'sm_mean_NSR': row[scm['mean_src']]/new_sigma if new_sigma > 0 else 0.0, 'org_mean_NSR': row[scm['mean_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1132 | 'org_mean_NSR' | 'sm_mean_NSR': row[scm['mean_src']]/new_sigma if new_sigma > 0 else 0.0, 'org_mean_NSR': row[scm['mean_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1132 | 0 | 'sm_mean_NSR': row[scm['mean_src']]/new_sigma if new_sigma > 0 else 0.0, 'org_mean_NSR': row[scm['mean_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1132 | 'mean_src' | 'sm_mean_NSR': row[scm['mean_src']]/new_sigma if new_sigma > 0 else 0.0, 'org_mean_NSR': row[scm['mean_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1133 | 'sm_median_NSR' | 'sm_median_NSR': row[scm['median_src']]/new_sigma if new_sigma > 0 else 0.0, 'org_median_NSR': row[scm['median_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1133 | 'org_median_NSR' | 'sm_median_NSR': row[scm['median_src']]/new_sigma if new_sigma > 0 else 0.0, 'org_median_NSR': row[scm['median_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1133 | 0 | 'sm_median_NSR': row[scm['median_src']]/new_sigma if new_sigma > 0 else 0.0, 'org_median_NSR': row[scm['median_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1133 | 'median_src' | 'sm_median_NSR': row[scm['median_src']]/new_sigma if new_sigma > 0 else 0.0, 'org_median_NSR': row[scm['median_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1134 | 'sm_peak_NSR' | 'sm_peak_NSR': row[scm['max_src']]/new_sigma if diff_sigma > 0 else 0.0, 'org_peak_NSR': row[scm['max_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1134 | 'org_peak_NSR' | 'sm_peak_NSR': row[scm['max_src']]/new_sigma if diff_sigma > 0 else 0.0, 'org_peak_NSR': row[scm['max_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1134 | 0 | 'sm_peak_NSR': row[scm['max_src']]/new_sigma if diff_sigma > 0 else 0.0, 'org_peak_NSR': row[scm['max_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1134 | 'max_src' | 'sm_peak_NSR': row[scm['max_src']]/new_sigma if diff_sigma > 0 else 0.0, 'org_peak_NSR': row[scm['max_src']]/sigma if sigma > 0 else 0.0, |
| python | src/training/utils.py | 1135 | 'exp_time' | 'exp_time': exp, 'new_exp_time': new_exp, 'exp_ratio': ratio |
| python | src/training/utils.py | 1135 | 'new_exp_time' | 'exp_time': exp, 'new_exp_time': new_exp, 'exp_ratio': ratio |
| python | src/training/utils.py | 1135 | 'exp_ratio' | 'exp_time': exp, 'new_exp_time': new_exp, 'exp_ratio': ratio |
| python | src/training/utils.py | 1162 | 'min_exp_ratio' | min_ratio = float(kwargs_data['min_exp_ratio']) |
| python | src/training/utils.py | 1163 | 'max_exp_ratio' | max_ratio = float(kwargs_data['max_exp_ratio']) |
| python | src/training/utils.py | 1164 | 'min_exp_time' | min_exp_time = float(kwargs_data['min_exp_time']) |
| python | src/training/utils.py | 1167 | 'combined_sigma' | numeric_columns = ['combined_sigma', 'org_sigma', 'exp_time', 'new_exp_time', 'exp_ratio'] |
| python | src/training/utils.py | 1167 | 'org_sigma' | numeric_columns = ['combined_sigma', 'org_sigma', 'exp_time', 'new_exp_time', 'exp_ratio'] |
| python | src/training/utils.py | 1167 | 'exp_time' | numeric_columns = ['combined_sigma', 'org_sigma', 'exp_time', 'new_exp_time', 'exp_ratio'] |
| python | src/training/utils.py | 1167 | 'new_exp_time' | numeric_columns = ['combined_sigma', 'org_sigma', 'exp_time', 'new_exp_time', 'exp_ratio'] |
| python | src/training/utils.py | 1167 | 'exp_ratio' | numeric_columns = ['combined_sigma', 'org_sigma', 'exp_time', 'new_exp_time', 'exp_ratio'] |
| python | src/training/utils.py | 1173 | 'combined_sigma' | info = info[info['combined_sigma'] > info['org_sigma']] |
| python | src/training/utils.py | 1173 | 'org_sigma' | info = info[info['combined_sigma'] > info['org_sigma']] |
| python | src/training/utils.py | 1174 | 'exp_time' | info = info[info['exp_time'] > info['new_exp_time']] |
| python | src/training/utils.py | 1174 | 'new_exp_time' | info = info[info['exp_time'] > info['new_exp_time']] |
| python | src/training/utils.py | 1175 | 'exp_ratio' | info = info[info['exp_ratio'] >= min_ratio] |
| python | src/training/utils.py | 1176 | 'exp_ratio' | info = info[info['exp_ratio'] <= max_ratio] |
| python | src/training/utils.py | 1177 | 'new_exp_time' | info = info[info['new_exp_time'] >= min_exp_time] |
| python | src/training/utils.py | 1197 | 'abs_mean' | return filtering_df(info, x, kwargs_data['delta'], 'dataset', 'abs_mean', |
| python | src/training/utils.py | 1198 | 'exponent_diff' | base='base', exponent='exponent_diff', noise_ratio=None) |
| python | src/training/utils.py | 1216 | 'abs_mean' | return filtering_df(info, x, kwargs_data['delta'], 'dataset', 'abs_mean', |
| python | src/training/utils.py | 1217 | 'exp_ratio' | base=None, exponent=None, noise_ratio='exp_ratio') |
| python | src/training/utils.py | 1237 | 'exp_ratio' | return filtering_df_v2(info, x, col_A='exp_ratio', |
| python | src/training/utils.py | 1238 | 'sm_peak_NSR' | col_B='sm_peak_NSR', col_C='dataset', col_D='location', |
| python | src/training/utils.py | 1239 | 'occurrences_per_col_D' | occurrences_per_col_D=kwargs_data['occurrences_per_col_D'], |
| python | src/training/utils.py | 1267 | 'candidates_fn' | if category == 'candidates_fn': |
| python | src/training/utils.py | 1269 | 'sigma_range' | 'sigma_range': candidates_based_on_range, |
| python | src/training/utils.py | 1270 | 'exposure_ratio' | 'exposure_ratio': candidates_based_on_ratio, |
| python | src/training/utils.py | 1277 | 'post_filter_fn' | if category == 'post_filter_fn': |
| python | src/training/utils.py | 1286 | 'sample_fn' | if category == 'sample_fn': |
| python | src/training/utils.py | 1288 | 'sigma_range' | 'sigma_range': sample_range, |
| python | src/training/utils.py | 1289 | 'exposure_ratio' | 'exposure_ratio': sample_ratio, |
| python | src/training/utils.py | 1290 | 'exposure_ratio_v2' | 'exposure_ratio_v2': sample_range_v2, |
| python | src/training/utils.py | 1297 | 'noise_fn' | if category == 'noise_fn': |
| python | src/training/utils.py | 1299 | '_simulated_image_from_exposure' | '_simulated_image_from_exposure': _simulated_image_from_exposure, |
| python | src/training/utils.py | 1300 | '_simulated_image_from_poisson' | '_simulated_image_from_poisson': _simulated_image_from_poisson, |
| python | src/training/utils.py | 1307 | 'sigma_kernel_fn' | if category == 'sigma_kernel_fn': |
| python | src/training/utils.py | 1317 | 'stats_name_fn' | if category == 'stats_name_fn': |
| python | src/training/utils.py | 1321 | 'exp_ratio' | 'exp_ratio': _stats_name_from_exp_ratio, |
| python | src/visualization/prepare_images.py | 38 | 4 | columns = 4 |
| python | src/visualization/prepare_images.py | 39 | 0 | panel_rows = 0 if num_panels == 0 else int(np.ceil(num_panels / columns)) * 2 |
| python | src/visualization/prepare_images.py | 39 | 2 | panel_rows = 0 if num_panels == 0 else int(np.ceil(num_panels / columns)) * 2 |
| python | src/visualization/prepare_images.py | 40 | 1 | total_rows = 1 + panel_rows |
| python | src/visualization/prepare_images.py | 42 | 16 | fig = plt.figure(figsize=(16, max(6, 4 * total_rows))) |
| python | src/visualization/prepare_images.py | 42 | 6 | fig = plt.figure(figsize=(16, max(6, 4 * total_rows))) |
| python | src/visualization/prepare_images.py | 42 | 4 | fig = plt.figure(figsize=(16, max(6, 4 * total_rows))) |
| python | src/visualization/prepare_images.py | 48 | 1 | base_row = 1 + 2 * (index // columns) |
| python | src/visualization/prepare_images.py | 48 | 2 | base_row = 1 + 2 * (index // columns) |
| python | src/visualization/prepare_images.py | 50 | 1 | panel_axes.append((plt.subplot(gs[base_row, column]), plt.subplot(gs[base_row + 1, column]))) |
| python | src/visualization/prepare_images.py | 54 | 256 | def create_image(row, model, kwargs_data, ps=256): |
| python | src/visualization/prepare_images.py | 61 | 'nan_value' | nan_value    = kwargs_data['nan_value'] |
| python | src/visualization/prepare_images.py | 62 | 'posinf_value' | posinf_value = kwargs_data['posinf_value'] |
| python | src/visualization/prepare_images.py | 63 | 'neginf_value' | neginf_value = kwargs_data['neginf_value'] |
| python | src/visualization/prepare_images.py | 66 | 'type_of_image' | image = open_fits(location, type_of_image=kwargs_data['type_of_image']) |
| python | src/visualization/prepare_images.py | 88 | 'combined_sigma' | new_sigma  = candidate_row['combined_sigma'] |
| python | src/visualization/prepare_images.py | 89 | 'crop_median_bkg' | median_bkg = candidate_row['crop_median_bkg'] |
| python | src/visualization/prepare_images.py | 92 | 0 | rec_image   = model.predict(input_image, verbose=0) |
| python | src/visualization/prepare_images.py | 93 | 'exp_ratio' | ratio = candidate_row['exp_ratio'] |
| python | src/visualization/prepare_images.py | 131 | 'Noisy Image, $\\gamma$=' | noisy_ax.set_title(rf"Noisy Image, $\gamma$={int(gamma)}") |
| python | src/visualization/prepare_images.py | 137 | 'Reconstructed Image, $\\gamma$=' | rec_ax.set_title(rf"Reconstructed Image, $\gamma$={int(gamma)}") |
| python | src/visualization/prepare_images.py | 141 | 16 | ax0.set_title(f"{label}", fontsize=16) |
| python | src/visualization/prepare_images.py | 143 | 500 | fig.savefig(output_filepath, dpi=500) |
| python | src/visualization/prepare_images.py | 189 | 2 | if image.ndim < 2: |
| python | src/visualization/prepare_images.py | 200 | 6 | e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), |
| python | src/visualization/prepare_images.py | 201 | 1.5 | edgecolor='blue', facecolor='none', linewidth=1.5) |
| python | src/visualization/prepare_images.py | 207 | 6 | e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), |
| python | src/visualization/prepare_images.py | 208 | 1.5 | edgecolor='red', facecolor='none', linewidth=1.5) |
| python | src/visualization/prepare_images.py | 215 | 6 | e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), |
| python | src/visualization/prepare_images.py | 216 | 1.5 | edgecolor='green', facecolor='none', linewidth=1.5) |
| python | src/visualization/prepare_images.py | 222 | 6 | e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), |
| python | src/visualization/prepare_images.py | 223 | 1.5 | edgecolor='red', facecolor='none', linewidth=1.5) |
| python | src/visualization/prepare_images.py | 230 | 6 | e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), |
| python | src/visualization/prepare_images.py | 231 | 1.5 | edgecolor='yellow', facecolor='none', linewidth=1.5) |
| python | src/visualization/prepare_images.py | 237 | 6 | e = Ellipse(xy=(x, y), width=6*a, height=6*b, angle=np.degrees(theta), |
| python | src/visualization/prepare_images.py | 238 | 1.5 | edgecolor='red', facecolor='none', linewidth=1.5) |
| python | src/visualization/prepare_images.py | 299 | 'distance_threshold' | distance_upper_bound=kwargs['distance_threshold'] |
| python | src/visualization/prepare_images.py | 301 | 'distance_threshold' | matched_org_clean_indices = np.where(distances < kwargs['distance_threshold'])[0] |
| python | src/visualization/prepare_images.py | 301 | 0 | matched_org_clean_indices = np.where(distances < kwargs['distance_threshold'])[0] |
| python | src/visualization/prepare_images.py | 309 | 'distance_threshold' | distance_upper_bound=kwargs['distance_threshold'] |
| python | src/visualization/prepare_images.py | 311 | 'distance_threshold' | matched_org_noisy_clean_indices = np.where(noisy_distances < kwargs['distance_threshold'])[0] |
| python | src/visualization/prepare_images.py | 311 | 0 | matched_org_noisy_clean_indices = np.where(noisy_distances < kwargs['distance_threshold'])[0] |
| python | src/visualization/prepare_images.py | 378 | 0 | org_ellipses = ellipses[0][0] |
| python | src/visualization/prepare_images.py | 387 | 'Noisy Image, $\\gamma$=' | noisy_ax.set_title(rf"Noisy Image, $\gamma$={int(gamma)}") |
| python | src/visualization/prepare_images.py | 391 | 2 | for ellipse in ellipsex[2]: |
| python | src/visualization/prepare_images.py | 395 | 'Reconstructed Image, $\\gamma$=' | rec_ax.set_title(rf"Reconstructed Image, $\gamma$={int(gamma)}") |
| python | src/visualization/prepare_images.py | 399 | 1 | for ellipse in ellipsex[1]: |
| python | src/visualization/prepare_images.py | 402 | 16 | ax0.set_title(f"{label}", fontsize=16) |
| python | src/visualization/prepare_images.py | 404 | 2 | Line2D([0], [0], color='red', lw=2, label='Matched Sources'), |
| python | src/visualization/prepare_images.py | 405 | 2 | Line2D([0], [0], color='blue', lw=2, label='Unmatched Sources (Original)'), |
| python | src/visualization/prepare_images.py | 406 | 2 | Line2D([0], [0], color='green', lw=2, label='Unmatched Sources (Reconstructed)'), |
| python | src/visualization/prepare_images.py | 407 | 2 | Line2D([0], [0], color='yellow', lw=2, label='Unmatched Sources (Noisy)') |
| python | src/visualization/prepare_images.py | 410 | 0.5 | fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.01), |
| python | src/visualization/prepare_images.py | 410 | 0.01 | fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.01), |
| python | src/visualization/prepare_images.py | 411 | 3 | ncol=3, frameon=False) |
| python | src/visualization/prepare_images.py | 412 | 0.05 | fig.tight_layout(rect=(0, 0.05, 1, 1)) |
| python | src/visualization/prepare_images.py | 413 | 500 | fig.savefig(output_filepath, dpi=500) |
| python | src/visualization/prepare_images.py | 482 | 'prepare_images' | vis_cfg = cfg['prepare_images'] |
| python | src/visualization/prepare_images.py | 483 | 'data_kwargs' | data_cfg = dict(vis_cfg['data_kwargs']) |
| python | src/visualization/prepare_images.py | 485 | 'output_dir' | _output_dir     = vis_cfg['output_dir'] |
| python | src/visualization/prepare_images.py | 488 | 'metadata_filepath' | _metadata_fp    = vis_cfg['metadata_filepath'] |
| python | src/visualization/prepare_images.py | 490 | 'sample_n' | _sample_n       = vis_cfg['sample_n'] |
| python | src/visualization/prepare_images.py | 492 | 'kwargs_source' | _kwargs_source  = vis_cfg['kwargs_source'] |
| python | src/visualization/prepare_images.py | 493 | 'exp_column' | _exp_col = vis_cfg['exp_column'] |
| python | src/visualization/prepare_images.py | 494 | 'targ_col' | _targ_col = vis_cfg['targ_col'] |
| python | src/visualization/prepare_images.py | 497 | 'type_of_image' | plot_data_cfg['type_of_image'] = vis_cfg['type_of_image'] |
| python | src/visualization/prepare_images.py | 498 | 'ratio_initial' | for x in ('ratio_initial', 'ratio_count', 'ratio_growth'): |
| python | src/visualization/prepare_images.py | 498 | 'ratio_count' | for x in ('ratio_initial', 'ratio_count', 'ratio_growth'): |
| python | src/visualization/prepare_images.py | 498 | 'ratio_growth' | for x in ('ratio_initial', 'ratio_count', 'ratio_growth'): |
| python | src/visualization/prepare_images.py | 502 | 'model_dir' | vis_cfg['model_dir'], |
| python | src/visualization/prepare_images.py | 503 | 0 | condition = lambda df: np.zeros(len(df)) == 0, |
| python | src/visualization/prepare_images.py | 505 | 'model_prototype' | model_prototype=vis_cfg['model_prototype'], |
| python | src/visualization/prepare_images.py | 506 | 1 | n=1, |
| python | src/visualization/prepare_images.py | 507 | 0 | index=0, |
| python | src/visualization/prepare_images.py | 508 | 1 | concurrent_workers=1 |
| python | src/visualization/prepare_images.py | 522 | 1 | metadata_df = metadata_df[metadata_df['location'].str.contains('test', na=False)].sample(frac=1) |
| python | src/visualization/prepare_images.py | 525 | 'ratio_initial' | metadata_df[_exp_col] / plot_data_cfg['ratio_initial']*(plot_data_cfg['ratio_initial']**plot_data_cfg['ratio_growth']) >= _low |
| python | src/visualization/prepare_images.py | 525 | 'ratio_growth' | metadata_df[_exp_col] / plot_data_cfg['ratio_initial']*(plot_data_cfg['ratio_initial']**plot_data_cfg['ratio_growth']) >= _low |
| python | src/visualization/prepare_images.py | 539 | 1 | [f"ID{i+1}" for i in range(len(selected_metadata_df))] |
| python | src/visualization/prepare_images.py | 543 | ', sci_data_set_name: ' | combined_label = f'target: {target}, sci_data_set_name: {id_}' |
| python | src/visualization/prepare_images.py | 552 | '_' | os.path.join(_output_dir, f'{label}_{index_2}.png'), |
| python | src/visualization/prepare_images.py | 552 | '.png' | os.path.join(_output_dir, f'{label}_{index_2}.png'), |
| python | src/visualization/prepare_images.py | 556 | '_' | os.path.join(_output_dir, f'{label}_{index_2}_detections.png'), |
| python | src/visualization/prepare_images.py | 556 | '_detections.png' | os.path.join(_output_dir, f'{label}_{index_2}_detections.png'), |
| python | src/visualization/prepare_images.py | 560 | '_' | logging.warning(f"Visualization error for {label}_{index_2}: {err}") |
| python | src/visualization/prepare_images.py | 562 | '__main__' | if __name__ == "__main__": |
| python | src/visualization/prepare_plots.py | 17 | 0 | return np.nan if denominator == 0 else numerator / denominator |
| python | src/visualization/prepare_plots.py | 19 | 0 | return numerator / denominator.replace(0, np.nan) |
| python | src/visualization/prepare_plots.py | 20 | 0 | return numerator / np.where(np.asarray(denominator) == 0, np.nan, denominator) |
| python | src/visualization/prepare_plots.py | 35 | 0 | mask &= df[column] > 0 |
| python | src/visualization/prepare_plots.py | 44 | 2 | 2 * (summary_df['Precision'] * summary_df['Recall']), |
| python | src/visualization/prepare_plots.py | 47 | 'IOU_sum' | summary_df['IoU'] = safe_divide(summary_df['IOU_sum'], summary_df['union']) |
| python | src/visualization/prepare_plots.py | 48 | 'SNR_org' | summary_df['SNR_org'] = safe_divide(summary_df['SNR_org_sum'], summary_df['TP']) |
| python | src/visualization/prepare_plots.py | 48 | 'SNR_org_sum' | summary_df['SNR_org'] = safe_divide(summary_df['SNR_org_sum'], summary_df['TP']) |
| python | src/visualization/prepare_plots.py | 49 | 'SNR_rec' | summary_df['SNR_rec'] = safe_divide(summary_df['SNR_rec_sum'], summary_df['TP']) |
| python | src/visualization/prepare_plots.py | 49 | 'SNR_rec_sum' | summary_df['SNR_rec'] = safe_divide(summary_df['SNR_rec_sum'], summary_df['TP']) |
| python | src/visualization/prepare_plots.py | 50 | 'SNR_f' | summary_df['SNR_f'] = safe_divide(summary_df['SNR_org'], summary_df['SNR_rec']) |
| python | src/visualization/prepare_plots.py | 50 | 'SNR_org' | summary_df['SNR_f'] = safe_divide(summary_df['SNR_org'], summary_df['SNR_rec']) |
| python | src/visualization/prepare_plots.py | 50 | 'SNR_rec' | summary_df['SNR_f'] = safe_divide(summary_df['SNR_org'], summary_df['SNR_rec']) |
| python | src/visualization/prepare_plots.py | 51 | 'RFE_sum' | summary_df['RFE'] = safe_divide(summary_df['RFE_sum'], summary_df['TP']) |
| python | src/visualization/prepare_plots.py | 58 | 'sci_data_set_name' | my_set = set(df.loc[df['location'].str.contains('eval\|test', na=False), 'sci_data_set_name'].unique()) |
| python | src/visualization/prepare_plots.py | 61 | 'image_id' | df['dataset'] = df['image_id'].apply(lambda x: x.split('_')[0]) |
| python | src/visualization/prepare_plots.py | 61 | '_' | df['dataset'] = df['image_id'].apply(lambda x: x.split('_')[0]) |
| python | src/visualization/prepare_plots.py | 61 | 0 | df['dataset'] = df['image_id'].apply(lambda x: x.split('_')[0]) |
| python | src/visualization/prepare_plots.py | 63 | 'IOU_sum' | df['IOU_sum'] = df['IoU']*df['union'] |
| python | src/visualization/prepare_plots.py | 64 | 'SNR_org_sum' | df['SNR_org_sum'] = df['SNR_org']*df['TP'] |
| python | src/visualization/prepare_plots.py | 64 | 'SNR_org' | df['SNR_org_sum'] = df['SNR_org']*df['TP'] |
| python | src/visualization/prepare_plots.py | 65 | 'SNR_rec_sum' | df['SNR_rec_sum'] = df['SNR_rec']*df['TP'] |
| python | src/visualization/prepare_plots.py | 65 | 'SNR_rec' | df['SNR_rec_sum'] = df['SNR_rec']*df['TP'] |
| python | src/visualization/prepare_plots.py | 66 | 'RFE_sum' | df['RFE_sum'] = df['RFE']*df['TP'] |
| python | src/visualization/prepare_plots.py | 67 | 'exp_ratio' | df['exp_ratio'] = safe_divide(df['org_exp_time'], df['new_exp_time']) |
| python | src/visualization/prepare_plots.py | 67 | 'org_exp_time' | df['exp_ratio'] = safe_divide(df['org_exp_time'], df['new_exp_time']) |
| python | src/visualization/prepare_plots.py | 67 | 'new_exp_time' | df['exp_ratio'] = safe_divide(df['org_exp_time'], df['new_exp_time']) |
| python | src/visualization/prepare_plots.py | 69 | 'new_exp_time' | summarized_mosaic = df.groupby(['dataset', 'new_exp_time']).agg({ |
| python | src/visualization/prepare_plots.py | 73 | 'IOU_sum' | 'IOU_sum': 'sum', |
| python | src/visualization/prepare_plots.py | 74 | 'SNR_org_sum' | 'SNR_org_sum': 'sum', |
| python | src/visualization/prepare_plots.py | 75 | 'SNR_rec_sum' | 'SNR_rec_sum': 'sum', |
| python | src/visualization/prepare_plots.py | 77 | 'org_exp_time' | 'org_exp_time' : 'mean', |
| python | src/visualization/prepare_plots.py | 78 | 'RFE_sum' | 'RFE_sum' : 'sum', |
| python | src/visualization/prepare_plots.py | 79 | 'PSNR_rec' | 'PSNR_rec' : 'max', |
| python | src/visualization/prepare_plots.py | 80 | 'PSNR_noisy' | 'PSNR_noisy' : "max", |
| python | src/visualization/prepare_plots.py | 81 | 'SSIM_rec' | 'SSIM_rec' : "mean", |
| python | src/visualization/prepare_plots.py | 82 | 'SSIM_noisy' | 'SSIM_noisy' : 'mean' |
| python | src/visualization/prepare_plots.py | 86 | 'exp_ratio' | summarized_mosaic['exp_ratio'] = safe_divide(summarized_mosaic['org_exp_time'], summarized_mosaic['new_exp_time']) |
| python | src/visualization/prepare_plots.py | 86 | 'org_exp_time' | summarized_mosaic['exp_ratio'] = safe_divide(summarized_mosaic['org_exp_time'], summarized_mosaic['new_exp_time']) |
| python | src/visualization/prepare_plots.py | 86 | 'new_exp_time' | summarized_mosaic['exp_ratio'] = safe_divide(summarized_mosaic['org_exp_time'], summarized_mosaic['new_exp_time']) |
| python | src/visualization/prepare_plots.py | 87 | 'IOU_sum' | summarized_mosaic['IOU_sum'] = summarized_mosaic['IoU']*summarized_mosaic['union'] |
| python | src/visualization/prepare_plots.py | 88 | 'SNR_org_sum' | summarized_mosaic['SNR_org_sum'] = summarized_mosaic['SNR_org']*summarized_mosaic['TP'] |
| python | src/visualization/prepare_plots.py | 88 | 'SNR_org' | summarized_mosaic['SNR_org_sum'] = summarized_mosaic['SNR_org']*summarized_mosaic['TP'] |
| python | src/visualization/prepare_plots.py | 89 | 'SNR_rec_sum' | summarized_mosaic['SNR_rec_sum'] = summarized_mosaic['SNR_rec']*summarized_mosaic['TP'] |
| python | src/visualization/prepare_plots.py | 89 | 'SNR_rec' | summarized_mosaic['SNR_rec_sum'] = summarized_mosaic['SNR_rec']*summarized_mosaic['TP'] |
| python | src/visualization/prepare_plots.py | 90 | 'RFE_sum' | summarized_mosaic['RFE_sum'] = summarized_mosaic['RFE']*summarized_mosaic['TP'] |
| python | src/visualization/prepare_plots.py | 91 | 'exp_ratio' | summarized_mosaic['exp_ratio'] = np.ceil(summarized_mosaic['exp_ratio']) |
| python | src/visualization/prepare_plots.py | 92 | 'exp_ratio' | summarized_mosaic = summarized_mosaic.dropna(subset=['exp_ratio']).copy() |
| python | src/visualization/prepare_plots.py | 93 | 'exp_ratio' | summarized_mosaic['exp_ratio'] = summarized_mosaic['exp_ratio'].astype(int) |
| python | src/visualization/prepare_plots.py | 95 | 'exp_ratio' | summarized = summarized_mosaic.groupby(['exp_ratio']).agg({ |
| python | src/visualization/prepare_plots.py | 99 | 'IOU_sum' | 'IOU_sum': 'sum', |
| python | src/visualization/prepare_plots.py | 100 | 'SNR_org_sum' | 'SNR_org_sum': 'sum', |
| python | src/visualization/prepare_plots.py | 101 | 'SNR_rec_sum' | 'SNR_rec_sum': 'sum', |
| python | src/visualization/prepare_plots.py | 103 | 'org_exp_time' | 'org_exp_time' : 'mean', |
| python | src/visualization/prepare_plots.py | 104 | 'new_exp_time' | 'new_exp_time' : 'mean', |
| python | src/visualization/prepare_plots.py | 105 | 'RFE_sum' | 'RFE_sum' : 'sum', |
| python | src/visualization/prepare_plots.py | 106 | 'PSNR_rec' | 'PSNR_rec' : 'mean', |
| python | src/visualization/prepare_plots.py | 107 | 'PSNR_noisy' | 'PSNR_noisy' : "mean", |
| python | src/visualization/prepare_plots.py | 108 | 'SSIM_rec' | 'SSIM_rec' : "mean", |
| python | src/visualization/prepare_plots.py | 109 | 'SSIM_noisy' | 'SSIM_noisy' : 'mean' |
| python | src/visualization/prepare_plots.py | 113 | 'IOU_sum' | part_1 = summarized.drop(labels=['IOU_sum', 'SNR_org_sum', 'SNR_rec_sum', 'union', 'RFE_sum'], axis=1).sort_values(by=['exp_ratio'], ascending=False).reset_index(drop=True) |
| python | src/visualization/prepare_plots.py | 113 | 'SNR_org_sum' | part_1 = summarized.drop(labels=['IOU_sum', 'SNR_org_sum', 'SNR_rec_sum', 'union', 'RFE_sum'], axis=1).sort_values(by=['exp_ratio'], ascending=False).reset_index(drop=True) |
| python | src/visualization/prepare_plots.py | 113 | 'SNR_rec_sum' | part_1 = summarized.drop(labels=['IOU_sum', 'SNR_org_sum', 'SNR_rec_sum', 'union', 'RFE_sum'], axis=1).sort_values(by=['exp_ratio'], ascending=False).reset_index(drop=True) |
| python | src/visualization/prepare_plots.py | 113 | 'RFE_sum' | part_1 = summarized.drop(labels=['IOU_sum', 'SNR_org_sum', 'SNR_rec_sum', 'union', 'RFE_sum'], axis=1).sort_values(by=['exp_ratio'], ascending=False).reset_index(drop=True) |
| python | src/visualization/prepare_plots.py | 113 | 1 | part_1 = summarized.drop(labels=['IOU_sum', 'SNR_org_sum', 'SNR_rec_sum', 'union', 'RFE_sum'], axis=1).sort_values(by=['exp_ratio'], ascending=False).reset_index(drop=True) |
| python | src/visualization/prepare_plots.py | 113 | 'exp_ratio' | part_1 = summarized.drop(labels=['IOU_sum', 'SNR_org_sum', 'SNR_rec_sum', 'union', 'RFE_sum'], axis=1).sort_values(by=['exp_ratio'], ascending=False).reset_index(drop=True) |
| python | src/visualization/prepare_plots.py | 115 | 1 | summarized_mosaic['temp'] = 1 |
| python | src/visualization/prepare_plots.py | 120 | 'IOU_sum' | 'IOU_sum': 'sum', |
| python | src/visualization/prepare_plots.py | 121 | 'SNR_org_sum' | 'SNR_org_sum': 'sum', |
| python | src/visualization/prepare_plots.py | 122 | 'SNR_rec_sum' | 'SNR_rec_sum': 'sum', |
| python | src/visualization/prepare_plots.py | 124 | 'org_exp_time' | 'org_exp_time' : 'mean', |
| python | src/visualization/prepare_plots.py | 125 | 'new_exp_time' | 'new_exp_time' : 'mean', |
| python | src/visualization/prepare_plots.py | 126 | 'RFE_sum' | 'RFE_sum' : 'sum', |
| python | src/visualization/prepare_plots.py | 127 | 'PSNR_rec' | 'PSNR_rec' : 'mean', |
| python | src/visualization/prepare_plots.py | 128 | 'PSNR_noisy' | 'PSNR_noisy' : "mean", |
| python | src/visualization/prepare_plots.py | 129 | 'SSIM_rec' | 'SSIM_rec' : "mean", |
| python | src/visualization/prepare_plots.py | 130 | 'SSIM_noisy' | 'SSIM_noisy' : 'mean' |
| python | src/visualization/prepare_plots.py | 134 | 'IOU_sum' | part_2 = summarized.drop(labels=['IOU_sum', 'SNR_org_sum', 'SNR_rec_sum', 'union', 'RFE_sum', 'temp'], axis=1) |
| python | src/visualization/prepare_plots.py | 134 | 'SNR_org_sum' | part_2 = summarized.drop(labels=['IOU_sum', 'SNR_org_sum', 'SNR_rec_sum', 'union', 'RFE_sum', 'temp'], axis=1) |
| python | src/visualization/prepare_plots.py | 134 | 'SNR_rec_sum' | part_2 = summarized.drop(labels=['IOU_sum', 'SNR_org_sum', 'SNR_rec_sum', 'union', 'RFE_sum', 'temp'], axis=1) |
| python | src/visualization/prepare_plots.py | 134 | 'RFE_sum' | part_2 = summarized.drop(labels=['IOU_sum', 'SNR_org_sum', 'SNR_rec_sum', 'union', 'RFE_sum', 'temp'], axis=1) |
| python | src/visualization/prepare_plots.py | 134 | 1 | part_2 = summarized.drop(labels=['IOU_sum', 'SNR_org_sum', 'SNR_rec_sum', 'union', 'RFE_sum', 'temp'], axis=1) |
| python | src/visualization/prepare_plots.py | 135 | 'exp_ratio' | part_2['exp_ratio'] = 0 |
| python | src/visualization/prepare_plots.py | 135 | 0 | part_2['exp_ratio'] = 0 |
| python | src/visualization/prepare_plots.py | 136 | 'exp_ratio' | final_table = pd.concat([part_1, part_2], ignore_index=True).sort_values(by=['exp_ratio'], ascending=False).reset_index(drop=True) |
| python | src/visualization/prepare_plots.py | 139 | 15369.17570896557 | def convert_to_jansky(flux_e_per_s, PHOTPLAM=15369.17570896557, |
| python | src/visualization/prepare_plots.py | 140 | 1.92756031304868e-20 | PHOTFLAM =1.92756031304868e-20): |
| python | src/visualization/prepare_plots.py | 143 | 33356.4 | factor = 3.33564e4 |
| python | src/visualization/prepare_plots.py | 144 | 2 | return flux_e_per_s*PHOTFLAM*PHOTPLAM**2*factor |
| python | src/visualization/prepare_plots.py | 146 | 1.92756031304868e-20 | def convert_to_angstrom(flux_e_per_s, PHOTFLAM =1.92756031304868e-20): |
| python | src/visualization/prepare_plots.py | 154 | 0 | if pd.isna(flux_jansky) or flux_jansky <= 0: |
| python | src/visualization/prepare_plots.py | 156 | 2.5 | return -2.5*np.log10(flux_jansky) + 8.9 |
| python | src/visualization/prepare_plots.py | 156 | 8.9 | return -2.5*np.log10(flux_jansky) + 8.9 |
| python | src/visualization/prepare_plots.py | 161 | 'exp_ratio' | df['exp_ratio'] = df.apply( |
| python | src/visualization/prepare_plots.py | 162 | 'exp_time_rec' | lambda row: row['exp_time_rec'] / row['new_exp_time_rec'] |
| python | src/visualization/prepare_plots.py | 162 | 'new_exp_time_rec' | lambda row: row['exp_time_rec'] / row['new_exp_time_rec'] |
| python | src/visualization/prepare_plots.py | 163 | 'exp_time_rec' | if not pd.isna(row['exp_time_rec']) else |
| python | src/visualization/prepare_plots.py | 164 | 'exp_time_noise' | (row['exp_time_noise'] / row['new_exp_time_noise'] if not pd.isna(row['exp_time_noise']) else 0.0), |
| python | src/visualization/prepare_plots.py | 164 | 'new_exp_time_noise' | (row['exp_time_noise'] / row['new_exp_time_noise'] if not pd.isna(row['exp_time_noise']) else 0.0), |
| python | src/visualization/prepare_plots.py | 165 | 1 | axis=1 |
| python | src/visualization/prepare_plots.py | 167 | 'exp_ratio' | df['exp_ratio'] = df['exp_ratio'].round(2) |
| python | src/visualization/prepare_plots.py | 167 | 2 | df['exp_ratio'] = df['exp_ratio'].round(2) |
| python | src/visualization/prepare_plots.py | 169 | 'flux_x_org' | for col in ['flux_x_org', 'flux_y_org', 'flux_x_rec', 'flux_y_rec', 'flux_x_noise', 'flux_y_noise', 'cflux_org', 'cflux_rec', 'cflux_noise', |
| python | src/visualization/prepare_plots.py | 169 | 'flux_y_org' | for col in ['flux_x_org', 'flux_y_org', 'flux_x_rec', 'flux_y_rec', 'flux_x_noise', 'flux_y_noise', 'cflux_org', 'cflux_rec', 'cflux_noise', |
| python | src/visualization/prepare_plots.py | 169 | 'flux_x_rec' | for col in ['flux_x_org', 'flux_y_org', 'flux_x_rec', 'flux_y_rec', 'flux_x_noise', 'flux_y_noise', 'cflux_org', 'cflux_rec', 'cflux_noise', |
| python | src/visualization/prepare_plots.py | 169 | 'flux_y_rec' | for col in ['flux_x_org', 'flux_y_org', 'flux_x_rec', 'flux_y_rec', 'flux_x_noise', 'flux_y_noise', 'cflux_org', 'cflux_rec', 'cflux_noise', |
| python | src/visualization/prepare_plots.py | 169 | 'flux_x_noise' | for col in ['flux_x_org', 'flux_y_org', 'flux_x_rec', 'flux_y_rec', 'flux_x_noise', 'flux_y_noise', 'cflux_org', 'cflux_rec', 'cflux_noise', |
| python | src/visualization/prepare_plots.py | 169 | 'flux_y_noise' | for col in ['flux_x_org', 'flux_y_org', 'flux_x_rec', 'flux_y_rec', 'flux_x_noise', 'flux_y_noise', 'cflux_org', 'cflux_rec', 'cflux_noise', |
| python | src/visualization/prepare_plots.py | 169 | 'cflux_org' | for col in ['flux_x_org', 'flux_y_org', 'flux_x_rec', 'flux_y_rec', 'flux_x_noise', 'flux_y_noise', 'cflux_org', 'cflux_rec', 'cflux_noise', |
| python | src/visualization/prepare_plots.py | 169 | 'cflux_rec' | for col in ['flux_x_org', 'flux_y_org', 'flux_x_rec', 'flux_y_rec', 'flux_x_noise', 'flux_y_noise', 'cflux_org', 'cflux_rec', 'cflux_noise', |
| python | src/visualization/prepare_plots.py | 169 | 'cflux_noise' | for col in ['flux_x_org', 'flux_y_org', 'flux_x_rec', 'flux_y_rec', 'flux_x_noise', 'flux_y_noise', 'cflux_org', 'cflux_rec', 'cflux_noise', |
| python | src/visualization/prepare_plots.py | 170 | 'flux_err_org' | 'flux_err_org', 'flux_err_rec', 'flux_err_noise']: |
| python | src/visualization/prepare_plots.py | 170 | 'flux_err_rec' | 'flux_err_org', 'flux_err_rec', 'flux_err_noise']: |
| python | src/visualization/prepare_plots.py | 170 | 'flux_err_noise' | 'flux_err_org', 'flux_err_rec', 'flux_err_noise']: |
| python | src/visualization/prepare_plots.py | 171 | 'j_' | df['j_' + col] = df[col].apply(convert_to_jansky) |
| python | src/visualization/prepare_plots.py | 173 | 'abmag_' | df['abmag_' + col] = df['j_' + col].apply(calculate_abmag) |
| python | src/visualization/prepare_plots.py | 173 | 'j_' | df['abmag_' + col] = df['j_' + col].apply(calculate_abmag) |
| python | src/visualization/prepare_plots.py | 174 | 'a_' | df['a_' + col] = df[col].apply(convert_to_angstrom) |
| python | src/visualization/prepare_plots.py | 176 | 'flux_x_org' | for flux, flux_err in zip(['flux_x_org', 'flux_y_org', 'flux_x_rec', 'flux_y_rec', |
| python | src/visualization/prepare_plots.py | 176 | 'flux_y_org' | for flux, flux_err in zip(['flux_x_org', 'flux_y_org', 'flux_x_rec', 'flux_y_rec', |
| python | src/visualization/prepare_plots.py | 176 | 'flux_x_rec' | for flux, flux_err in zip(['flux_x_org', 'flux_y_org', 'flux_x_rec', 'flux_y_rec', |
| python | src/visualization/prepare_plots.py | 176 | 'flux_y_rec' | for flux, flux_err in zip(['flux_x_org', 'flux_y_org', 'flux_x_rec', 'flux_y_rec', |
| python | src/visualization/prepare_plots.py | 177 | 'flux_x_noise' | 'flux_x_noise', 'flux_y_noise', 'cflux_org', 'cflux_rec', 'cflux_noise'], |
| python | src/visualization/prepare_plots.py | 177 | 'flux_y_noise' | 'flux_x_noise', 'flux_y_noise', 'cflux_org', 'cflux_rec', 'cflux_noise'], |
| python | src/visualization/prepare_plots.py | 177 | 'cflux_org' | 'flux_x_noise', 'flux_y_noise', 'cflux_org', 'cflux_rec', 'cflux_noise'], |
| python | src/visualization/prepare_plots.py | 177 | 'cflux_rec' | 'flux_x_noise', 'flux_y_noise', 'cflux_org', 'cflux_rec', 'cflux_noise'], |
| python | src/visualization/prepare_plots.py | 177 | 'cflux_noise' | 'flux_x_noise', 'flux_y_noise', 'cflux_org', 'cflux_rec', 'cflux_noise'], |
| python | src/visualization/prepare_plots.py | 178 | 'flux_err_org' | ['flux_err_org', 'flux_err_org', 'flux_err_rec', 'flux_err_rec', |
| python | src/visualization/prepare_plots.py | 178 | 'flux_err_rec' | ['flux_err_org', 'flux_err_org', 'flux_err_rec', 'flux_err_rec', |
| python | src/visualization/prepare_plots.py | 179 | 'flux_err_noise' | 'flux_err_noise', 'flux_err_noise', 'flux_err_org', 'flux_err_org', 'flux_err_rec']): |
| python | src/visualization/prepare_plots.py | 179 | 'flux_err_org' | 'flux_err_noise', 'flux_err_noise', 'flux_err_org', 'flux_err_org', 'flux_err_rec']): |
| python | src/visualization/prepare_plots.py | 179 | 'flux_err_rec' | 'flux_err_noise', 'flux_err_noise', 'flux_err_org', 'flux_err_org', 'flux_err_rec']): |
| python | src/visualization/prepare_plots.py | 180 | 'delta_' | df['delta_' + flux] = safe_divide(df[flux], df[flux_err]) |
| python | src/visualization/prepare_plots.py | 181 | 'delta_j_' | df['delta_j_' + flux] = safe_divide(df['j_' + flux], df['j_' + flux_err]) |
| python | src/visualization/prepare_plots.py | 181 | 'j_' | df['delta_j_' + flux] = safe_divide(df['j_' + flux], df['j_' + flux_err]) |
| python | src/visualization/prepare_plots.py | 182 | 'delta_a_' | df['delta_a_' + flux] = safe_divide(df['a_' + flux], df['a_' + flux_err]) |
| python | src/visualization/prepare_plots.py | 182 | 'a_' | df['delta_a_' + flux] = safe_divide(df['a_' + flux], df['a_' + flux_err]) |
| python | src/visualization/prepare_plots.py | 188 | 0 | if int(exp_ratio) != 0: |
| python | src/visualization/prepare_plots.py | 189 | 'exp_ratio' | return df[df['exp_ratio'] == exp_ratio], rf'$\gamma={int(exp_ratio)}$' |
| python | src/visualization/prepare_plots.py | 189 | '$\\gamma=' | return df[df['exp_ratio'] == exp_ratio], rf'$\gamma={int(exp_ratio)}$' |
| python | src/visualization/prepare_plots.py | 195 | 100 | return max(100, int(np.sqrt(len(values) / 10))) |
| python | src/visualization/prepare_plots.py | 195 | 10 | return max(100, int(np.sqrt(len(values) / 10))) |
| python | src/visualization/prepare_plots.py | 216 | 99 | def clip_single_series(x_values, y_values, x_percentiles=(1, 99), y_percentiles=(1, 99)): |
| python | src/visualization/prepare_plots.py | 234 | 99 | def clip_aligned_paired_series(x_rec, y_rec, x_noise, y_noise, y_percentiles=(1, 99)): |
| python | src/visualization/prepare_plots.py | 238 | 'x_rec' | {'x_rec': x_rec, 'y_rec': y_rec, 'x_noise': x_noise, 'y_noise': y_noise} |
| python | src/visualization/prepare_plots.py | 238 | 'y_rec' | {'x_rec': x_rec, 'y_rec': y_rec, 'x_noise': x_noise, 'y_noise': y_noise} |
| python | src/visualization/prepare_plots.py | 238 | 'x_noise' | {'x_rec': x_rec, 'y_rec': y_rec, 'x_noise': x_noise, 'y_noise': y_noise} |
| python | src/visualization/prepare_plots.py | 238 | 'y_noise' | {'x_rec': x_rec, 'y_rec': y_rec, 'x_noise': x_noise, 'y_noise': y_noise} |
| python | src/visualization/prepare_plots.py | 243 | 'y_rec' | rec_low, rec_high = np.percentile(temp_df['y_rec'], y_percentiles) |
| python | src/visualization/prepare_plots.py | 244 | 'y_noise' | noise_low, noise_high = np.percentile(temp_df['y_noise'], y_percentiles) |
| python | src/visualization/prepare_plots.py | 248 | 'y_rec' | (temp_df['y_rec'] >= y_low) & (temp_df['y_rec'] <= y_high) |
| python | src/visualization/prepare_plots.py | 249 | 'y_noise' | & (temp_df['y_noise'] >= y_low) & (temp_df['y_noise'] <= y_high) |
| python | src/visualization/prepare_plots.py | 254 | 'x_rec' | return temp_df['x_rec'], temp_df['y_rec'], temp_df['x_noise'], temp_df['y_noise'], (y_low, y_high) |
| python | src/visualization/prepare_plots.py | 254 | 'y_rec' | return temp_df['x_rec'], temp_df['y_rec'], temp_df['x_noise'], temp_df['y_noise'], (y_low, y_high) |
| python | src/visualization/prepare_plots.py | 254 | 'x_noise' | return temp_df['x_rec'], temp_df['y_rec'], temp_df['x_noise'], temp_df['y_noise'], (y_low, y_high) |
| python | src/visualization/prepare_plots.py | 254 | 'y_noise' | return temp_df['x_rec'], temp_df['y_rec'], temp_df['x_noise'], temp_df['y_noise'], (y_low, y_high) |
| python | src/visualization/prepare_plots.py | 256 | 99 | def clip_separate_paired_series(x_rec, y_rec, x_noise, y_noise, y_percentiles=(1, 99)): |
| python | src/visualization/prepare_plots.py | 298 | 0 | if len(x_values) == 0 or len(y_values) == 0: |
| python | src/visualization/prepare_plots.py | 306 | 1 | ax.set_box_aspect(1) |
| python | src/visualization/prepare_plots.py | 310 | 1000 | one_to_one = np.linspace(left, right, 1000) |
| python | src/visualization/prepare_plots.py | 311 | 0.5 | ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5) |
| python | src/visualization/prepare_plots.py | 319 | 2 | mincnt=2, |
| python | src/visualization/prepare_plots.py | 320 | 0.1 | linewidths=0.1, |
| python | src/visualization/prepare_plots.py | 335 | 0.5 | color = cmap(0.5) |
| python | src/visualization/prepare_plots.py | 338 | 4 | gradient_line = mlines.Line2D([0], [0], color=color, lw=4, label=legend_label) |
| python | src/visualization/prepare_plots.py | 342 | 0.5 | bbox_to_anchor=(0.5, 1.05), |
| python | src/visualization/prepare_plots.py | 342 | 1.05 | bbox_to_anchor=(0.5, 1.05), |
| python | src/visualization/prepare_plots.py | 343 | 1 | ncol=1, |
| python | src/visualization/prepare_plots.py | 354 | 20 | fig.suptitle(suptitle, fontsize=20) |
| python | src/visualization/prepare_plots.py | 356 | 0.2 | fig.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92) |
| python | src/visualization/prepare_plots.py | 356 | 0.3 | fig.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92) |
| python | src/visualization/prepare_plots.py | 356 | 0.92 | fig.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92) |
| python | src/visualization/prepare_plots.py | 428 | 'norm_mode' | norm_mode = spec.get('norm_mode') |
| python | src/visualization/prepare_plots.py | 430 | 'norm_key' | return build_norm(df[columns[spec['norm_key']]], norm_quantiles) |
| python | src/visualization/prepare_plots.py | 468 | 'x_clip' | x_percentiles=spec.get('x_clip', (1, 99)), |
| python | src/visualization/prepare_plots.py | 468 | 99 | x_percentiles=spec.get('x_clip', (1, 99)), |
| python | src/visualization/prepare_plots.py | 469 | 'y_clip' | y_percentiles=spec.get('y_clip', (1, 99)), |
| python | src/visualization/prepare_plots.py | 469 | 99 | y_percentiles=spec.get('y_clip', (1, 99)), |
| python | src/visualization/prepare_plots.py | 476 | 'xlim_floor' | if spec.get('xlim_floor') is not None: |
| python | src/visualization/prepare_plots.py | 477 | 'xlim_floor' | xlim = (spec['xlim_floor'], x_values.max()) |
| python | src/visualization/prepare_plots.py | 490 | 'pre_window' | sub_df, raw_limits = apply_pre_window(sub_df, columns, spec.get('pre_window')) |
| python | src/visualization/prepare_plots.py | 494 | 'mask_mode' | if spec['mask_mode'] == 'shared': |
| python | src/visualization/prepare_plots.py | 501 | 'rec_mask' | rec_mask = build_mask(sub_df, columns, spec['rec_mask']) |
| python | src/visualization/prepare_plots.py | 502 | 'noise_mask' | noise_mask = build_mask(sub_df, columns, spec['noise_mask']) |
| python | src/visualization/prepare_plots.py | 513 | 'clip_mode' | clip_fn = clip_aligned_paired_series if spec['clip_mode'] == 'aligned' else clip_separate_paired_series |
| python | src/visualization/prepare_plots.py | 519 | 'y_clip' | y_percentiles=spec.get('y_clip', (1, 99)), |
| python | src/visualization/prepare_plots.py | 519 | 99 | y_percentiles=spec.get('y_clip', (1, 99)), |
| python | src/visualization/prepare_plots.py | 527 | 'x_rec' | 'x_rec': x_rec, |
| python | src/visualization/prepare_plots.py | 528 | 'y_rec' | 'y_rec': y_rec, |
| python | src/visualization/prepare_plots.py | 529 | 'x_noise' | 'x_noise': x_noise, |
| python | src/visualization/prepare_plots.py | 530 | 'y_noise' | 'y_noise': y_noise, |
| python | src/visualization/prepare_plots.py | 531 | 'x_limits' | 'x_limits': x_limits, |
| python | src/visualization/prepare_plots.py | 532 | 'y_limits' | 'y_limits': y_limits, |
| python | src/visualization/prepare_plots.py | 540 | 2 | fig, axes = plt.subplots(2, 4, figsize=(16, 8)) |
| python | src/visualization/prepare_plots.py | 540 | 4 | fig, axes = plt.subplots(2, 4, figsize=(16, 8)) |
| python | src/visualization/prepare_plots.py | 540 | 16 | fig, axes = plt.subplots(2, 4, figsize=(16, 8)) |
| python | src/visualization/prepare_plots.py | 540 | 8 | fig, axes = plt.subplots(2, 4, figsize=(16, 8)) |
| python | src/visualization/prepare_plots.py | 541 | 'rec_cmap' | cmap = plt.get_cmap(spec['rec_cmap']) |
| python | src/visualization/prepare_plots.py | 544 | 'exp_ratio' | for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())): |
| python | src/visualization/prepare_plots.py | 545 | 2 | column = index // 2 |
| python | src/visualization/prepare_plots.py | 546 | 1 | if column >= axes.shape[1]: |
| python | src/visualization/prepare_plots.py | 547 | 'Single-panel grid supports at most 8 exp_ratio groups. Stopping at exp_ratio %s.' | logging.warning('Single-panel grid supports at most 8 exp_ratio groups. Stopping at exp_ratio %s.', exp_ratio) |
| python | src/visualization/prepare_plots.py | 550 | 2 | row = index % 2 |
| python | src/visualization/prepare_plots.py | 551 | 'total_label' | sub_df, label = get_exp_ratio_subset(df, exp_ratio, spec.get('total_label', 'Total')) |
| python | src/visualization/prepare_plots.py | 554 | 'No plot data for %s at exp_ratio %s, skipping.' | logging.warning('No plot data for %s at exp_ratio %s, skipping.', spec['output_filename'], exp_ratio) |
| python | src/visualization/prepare_plots.py | 554 | 'output_filename' | logging.warning('No plot data for %s at exp_ratio %s, skipping.', spec['output_filename'], exp_ratio) |
| python | src/visualization/prepare_plots.py | 562 | 'rec_cmap' | cmap_name=spec['rec_cmap'], |
| python | src/visualization/prepare_plots.py | 565 | 1 | xlabel=spec['xlabel'] if row == 1 else None, |
| python | src/visualization/prepare_plots.py | 566 | 0 | ylabel=spec['ylabel'] if column == 0 else None, |
| python | src/visualization/prepare_plots.py | 569 | 'legend_stat' | legend_value=get_legend_value(panel_data['x'], spec.get('legend_stat', 'min')), |
| python | src/visualization/prepare_plots.py | 571 | 'one_to_one' | add_one_to_one=spec.get('one_to_one', False), |
| python | src/visualization/prepare_plots.py | 574 | 250 | finalize_plot(fig, output_filepath, spec['suptitle'], spec.get('dpi', 250)) |
| python | src/visualization/prepare_plots.py | 581 | 4 | fig, axes = plt.subplots(4, 4, figsize=(16, 16)) |
| python | src/visualization/prepare_plots.py | 581 | 16 | fig, axes = plt.subplots(4, 4, figsize=(16, 16)) |
| python | src/visualization/prepare_plots.py | 582 | 'rec_cmap' | rec_cmap = plt.get_cmap(spec['rec_cmap']) |
| python | src/visualization/prepare_plots.py | 583 | 'noise_cmap' | noise_cmap = plt.get_cmap(spec['noise_cmap']) |
| python | src/visualization/prepare_plots.py | 586 | 'exp_ratio' | for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())): |
| python | src/visualization/prepare_plots.py | 587 | 8 | if index >= 8: |
| python | src/visualization/prepare_plots.py | 588 | 'Paired grid supports at most 8 exp_ratio groups. Stopping at exp_ratio %s.' | logging.warning('Paired grid supports at most 8 exp_ratio groups. Stopping at exp_ratio %s.', exp_ratio) |
| python | src/visualization/prepare_plots.py | 591 | 4 | row = index % 4 |
| python | src/visualization/prepare_plots.py | 592 | 4 | rec_column, noise_column = (0, 1) if index < 4 else (2, 3) |
| python | src/visualization/prepare_plots.py | 592 | 2 | rec_column, noise_column = (0, 1) if index < 4 else (2, 3) |
| python | src/visualization/prepare_plots.py | 592 | 3 | rec_column, noise_column = (0, 1) if index < 4 else (2, 3) |
| python | src/visualization/prepare_plots.py | 593 | 'total_label' | sub_df, label = get_exp_ratio_subset(df, exp_ratio, spec.get('total_label', r"all $\gamma$'s")) |
| python | src/visualization/prepare_plots.py | 593 | "all $\\gamma$'s" | sub_df, label = get_exp_ratio_subset(df, exp_ratio, spec.get('total_label', r"all $\gamma$'s")) |
| python | src/visualization/prepare_plots.py | 596 | 'No plot data for %s at exp_ratio %s, skipping.' | logging.warning('No plot data for %s at exp_ratio %s, skipping.', spec['output_filename'], exp_ratio) |
| python | src/visualization/prepare_plots.py | 596 | 'output_filename' | logging.warning('No plot data for %s at exp_ratio %s, skipping.', spec['output_filename'], exp_ratio) |
| python | src/visualization/prepare_plots.py | 602 | 'x_rec' | panel_data['x_rec'], |
| python | src/visualization/prepare_plots.py | 603 | 'y_rec' | panel_data['y_rec'], |
| python | src/visualization/prepare_plots.py | 604 | 'rec_cmap' | cmap_name=spec['rec_cmap'], |
| python | src/visualization/prepare_plots.py | 606 | 'x_rec' | gridsize=get_gridsize(panel_data['x_rec']), |
| python | src/visualization/prepare_plots.py | 607 | 3 | xlabel=spec.get('xlabel_rec') if row == 3 else None, |
| python | src/visualization/prepare_plots.py | 607 | 'xlabel_rec' | xlabel=spec.get('xlabel_rec') if row == 3 else None, |
| python | src/visualization/prepare_plots.py | 608 | 'ylabel_rec' | ylabel=spec.get('ylabel_rec'), |
| python | src/visualization/prepare_plots.py | 609 | 0 | title='Reconstructed Image' if row == 0 else None, |
| python | src/visualization/prepare_plots.py | 610 | 'x_limits' | xlim=panel_data['x_limits'], |
| python | src/visualization/prepare_plots.py | 611 | 'y_limits' | ylim=panel_data['y_limits'], |
| python | src/visualization/prepare_plots.py | 612 | 'x_rec' | legend_label=f"{label}, N={len(panel_data['x_rec'])}", |
| python | src/visualization/prepare_plots.py | 613 | 'x_rec' | legend_value=get_legend_value(panel_data['x_rec'], spec.get('legend_stat', 'median')), |
| python | src/visualization/prepare_plots.py | 613 | 'legend_stat' | legend_value=get_legend_value(panel_data['x_rec'], spec.get('legend_stat', 'median')), |
| python | src/visualization/prepare_plots.py | 615 | 'one_to_one' | add_one_to_one=spec.get('one_to_one', False), |
| python | src/visualization/prepare_plots.py | 620 | 'x_noise' | panel_data['x_noise'], |
| python | src/visualization/prepare_plots.py | 621 | 'y_noise' | panel_data['y_noise'], |
| python | src/visualization/prepare_plots.py | 622 | 'noise_cmap' | cmap_name=spec['noise_cmap'], |
| python | src/visualization/prepare_plots.py | 624 | 'x_noise' | gridsize=get_gridsize(panel_data['x_noise']), |
| python | src/visualization/prepare_plots.py | 625 | 3 | xlabel=spec.get('xlabel_noise') if row == 3 else None, |
| python | src/visualization/prepare_plots.py | 625 | 'xlabel_noise' | xlabel=spec.get('xlabel_noise') if row == 3 else None, |
| python | src/visualization/prepare_plots.py | 626 | 'ylabel_noise' | ylabel=spec.get('ylabel_noise'), |
| python | src/visualization/prepare_plots.py | 627 | 0 | title='Noisy Image' if row == 0 else None, |
| python | src/visualization/prepare_plots.py | 628 | 'x_limits' | xlim=panel_data['x_limits'], |
| python | src/visualization/prepare_plots.py | 629 | 'y_limits' | ylim=panel_data['y_limits'], |
| python | src/visualization/prepare_plots.py | 630 | 'x_noise' | legend_label=f"{label}, N={len(panel_data['x_noise'])}", |
| python | src/visualization/prepare_plots.py | 631 | 'x_noise' | legend_value=get_legend_value(panel_data['x_noise'], spec.get('legend_stat', 'median')), |
| python | src/visualization/prepare_plots.py | 631 | 'legend_stat' | legend_value=get_legend_value(panel_data['x_noise'], spec.get('legend_stat', 'median')), |
| python | src/visualization/prepare_plots.py | 633 | 'one_to_one' | add_one_to_one=spec.get('one_to_one', False), |
| python | src/visualization/prepare_plots.py | 636 | 500 | finalize_plot(fig, output_filepath, spec['suptitle'], spec.get('dpi', 500)) |
| python | src/visualization/prepare_plots.py | 654 | 'Org. Flux $[erg\\ s^{-1}\\ cm^{-2}\\ \\AA^{-1}]$' | flux_label = r'Org. Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$' |
| python | src/visualization/prepare_plots.py | 655 | 'Rec. Flux $[erg\\ s^{-1}\\ cm^{-2}\\ \\AA^{-1}]$' | rec_flux_label = r'Rec. Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$' |
| python | src/visualization/prepare_plots.py | 656 | 'Noisy Flux $[erg\\ s^{-1}\\ cm^{-2}\\ \\AA^{-1}]$' | noisy_flux_label = r'Noisy Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$' |
| python | src/visualization/prepare_plots.py | 660 | '$\\Delta$Kron Radius $[arcsec]$' | delta_kron_label = r'$\Delta$Kron Radius $[arcsec]$' |
| python | src/visualization/prepare_plots.py | 665 | 'output_filename' | 'output_filename': 'flux_ratio_combined_both.png', |
| python | src/visualization/prepare_plots.py | 665 | 'flux_ratio_combined_both.png' | 'output_filename': 'flux_ratio_combined_both.png', |
| python | src/visualization/prepare_plots.py | 666 | 'Flux Ratio vs. $\\log$(Flux) (Detected in Both)' | 'suptitle': r"Flux Ratio vs. $\log$(Flux) (Detected in Both)", |
| python | src/visualization/prepare_plots.py | 667 | 'rec_cmap' | 'rec_cmap': rec_cmap, |
| python | src/visualization/prepare_plots.py | 668 | 'noise_cmap' | 'noise_cmap': noise_cmap, |
| python | src/visualization/prepare_plots.py | 669 | 'pre_window' | 'pre_window': {'key': 'org_flux', 'positive': True, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 669 | 'org_flux' | 'pre_window': {'key': 'org_flux', 'positive': True, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 669 | 98 | 'pre_window': {'key': 'org_flux', 'positive': True, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 670 | 'mask_mode' | 'mask_mode': 'shared', |
| python | src/visualization/prepare_plots.py | 671 | 'org_flux' | 'mask': {'positive': ['org_flux'], 'finite': ['rec_flux', 'noise_flux']}, |
| python | src/visualization/prepare_plots.py | 671 | 'rec_flux' | 'mask': {'positive': ['org_flux'], 'finite': ['rec_flux', 'noise_flux']}, |
| python | src/visualization/prepare_plots.py | 671 | 'noise_flux' | 'mask': {'positive': ['org_flux'], 'finite': ['rec_flux', 'noise_flux']}, |
| python | src/visualization/prepare_plots.py | 672 | 'org_flux' | 'rec': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'rec_flux', 'transform': 'ratio', 'reference': 'org_flux'}}, |
| python | src/visualization/prepare_plots.py | 672 | 'rec_flux' | 'rec': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'rec_flux', 'transform': 'ratio', 'reference': 'org_flux'}}, |
| python | src/visualization/prepare_plots.py | 673 | 'org_flux' | 'noise': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'noise_flux', 'transform': 'ratio', 'reference': 'org_flux'}}, |
| python | src/visualization/prepare_plots.py | 673 | 'noise_flux' | 'noise': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'noise_flux', 'transform': 'ratio', 'reference': 'org_flux'}}, |
| python | src/visualization/prepare_plots.py | 674 | 'clip_mode' | 'clip_mode': 'aligned', |
| python | src/visualization/prepare_plots.py | 675 | 'norm_mode' | 'norm_mode': 'column', |
| python | src/visualization/prepare_plots.py | 676 | 'norm_key' | 'norm_key': 'org_flux', |
| python | src/visualization/prepare_plots.py | 676 | 'org_flux' | 'norm_key': 'org_flux', |
| python | src/visualization/prepare_plots.py | 677 | 'legend_stat' | 'legend_stat': 'median', |
| python | src/visualization/prepare_plots.py | 678 | 'xlabel_rec' | 'xlabel_rec': flux_label, |
| python | src/visualization/prepare_plots.py | 679 | 'xlabel_noise' | 'xlabel_noise': flux_label, |
| python | src/visualization/prepare_plots.py | 680 | 'ylabel_rec' | 'ylabel_rec': r'Flux Ratio', |
| python | src/visualization/prepare_plots.py | 684 | 'output_filename' | 'output_filename': 'delta_kron_combined_both.png', |
| python | src/visualization/prepare_plots.py | 684 | 'delta_kron_combined_both.png' | 'output_filename': 'delta_kron_combined_both.png', |
| python | src/visualization/prepare_plots.py | 685 | 'Rec., Noisy $\\Delta$Kron Radius vs. Org. Kron Radius (Detected in Both)' | 'suptitle': r"Rec., Noisy $\Delta$Kron Radius vs. Org. Kron Radius (Detected in Both)", |
| python | src/visualization/prepare_plots.py | 686 | 'rec_cmap' | 'rec_cmap': rec_cmap, |
| python | src/visualization/prepare_plots.py | 687 | 'noise_cmap' | 'noise_cmap': noise_cmap, |
| python | src/visualization/prepare_plots.py | 688 | 'pre_window' | 'pre_window': {'key': 'org_kron', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 688 | 'org_kron' | 'pre_window': {'key': 'org_kron', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 688 | 98 | 'pre_window': {'key': 'org_kron', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 689 | 'mask_mode' | 'mask_mode': 'shared', |
| python | src/visualization/prepare_plots.py | 690 | 'org_kron' | 'mask': {'finite': ['org_kron', 'rec_kron', 'noise_kron']}, |
| python | src/visualization/prepare_plots.py | 690 | 'rec_kron' | 'mask': {'finite': ['org_kron', 'rec_kron', 'noise_kron']}, |
| python | src/visualization/prepare_plots.py | 690 | 'noise_kron' | 'mask': {'finite': ['org_kron', 'rec_kron', 'noise_kron']}, |
| python | src/visualization/prepare_plots.py | 691 | 'org_kron' | 'rec': {'x': {'key': 'org_kron'}, 'y': {'key': 'rec_kron', 'transform': 'delta', 'reference': 'org_kron'}}, |
| python | src/visualization/prepare_plots.py | 691 | 'rec_kron' | 'rec': {'x': {'key': 'org_kron'}, 'y': {'key': 'rec_kron', 'transform': 'delta', 'reference': 'org_kron'}}, |
| python | src/visualization/prepare_plots.py | 692 | 'org_kron' | 'noise': {'x': {'key': 'org_kron'}, 'y': {'key': 'noise_kron', 'transform': 'delta', 'reference': 'org_kron'}}, |
| python | src/visualization/prepare_plots.py | 692 | 'noise_kron' | 'noise': {'x': {'key': 'org_kron'}, 'y': {'key': 'noise_kron', 'transform': 'delta', 'reference': 'org_kron'}}, |
| python | src/visualization/prepare_plots.py | 693 | 'clip_mode' | 'clip_mode': 'aligned', |
| python | src/visualization/prepare_plots.py | 694 | 'norm_mode' | 'norm_mode': 'column', |
| python | src/visualization/prepare_plots.py | 695 | 'norm_key' | 'norm_key': 'org_kron', |
| python | src/visualization/prepare_plots.py | 695 | 'org_kron' | 'norm_key': 'org_kron', |
| python | src/visualization/prepare_plots.py | 696 | 'legend_stat' | 'legend_stat': 'min', |
| python | src/visualization/prepare_plots.py | 697 | 'xlabel_rec' | 'xlabel_rec': r'Kron Radius $[arcsec]$', |
| python | src/visualization/prepare_plots.py | 698 | 'xlabel_noise' | 'xlabel_noise': arcsec_label, |
| python | src/visualization/prepare_plots.py | 699 | 'ylabel_rec' | 'ylabel_rec': delta_kron_label, |
| python | src/visualization/prepare_plots.py | 703 | 'output_filename' | 'output_filename': 'flux_ratio_mag_combined_both.png', |
| python | src/visualization/prepare_plots.py | 703 | 'flux_ratio_mag_combined_both.png' | 'output_filename': 'flux_ratio_mag_combined_both.png', |
| python | src/visualization/prepare_plots.py | 705 | 'rec_cmap' | 'rec_cmap': rec_cmap, |
| python | src/visualization/prepare_plots.py | 706 | 'noise_cmap' | 'noise_cmap': noise_cmap, |
| python | src/visualization/prepare_plots.py | 707 | 'pre_window' | 'pre_window': {'key': 'org_mag', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 707 | 'org_mag' | 'pre_window': {'key': 'org_mag', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 707 | 98 | 'pre_window': {'key': 'org_mag', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 708 | 'mask_mode' | 'mask_mode': 'shared', |
| python | src/visualization/prepare_plots.py | 709 | 'org_flux' | 'mask': {'positive': ['org_flux'], 'finite': ['rec_flux', 'noise_flux', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 709 | 'rec_flux' | 'mask': {'positive': ['org_flux'], 'finite': ['rec_flux', 'noise_flux', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 709 | 'noise_flux' | 'mask': {'positive': ['org_flux'], 'finite': ['rec_flux', 'noise_flux', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 709 | 'org_mag' | 'mask': {'positive': ['org_flux'], 'finite': ['rec_flux', 'noise_flux', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 710 | 'org_mag' | 'rec': {'x': {'key': 'org_mag'}, 'y': {'key': 'rec_flux', 'transform': 'ratio', 'reference': 'org_flux'}}, |
| python | src/visualization/prepare_plots.py | 710 | 'rec_flux' | 'rec': {'x': {'key': 'org_mag'}, 'y': {'key': 'rec_flux', 'transform': 'ratio', 'reference': 'org_flux'}}, |
| python | src/visualization/prepare_plots.py | 710 | 'org_flux' | 'rec': {'x': {'key': 'org_mag'}, 'y': {'key': 'rec_flux', 'transform': 'ratio', 'reference': 'org_flux'}}, |
| python | src/visualization/prepare_plots.py | 711 | 'org_mag' | 'noise': {'x': {'key': 'org_mag'}, 'y': {'key': 'noise_flux', 'transform': 'ratio', 'reference': 'org_flux'}}, |
| python | src/visualization/prepare_plots.py | 711 | 'noise_flux' | 'noise': {'x': {'key': 'org_mag'}, 'y': {'key': 'noise_flux', 'transform': 'ratio', 'reference': 'org_flux'}}, |
| python | src/visualization/prepare_plots.py | 711 | 'org_flux' | 'noise': {'x': {'key': 'org_mag'}, 'y': {'key': 'noise_flux', 'transform': 'ratio', 'reference': 'org_flux'}}, |
| python | src/visualization/prepare_plots.py | 712 | 'clip_mode' | 'clip_mode': 'aligned', |
| python | src/visualization/prepare_plots.py | 713 | 'norm_mode' | 'norm_mode': 'column', |
| python | src/visualization/prepare_plots.py | 714 | 'norm_key' | 'norm_key': 'org_flux', |
| python | src/visualization/prepare_plots.py | 714 | 'org_flux' | 'norm_key': 'org_flux', |
| python | src/visualization/prepare_plots.py | 715 | 'legend_stat' | 'legend_stat': 'median', |
| python | src/visualization/prepare_plots.py | 716 | 'xlabel_rec' | 'xlabel_rec': abmag_label, |
| python | src/visualization/prepare_plots.py | 717 | 'xlabel_noise' | 'xlabel_noise': abmag_label, |
| python | src/visualization/prepare_plots.py | 718 | 'ylabel_rec' | 'ylabel_rec': r'Flux Ratio', |
| python | src/visualization/prepare_plots.py | 722 | 'output_filename' | 'output_filename': 'delta_kron_mag_combined_both.png', |
| python | src/visualization/prepare_plots.py | 722 | 'delta_kron_mag_combined_both.png' | 'output_filename': 'delta_kron_mag_combined_both.png', |
| python | src/visualization/prepare_plots.py | 723 | 'Rec., Noisy $\\Delta$Kron Radius vs. AB MAG (Detected in Both)' | 'suptitle': r"Rec., Noisy $\Delta$Kron Radius vs. AB MAG (Detected in Both)", |
| python | src/visualization/prepare_plots.py | 724 | 'rec_cmap' | 'rec_cmap': rec_cmap, |
| python | src/visualization/prepare_plots.py | 725 | 'noise_cmap' | 'noise_cmap': noise_cmap, |
| python | src/visualization/prepare_plots.py | 726 | 'pre_window' | 'pre_window': {'key': 'org_mag', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 726 | 'org_mag' | 'pre_window': {'key': 'org_mag', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 726 | 98 | 'pre_window': {'key': 'org_mag', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 727 | 'mask_mode' | 'mask_mode': 'shared', |
| python | src/visualization/prepare_plots.py | 728 | 'org_kron' | 'mask': {'finite': ['org_kron', 'rec_kron', 'noise_kron', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 728 | 'rec_kron' | 'mask': {'finite': ['org_kron', 'rec_kron', 'noise_kron', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 728 | 'noise_kron' | 'mask': {'finite': ['org_kron', 'rec_kron', 'noise_kron', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 728 | 'org_mag' | 'mask': {'finite': ['org_kron', 'rec_kron', 'noise_kron', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 729 | 'org_mag' | 'rec': {'x': {'key': 'org_mag'}, 'y': {'key': 'rec_kron', 'transform': 'delta', 'reference': 'org_kron'}}, |
| python | src/visualization/prepare_plots.py | 729 | 'rec_kron' | 'rec': {'x': {'key': 'org_mag'}, 'y': {'key': 'rec_kron', 'transform': 'delta', 'reference': 'org_kron'}}, |
| python | src/visualization/prepare_plots.py | 729 | 'org_kron' | 'rec': {'x': {'key': 'org_mag'}, 'y': {'key': 'rec_kron', 'transform': 'delta', 'reference': 'org_kron'}}, |
| python | src/visualization/prepare_plots.py | 730 | 'org_mag' | 'noise': {'x': {'key': 'org_mag'}, 'y': {'key': 'noise_kron', 'transform': 'delta', 'reference': 'org_kron'}}, |
| python | src/visualization/prepare_plots.py | 730 | 'noise_kron' | 'noise': {'x': {'key': 'org_mag'}, 'y': {'key': 'noise_kron', 'transform': 'delta', 'reference': 'org_kron'}}, |
| python | src/visualization/prepare_plots.py | 730 | 'org_kron' | 'noise': {'x': {'key': 'org_mag'}, 'y': {'key': 'noise_kron', 'transform': 'delta', 'reference': 'org_kron'}}, |
| python | src/visualization/prepare_plots.py | 731 | 'clip_mode' | 'clip_mode': 'aligned', |
| python | src/visualization/prepare_plots.py | 732 | 'norm_mode' | 'norm_mode': 'column', |
| python | src/visualization/prepare_plots.py | 733 | 'norm_key' | 'norm_key': 'org_kron', |
| python | src/visualization/prepare_plots.py | 733 | 'org_kron' | 'norm_key': 'org_kron', |
| python | src/visualization/prepare_plots.py | 734 | 'legend_stat' | 'legend_stat': 'min', |
| python | src/visualization/prepare_plots.py | 735 | 'xlabel_rec' | 'xlabel_rec': abmag_label, |
| python | src/visualization/prepare_plots.py | 736 | 'xlabel_noise' | 'xlabel_noise': abmag_label, |
| python | src/visualization/prepare_plots.py | 737 | 'ylabel_rec' | 'ylabel_rec': delta_kron_label, |
| python | src/visualization/prepare_plots.py | 741 | 'output_filename' | 'output_filename': 'delta_kron_mag_rec.png', |
| python | src/visualization/prepare_plots.py | 741 | 'delta_kron_mag_rec.png' | 'output_filename': 'delta_kron_mag_rec.png', |
| python | src/visualization/prepare_plots.py | 742 | 'Rec. $\\Delta$Kron Radius vs. AB MAG (Detected only in Rec.)' | 'suptitle': r"Rec. $\Delta$Kron Radius vs. AB MAG (Detected only in Rec.)", |
| python | src/visualization/prepare_plots.py | 743 | 'rec_cmap' | 'rec_cmap': rec_cmap, |
| python | src/visualization/prepare_plots.py | 744 | 'org_kron' | 'mask': {'finite': ['org_kron', 'rec_kron', 'org_mag'], 'missing': ['noise_kron']}, |
| python | src/visualization/prepare_plots.py | 744 | 'rec_kron' | 'mask': {'finite': ['org_kron', 'rec_kron', 'org_mag'], 'missing': ['noise_kron']}, |
| python | src/visualization/prepare_plots.py | 744 | 'org_mag' | 'mask': {'finite': ['org_kron', 'rec_kron', 'org_mag'], 'missing': ['noise_kron']}, |
| python | src/visualization/prepare_plots.py | 744 | 'noise_kron' | 'mask': {'finite': ['org_kron', 'rec_kron', 'org_mag'], 'missing': ['noise_kron']}, |
| python | src/visualization/prepare_plots.py | 745 | 'org_mag' | 'x': {'key': 'org_mag'}, |
| python | src/visualization/prepare_plots.py | 746 | 'rec_kron' | 'y': {'key': 'rec_kron', 'transform': 'delta', 'reference': 'org_kron'}, |
| python | src/visualization/prepare_plots.py | 746 | 'org_kron' | 'y': {'key': 'rec_kron', 'transform': 'delta', 'reference': 'org_kron'}, |
| python | src/visualization/prepare_plots.py | 747 | 'norm_mode' | 'norm_mode': 'dynamic', |
| python | src/visualization/prepare_plots.py | 748 | 'legend_stat' | 'legend_stat': 'min', |
| python | src/visualization/prepare_plots.py | 749 | 'xlim_floor' | 'xlim_floor': 22.7, |
| python | src/visualization/prepare_plots.py | 749 | 22.7 | 'xlim_floor': 22.7, |
| python | src/visualization/prepare_plots.py | 755 | 'output_filename' | 'output_filename': 'flux_ratio_mag_rec.png', |
| python | src/visualization/prepare_plots.py | 755 | 'flux_ratio_mag_rec.png' | 'output_filename': 'flux_ratio_mag_rec.png', |
| python | src/visualization/prepare_plots.py | 757 | 'rec_cmap' | 'rec_cmap': rec_cmap, |
| python | src/visualization/prepare_plots.py | 758 | 'org_flux' | 'mask': {'positive': ['org_flux'], 'finite': ['rec_flux', 'org_mag'], 'missing': ['noise_flux']}, |
| python | src/visualization/prepare_plots.py | 758 | 'rec_flux' | 'mask': {'positive': ['org_flux'], 'finite': ['rec_flux', 'org_mag'], 'missing': ['noise_flux']}, |
| python | src/visualization/prepare_plots.py | 758 | 'org_mag' | 'mask': {'positive': ['org_flux'], 'finite': ['rec_flux', 'org_mag'], 'missing': ['noise_flux']}, |
| python | src/visualization/prepare_plots.py | 758 | 'noise_flux' | 'mask': {'positive': ['org_flux'], 'finite': ['rec_flux', 'org_mag'], 'missing': ['noise_flux']}, |
| python | src/visualization/prepare_plots.py | 759 | 'org_mag' | 'x': {'key': 'org_mag'}, |
| python | src/visualization/prepare_plots.py | 760 | 'rec_flux' | 'y': {'key': 'rec_flux', 'transform': 'ratio', 'reference': 'org_flux'}, |
| python | src/visualization/prepare_plots.py | 760 | 'org_flux' | 'y': {'key': 'rec_flux', 'transform': 'ratio', 'reference': 'org_flux'}, |
| python | src/visualization/prepare_plots.py | 761 | 'norm_mode' | 'norm_mode': 'dynamic', |
| python | src/visualization/prepare_plots.py | 762 | 'legend_stat' | 'legend_stat': 'min', |
| python | src/visualization/prepare_plots.py | 763 | 'xlim_floor' | 'xlim_floor': 22.7, |
| python | src/visualization/prepare_plots.py | 763 | 22.7 | 'xlim_floor': 22.7, |
| python | src/visualization/prepare_plots.py | 769 | 'output_filename' | 'output_filename': 'delta_kron_rec.png', |
| python | src/visualization/prepare_plots.py | 769 | 'delta_kron_rec.png' | 'output_filename': 'delta_kron_rec.png', |
| python | src/visualization/prepare_plots.py | 770 | 'Rec. $\\Delta$Kron Radius vs. Org. Kron Radius (Detected only in Rec.)' | 'suptitle': r"Rec. $\Delta$Kron Radius vs. Org. Kron Radius (Detected only in Rec.)", |
| python | src/visualization/prepare_plots.py | 771 | 'rec_cmap' | 'rec_cmap': rec_cmap, |
| python | src/visualization/prepare_plots.py | 772 | 'org_kron' | 'mask': {'finite': ['org_kron', 'rec_kron'], 'missing': ['noise_kron']}, |
| python | src/visualization/prepare_plots.py | 772 | 'rec_kron' | 'mask': {'finite': ['org_kron', 'rec_kron'], 'missing': ['noise_kron']}, |
| python | src/visualization/prepare_plots.py | 772 | 'noise_kron' | 'mask': {'finite': ['org_kron', 'rec_kron'], 'missing': ['noise_kron']}, |
| python | src/visualization/prepare_plots.py | 773 | 'org_kron' | 'x': {'key': 'org_kron'}, |
| python | src/visualization/prepare_plots.py | 774 | 'rec_kron' | 'y': {'key': 'rec_kron', 'transform': 'delta', 'reference': 'org_kron'}, |
| python | src/visualization/prepare_plots.py | 774 | 'org_kron' | 'y': {'key': 'rec_kron', 'transform': 'delta', 'reference': 'org_kron'}, |
| python | src/visualization/prepare_plots.py | 775 | 'norm_mode' | 'norm_mode': 'dynamic', |
| python | src/visualization/prepare_plots.py | 776 | 'legend_stat' | 'legend_stat': 'min', |
| python | src/visualization/prepare_plots.py | 782 | 'output_filename' | 'output_filename': 'flux_ratio_rec.png', |
| python | src/visualization/prepare_plots.py | 782 | 'flux_ratio_rec.png' | 'output_filename': 'flux_ratio_rec.png', |
| python | src/visualization/prepare_plots.py | 783 | 'Rec.Flux Ratio vs. Org. $\\log($Flux) (Detected only in Rec.)' | 'suptitle': r"Rec.Flux Ratio vs. Org. $\log($Flux) (Detected only in Rec.)", |
| python | src/visualization/prepare_plots.py | 784 | 'rec_cmap' | 'rec_cmap': rec_cmap, |
| python | src/visualization/prepare_plots.py | 785 | 'org_flux' | 'mask': {'positive': ['org_flux'], 'finite': ['rec_flux'], 'missing': ['noise_flux']}, |
| python | src/visualization/prepare_plots.py | 785 | 'rec_flux' | 'mask': {'positive': ['org_flux'], 'finite': ['rec_flux'], 'missing': ['noise_flux']}, |
| python | src/visualization/prepare_plots.py | 785 | 'noise_flux' | 'mask': {'positive': ['org_flux'], 'finite': ['rec_flux'], 'missing': ['noise_flux']}, |
| python | src/visualization/prepare_plots.py | 786 | 'org_flux' | 'x': {'key': 'org_flux', 'transform': 'log10'}, |
| python | src/visualization/prepare_plots.py | 787 | 'rec_flux' | 'y': {'key': 'rec_flux', 'transform': 'ratio', 'reference': 'org_flux'}, |
| python | src/visualization/prepare_plots.py | 787 | 'org_flux' | 'y': {'key': 'rec_flux', 'transform': 'ratio', 'reference': 'org_flux'}, |
| python | src/visualization/prepare_plots.py | 788 | 'norm_mode' | 'norm_mode': 'column', |
| python | src/visualization/prepare_plots.py | 789 | 'norm_key' | 'norm_key': 'org_flux', |
| python | src/visualization/prepare_plots.py | 789 | 'org_flux' | 'norm_key': 'org_flux', |
| python | src/visualization/prepare_plots.py | 790 | 'legend_stat' | 'legend_stat': 'min', |
| python | src/visualization/prepare_plots.py | 796 | 'output_filename' | 'output_filename': 'kron_combined.png', |
| python | src/visualization/prepare_plots.py | 796 | 'kron_combined.png' | 'output_filename': 'kron_combined.png', |
| python | src/visualization/prepare_plots.py | 798 | 'rec_cmap' | 'rec_cmap': rec_cmap, |
| python | src/visualization/prepare_plots.py | 799 | 'noise_cmap' | 'noise_cmap': noise_cmap, |
| python | src/visualization/prepare_plots.py | 800 | 'pre_window' | 'pre_window': {'key': 'org_kron', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 800 | 'org_kron' | 'pre_window': {'key': 'org_kron', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 800 | 98 | 'pre_window': {'key': 'org_kron', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 801 | 'mask_mode' | 'mask_mode': 'separate', |
| python | src/visualization/prepare_plots.py | 802 | 'rec_mask' | 'rec_mask': {'finite': ['org_kron', 'rec_kron']}, |
| python | src/visualization/prepare_plots.py | 802 | 'org_kron' | 'rec_mask': {'finite': ['org_kron', 'rec_kron']}, |
| python | src/visualization/prepare_plots.py | 802 | 'rec_kron' | 'rec_mask': {'finite': ['org_kron', 'rec_kron']}, |
| python | src/visualization/prepare_plots.py | 803 | 'noise_mask' | 'noise_mask': {'finite': ['org_kron', 'noise_kron']}, |
| python | src/visualization/prepare_plots.py | 803 | 'org_kron' | 'noise_mask': {'finite': ['org_kron', 'noise_kron']}, |
| python | src/visualization/prepare_plots.py | 803 | 'noise_kron' | 'noise_mask': {'finite': ['org_kron', 'noise_kron']}, |
| python | src/visualization/prepare_plots.py | 804 | 'org_kron' | 'rec': {'x': {'key': 'org_kron'}, 'y': {'key': 'rec_kron'}}, |
| python | src/visualization/prepare_plots.py | 804 | 'rec_kron' | 'rec': {'x': {'key': 'org_kron'}, 'y': {'key': 'rec_kron'}}, |
| python | src/visualization/prepare_plots.py | 805 | 'org_kron' | 'noise': {'x': {'key': 'org_kron'}, 'y': {'key': 'noise_kron'}}, |
| python | src/visualization/prepare_plots.py | 805 | 'noise_kron' | 'noise': {'x': {'key': 'org_kron'}, 'y': {'key': 'noise_kron'}}, |
| python | src/visualization/prepare_plots.py | 806 | 'clip_mode' | 'clip_mode': 'separate', |
| python | src/visualization/prepare_plots.py | 807 | 'norm_mode' | 'norm_mode': 'column', |
| python | src/visualization/prepare_plots.py | 808 | 'norm_key' | 'norm_key': 'org_kron', |
| python | src/visualization/prepare_plots.py | 808 | 'org_kron' | 'norm_key': 'org_kron', |
| python | src/visualization/prepare_plots.py | 809 | 'legend_stat' | 'legend_stat': 'min', |
| python | src/visualization/prepare_plots.py | 810 | 'xlabel_rec' | 'xlabel_rec': kron_label, |
| python | src/visualization/prepare_plots.py | 811 | 'xlabel_noise' | 'xlabel_noise': arcsec_label, |
| python | src/visualization/prepare_plots.py | 812 | 'ylabel_rec' | 'ylabel_rec': r'Kron Radius $[arcsec]$', |
| python | src/visualization/prepare_plots.py | 813 | 'one_to_one' | 'one_to_one': True, |
| python | src/visualization/prepare_plots.py | 817 | 'output_filename' | 'output_filename': 'flux_combined.png', |
| python | src/visualization/prepare_plots.py | 817 | 'flux_combined.png' | 'output_filename': 'flux_combined.png', |
| python | src/visualization/prepare_plots.py | 818 | 'Rec., Noisy $\\log($Flux) vs. Org. $\\log($Flux)' | 'suptitle': r"Rec., Noisy $\log($Flux) vs. Org. $\log($Flux)", |
| python | src/visualization/prepare_plots.py | 819 | 'rec_cmap' | 'rec_cmap': rec_cmap, |
| python | src/visualization/prepare_plots.py | 820 | 'noise_cmap' | 'noise_cmap': noise_cmap, |
| python | src/visualization/prepare_plots.py | 821 | 'pre_window' | 'pre_window': {'key': 'org_flux', 'positive': True, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 821 | 'org_flux' | 'pre_window': {'key': 'org_flux', 'positive': True, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 821 | 98 | 'pre_window': {'key': 'org_flux', 'positive': True, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 822 | 'mask_mode' | 'mask_mode': 'separate', |
| python | src/visualization/prepare_plots.py | 823 | 'rec_mask' | 'rec_mask': {'positive': ['org_flux', 'rec_flux']}, |
| python | src/visualization/prepare_plots.py | 823 | 'org_flux' | 'rec_mask': {'positive': ['org_flux', 'rec_flux']}, |
| python | src/visualization/prepare_plots.py | 823 | 'rec_flux' | 'rec_mask': {'positive': ['org_flux', 'rec_flux']}, |
| python | src/visualization/prepare_plots.py | 824 | 'noise_mask' | 'noise_mask': {'positive': ['org_flux', 'noise_flux']}, |
| python | src/visualization/prepare_plots.py | 824 | 'org_flux' | 'noise_mask': {'positive': ['org_flux', 'noise_flux']}, |
| python | src/visualization/prepare_plots.py | 824 | 'noise_flux' | 'noise_mask': {'positive': ['org_flux', 'noise_flux']}, |
| python | src/visualization/prepare_plots.py | 825 | 'org_flux' | 'rec': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'rec_flux', 'transform': 'log10'}}, |
| python | src/visualization/prepare_plots.py | 825 | 'rec_flux' | 'rec': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'rec_flux', 'transform': 'log10'}}, |
| python | src/visualization/prepare_plots.py | 826 | 'org_flux' | 'noise': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'noise_flux', 'transform': 'log10'}}, |
| python | src/visualization/prepare_plots.py | 826 | 'noise_flux' | 'noise': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'noise_flux', 'transform': 'log10'}}, |
| python | src/visualization/prepare_plots.py | 827 | 'clip_mode' | 'clip_mode': 'separate', |
| python | src/visualization/prepare_plots.py | 828 | 'norm_mode' | 'norm_mode': 'column', |
| python | src/visualization/prepare_plots.py | 829 | 'norm_key' | 'norm_key': 'org_flux', |
| python | src/visualization/prepare_plots.py | 829 | 'org_flux' | 'norm_key': 'org_flux', |
| python | src/visualization/prepare_plots.py | 830 | 'legend_stat' | 'legend_stat': 'median', |
| python | src/visualization/prepare_plots.py | 831 | 'xlabel_rec' | 'xlabel_rec': flux_label, |
| python | src/visualization/prepare_plots.py | 832 | 'xlabel_noise' | 'xlabel_noise': flux_label, |
| python | src/visualization/prepare_plots.py | 833 | 'ylabel_rec' | 'ylabel_rec': rec_flux_label, |
| python | src/visualization/prepare_plots.py | 834 | 'ylabel_noise' | 'ylabel_noise': noisy_flux_label, |
| python | src/visualization/prepare_plots.py | 835 | 'one_to_one' | 'one_to_one': True, |
| python | src/visualization/prepare_plots.py | 839 | 'output_filename' | 'output_filename': 'kron_combined_both.png', |
| python | src/visualization/prepare_plots.py | 839 | 'kron_combined_both.png' | 'output_filename': 'kron_combined_both.png', |
| python | src/visualization/prepare_plots.py | 841 | 'rec_cmap' | 'rec_cmap': rec_cmap, |
| python | src/visualization/prepare_plots.py | 842 | 'noise_cmap' | 'noise_cmap': noise_cmap, |
| python | src/visualization/prepare_plots.py | 843 | 'pre_window' | 'pre_window': {'key': 'org_kron', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 843 | 'org_kron' | 'pre_window': {'key': 'org_kron', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 843 | 98 | 'pre_window': {'key': 'org_kron', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 844 | 'mask_mode' | 'mask_mode': 'shared', |
| python | src/visualization/prepare_plots.py | 845 | 'org_kron' | 'mask': {'finite': ['org_kron', 'rec_kron', 'noise_kron']}, |
| python | src/visualization/prepare_plots.py | 845 | 'rec_kron' | 'mask': {'finite': ['org_kron', 'rec_kron', 'noise_kron']}, |
| python | src/visualization/prepare_plots.py | 845 | 'noise_kron' | 'mask': {'finite': ['org_kron', 'rec_kron', 'noise_kron']}, |
| python | src/visualization/prepare_plots.py | 846 | 'org_kron' | 'rec': {'x': {'key': 'org_kron'}, 'y': {'key': 'rec_kron'}}, |
| python | src/visualization/prepare_plots.py | 846 | 'rec_kron' | 'rec': {'x': {'key': 'org_kron'}, 'y': {'key': 'rec_kron'}}, |
| python | src/visualization/prepare_plots.py | 847 | 'org_kron' | 'noise': {'x': {'key': 'org_kron'}, 'y': {'key': 'noise_kron'}}, |
| python | src/visualization/prepare_plots.py | 847 | 'noise_kron' | 'noise': {'x': {'key': 'org_kron'}, 'y': {'key': 'noise_kron'}}, |
| python | src/visualization/prepare_plots.py | 848 | 'clip_mode' | 'clip_mode': 'aligned', |
| python | src/visualization/prepare_plots.py | 849 | 'norm_mode' | 'norm_mode': 'column', |
| python | src/visualization/prepare_plots.py | 850 | 'norm_key' | 'norm_key': 'org_kron', |
| python | src/visualization/prepare_plots.py | 850 | 'org_kron' | 'norm_key': 'org_kron', |
| python | src/visualization/prepare_plots.py | 851 | 'legend_stat' | 'legend_stat': 'min', |
| python | src/visualization/prepare_plots.py | 852 | 'xlabel_rec' | 'xlabel_rec': kron_label, |
| python | src/visualization/prepare_plots.py | 853 | 'xlabel_noise' | 'xlabel_noise': arcsec_label, |
| python | src/visualization/prepare_plots.py | 854 | 'ylabel_rec' | 'ylabel_rec': r'Kron Radius $[arcsec]$', |
| python | src/visualization/prepare_plots.py | 855 | 'one_to_one' | 'one_to_one': True, |
| python | src/visualization/prepare_plots.py | 859 | 'output_filename' | 'output_filename': 'flux_combined_both.png', |
| python | src/visualization/prepare_plots.py | 859 | 'flux_combined_both.png' | 'output_filename': 'flux_combined_both.png', |
| python | src/visualization/prepare_plots.py | 860 | 'Rec., Noisy $\\log($Flux) vs. Org. $\\log($Flux) (Detected in Both)' | 'suptitle': r"Rec., Noisy $\log($Flux) vs. Org. $\log($Flux) (Detected in Both)", |
| python | src/visualization/prepare_plots.py | 861 | 'rec_cmap' | 'rec_cmap': rec_cmap, |
| python | src/visualization/prepare_plots.py | 862 | 'noise_cmap' | 'noise_cmap': noise_cmap, |
| python | src/visualization/prepare_plots.py | 863 | 'pre_window' | 'pre_window': {'key': 'org_flux', 'positive': True, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 863 | 'org_flux' | 'pre_window': {'key': 'org_flux', 'positive': True, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 863 | 98 | 'pre_window': {'key': 'org_flux', 'positive': True, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 864 | 'mask_mode' | 'mask_mode': 'shared', |
| python | src/visualization/prepare_plots.py | 865 | 'org_flux' | 'mask': {'positive': ['org_flux', 'rec_flux', 'noise_flux']}, |
| python | src/visualization/prepare_plots.py | 865 | 'rec_flux' | 'mask': {'positive': ['org_flux', 'rec_flux', 'noise_flux']}, |
| python | src/visualization/prepare_plots.py | 865 | 'noise_flux' | 'mask': {'positive': ['org_flux', 'rec_flux', 'noise_flux']}, |
| python | src/visualization/prepare_plots.py | 866 | 'org_flux' | 'rec': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'rec_flux', 'transform': 'log10'}}, |
| python | src/visualization/prepare_plots.py | 866 | 'rec_flux' | 'rec': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'rec_flux', 'transform': 'log10'}}, |
| python | src/visualization/prepare_plots.py | 867 | 'org_flux' | 'noise': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'noise_flux', 'transform': 'log10'}}, |
| python | src/visualization/prepare_plots.py | 867 | 'noise_flux' | 'noise': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'noise_flux', 'transform': 'log10'}}, |
| python | src/visualization/prepare_plots.py | 868 | 'clip_mode' | 'clip_mode': 'aligned', |
| python | src/visualization/prepare_plots.py | 869 | 'norm_mode' | 'norm_mode': 'column', |
| python | src/visualization/prepare_plots.py | 870 | 'norm_key' | 'norm_key': 'org_flux', |
| python | src/visualization/prepare_plots.py | 870 | 'org_flux' | 'norm_key': 'org_flux', |
| python | src/visualization/prepare_plots.py | 871 | 'legend_stat' | 'legend_stat': 'median', |
| python | src/visualization/prepare_plots.py | 872 | 'xlabel_rec' | 'xlabel_rec': flux_label, |
| python | src/visualization/prepare_plots.py | 873 | 'xlabel_noise' | 'xlabel_noise': flux_label, |
| python | src/visualization/prepare_plots.py | 874 | 'ylabel_rec' | 'ylabel_rec': rec_flux_label, |
| python | src/visualization/prepare_plots.py | 875 | 'ylabel_noise' | 'ylabel_noise': noisy_flux_label, |
| python | src/visualization/prepare_plots.py | 876 | 'one_to_one' | 'one_to_one': True, |
| python | src/visualization/prepare_plots.py | 880 | 'output_filename' | 'output_filename': 'flux_rec.png', |
| python | src/visualization/prepare_plots.py | 880 | 'flux_rec.png' | 'output_filename': 'flux_rec.png', |
| python | src/visualization/prepare_plots.py | 881 | 'Rec. $\\log($Flux) vs. Org. $\\log($Flux) (Detected only in Rec.)' | 'suptitle': r"Rec. $\log($Flux) vs. Org. $\log($Flux) (Detected only in Rec.)", |
| python | src/visualization/prepare_plots.py | 882 | 'rec_cmap' | 'rec_cmap': rec_cmap, |
| python | src/visualization/prepare_plots.py | 883 | 'org_flux' | 'mask': {'finite': ['org_flux', 'rec_flux'], 'missing': ['noise_flux']}, |
| python | src/visualization/prepare_plots.py | 883 | 'rec_flux' | 'mask': {'finite': ['org_flux', 'rec_flux'], 'missing': ['noise_flux']}, |
| python | src/visualization/prepare_plots.py | 883 | 'noise_flux' | 'mask': {'finite': ['org_flux', 'rec_flux'], 'missing': ['noise_flux']}, |
| python | src/visualization/prepare_plots.py | 884 | 'org_flux' | 'x': {'key': 'org_flux'}, |
| python | src/visualization/prepare_plots.py | 885 | 'rec_flux' | 'y': {'key': 'rec_flux'}, |
| python | src/visualization/prepare_plots.py | 886 | 'norm_mode' | 'norm_mode': 'column', |
| python | src/visualization/prepare_plots.py | 887 | 'norm_key' | 'norm_key': 'org_flux', |
| python | src/visualization/prepare_plots.py | 887 | 'org_flux' | 'norm_key': 'org_flux', |
| python | src/visualization/prepare_plots.py | 888 | 'legend_stat' | 'legend_stat': 'min', |
| python | src/visualization/prepare_plots.py | 891 | 'one_to_one' | 'one_to_one': True, |
| python | src/visualization/prepare_plots.py | 895 | 'output_filename' | 'output_filename': 'kron_rec.png', |
| python | src/visualization/prepare_plots.py | 895 | 'kron_rec.png' | 'output_filename': 'kron_rec.png', |
| python | src/visualization/prepare_plots.py | 897 | 'rec_cmap' | 'rec_cmap': rec_cmap, |
| python | src/visualization/prepare_plots.py | 898 | 'org_kron' | 'mask': {'finite': ['org_kron', 'rec_kron'], 'missing': ['noise_kron']}, |
| python | src/visualization/prepare_plots.py | 898 | 'rec_kron' | 'mask': {'finite': ['org_kron', 'rec_kron'], 'missing': ['noise_kron']}, |
| python | src/visualization/prepare_plots.py | 898 | 'noise_kron' | 'mask': {'finite': ['org_kron', 'rec_kron'], 'missing': ['noise_kron']}, |
| python | src/visualization/prepare_plots.py | 899 | 'org_kron' | 'x': {'key': 'org_kron'}, |
| python | src/visualization/prepare_plots.py | 900 | 'rec_kron' | 'y': {'key': 'rec_kron'}, |
| python | src/visualization/prepare_plots.py | 901 | 'y_clip' | 'y_clip': (1, 98), |
| python | src/visualization/prepare_plots.py | 901 | 98 | 'y_clip': (1, 98), |
| python | src/visualization/prepare_plots.py | 902 | 'norm_mode' | 'norm_mode': 'dynamic', |
| python | src/visualization/prepare_plots.py | 903 | 'legend_stat' | 'legend_stat': 'min', |
| python | src/visualization/prepare_plots.py | 906 | 'one_to_one' | 'one_to_one': True, |
| python | src/visualization/prepare_plots.py | 910 | 'output_filename' | 'output_filename': 'flux_ratio_mag_combined.png', |
| python | src/visualization/prepare_plots.py | 910 | 'flux_ratio_mag_combined.png' | 'output_filename': 'flux_ratio_mag_combined.png', |
| python | src/visualization/prepare_plots.py | 912 | 'rec_cmap' | 'rec_cmap': rec_cmap, |
| python | src/visualization/prepare_plots.py | 913 | 'noise_cmap' | 'noise_cmap': noise_cmap, |
| python | src/visualization/prepare_plots.py | 914 | 'pre_window' | 'pre_window': {'key': 'org_mag', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 914 | 'org_mag' | 'pre_window': {'key': 'org_mag', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 914 | 98 | 'pre_window': {'key': 'org_mag', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 915 | 'mask_mode' | 'mask_mode': 'separate', |
| python | src/visualization/prepare_plots.py | 916 | 'rec_mask' | 'rec_mask': {'positive': ['org_flux'], 'finite': ['rec_flux', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 916 | 'org_flux' | 'rec_mask': {'positive': ['org_flux'], 'finite': ['rec_flux', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 916 | 'rec_flux' | 'rec_mask': {'positive': ['org_flux'], 'finite': ['rec_flux', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 916 | 'org_mag' | 'rec_mask': {'positive': ['org_flux'], 'finite': ['rec_flux', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 917 | 'noise_mask' | 'noise_mask': {'positive': ['org_flux'], 'finite': ['noise_flux', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 917 | 'org_flux' | 'noise_mask': {'positive': ['org_flux'], 'finite': ['noise_flux', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 917 | 'noise_flux' | 'noise_mask': {'positive': ['org_flux'], 'finite': ['noise_flux', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 917 | 'org_mag' | 'noise_mask': {'positive': ['org_flux'], 'finite': ['noise_flux', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 918 | 'org_mag' | 'rec': {'x': {'key': 'org_mag'}, 'y': {'key': 'rec_flux', 'transform': 'ratio', 'reference': 'org_flux'}}, |
| python | src/visualization/prepare_plots.py | 918 | 'rec_flux' | 'rec': {'x': {'key': 'org_mag'}, 'y': {'key': 'rec_flux', 'transform': 'ratio', 'reference': 'org_flux'}}, |
| python | src/visualization/prepare_plots.py | 918 | 'org_flux' | 'rec': {'x': {'key': 'org_mag'}, 'y': {'key': 'rec_flux', 'transform': 'ratio', 'reference': 'org_flux'}}, |
| python | src/visualization/prepare_plots.py | 919 | 'org_mag' | 'noise': {'x': {'key': 'org_mag'}, 'y': {'key': 'noise_flux', 'transform': 'ratio', 'reference': 'org_flux'}}, |
| python | src/visualization/prepare_plots.py | 919 | 'noise_flux' | 'noise': {'x': {'key': 'org_mag'}, 'y': {'key': 'noise_flux', 'transform': 'ratio', 'reference': 'org_flux'}}, |
| python | src/visualization/prepare_plots.py | 919 | 'org_flux' | 'noise': {'x': {'key': 'org_mag'}, 'y': {'key': 'noise_flux', 'transform': 'ratio', 'reference': 'org_flux'}}, |
| python | src/visualization/prepare_plots.py | 920 | 'clip_mode' | 'clip_mode': 'separate', |
| python | src/visualization/prepare_plots.py | 921 | 'norm_mode' | 'norm_mode': 'column', |
| python | src/visualization/prepare_plots.py | 922 | 'norm_key' | 'norm_key': 'org_flux', |
| python | src/visualization/prepare_plots.py | 922 | 'org_flux' | 'norm_key': 'org_flux', |
| python | src/visualization/prepare_plots.py | 923 | 'legend_stat' | 'legend_stat': 'median', |
| python | src/visualization/prepare_plots.py | 924 | 'xlabel_rec' | 'xlabel_rec': abmag_label, |
| python | src/visualization/prepare_plots.py | 925 | 'xlabel_noise' | 'xlabel_noise': abmag_label, |
| python | src/visualization/prepare_plots.py | 926 | 'ylabel_rec' | 'ylabel_rec': r'Flux Ratio', |
| python | src/visualization/prepare_plots.py | 930 | 'output_filename' | 'output_filename': 'delta_kron_mag_combined.png', |
| python | src/visualization/prepare_plots.py | 930 | 'delta_kron_mag_combined.png' | 'output_filename': 'delta_kron_mag_combined.png', |
| python | src/visualization/prepare_plots.py | 931 | 'Rec., Noisy $\\Delta$Kron Radius vs. AB MAG' | 'suptitle': r"Rec., Noisy $\Delta$Kron Radius vs. AB MAG", |
| python | src/visualization/prepare_plots.py | 932 | 'rec_cmap' | 'rec_cmap': rec_cmap, |
| python | src/visualization/prepare_plots.py | 933 | 'noise_cmap' | 'noise_cmap': noise_cmap, |
| python | src/visualization/prepare_plots.py | 934 | 'pre_window' | 'pre_window': {'key': 'org_mag', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 934 | 'org_mag' | 'pre_window': {'key': 'org_mag', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 934 | 98 | 'pre_window': {'key': 'org_mag', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 935 | 'mask_mode' | 'mask_mode': 'separate', |
| python | src/visualization/prepare_plots.py | 936 | 'rec_mask' | 'rec_mask': {'finite': ['org_kron', 'rec_kron', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 936 | 'org_kron' | 'rec_mask': {'finite': ['org_kron', 'rec_kron', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 936 | 'rec_kron' | 'rec_mask': {'finite': ['org_kron', 'rec_kron', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 936 | 'org_mag' | 'rec_mask': {'finite': ['org_kron', 'rec_kron', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 937 | 'noise_mask' | 'noise_mask': {'finite': ['org_kron', 'noise_kron', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 937 | 'org_kron' | 'noise_mask': {'finite': ['org_kron', 'noise_kron', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 937 | 'noise_kron' | 'noise_mask': {'finite': ['org_kron', 'noise_kron', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 937 | 'org_mag' | 'noise_mask': {'finite': ['org_kron', 'noise_kron', 'org_mag']}, |
| python | src/visualization/prepare_plots.py | 938 | 'org_mag' | 'rec': {'x': {'key': 'org_mag'}, 'y': {'key': 'rec_kron', 'transform': 'delta', 'reference': 'org_kron'}}, |
| python | src/visualization/prepare_plots.py | 938 | 'rec_kron' | 'rec': {'x': {'key': 'org_mag'}, 'y': {'key': 'rec_kron', 'transform': 'delta', 'reference': 'org_kron'}}, |
| python | src/visualization/prepare_plots.py | 938 | 'org_kron' | 'rec': {'x': {'key': 'org_mag'}, 'y': {'key': 'rec_kron', 'transform': 'delta', 'reference': 'org_kron'}}, |
| python | src/visualization/prepare_plots.py | 939 | 'org_mag' | 'noise': {'x': {'key': 'org_mag'}, 'y': {'key': 'noise_kron', 'transform': 'delta', 'reference': 'org_kron'}}, |
| python | src/visualization/prepare_plots.py | 939 | 'noise_kron' | 'noise': {'x': {'key': 'org_mag'}, 'y': {'key': 'noise_kron', 'transform': 'delta', 'reference': 'org_kron'}}, |
| python | src/visualization/prepare_plots.py | 939 | 'org_kron' | 'noise': {'x': {'key': 'org_mag'}, 'y': {'key': 'noise_kron', 'transform': 'delta', 'reference': 'org_kron'}}, |
| python | src/visualization/prepare_plots.py | 940 | 'clip_mode' | 'clip_mode': 'separate', |
| python | src/visualization/prepare_plots.py | 941 | 'norm_mode' | 'norm_mode': 'column', |
| python | src/visualization/prepare_plots.py | 942 | 'norm_key' | 'norm_key': 'org_kron', |
| python | src/visualization/prepare_plots.py | 942 | 'org_kron' | 'norm_key': 'org_kron', |
| python | src/visualization/prepare_plots.py | 943 | 'legend_stat' | 'legend_stat': 'min', |
| python | src/visualization/prepare_plots.py | 944 | 'xlabel_rec' | 'xlabel_rec': abmag_label, |
| python | src/visualization/prepare_plots.py | 945 | 'xlabel_noise' | 'xlabel_noise': abmag_label, |
| python | src/visualization/prepare_plots.py | 946 | 'ylabel_rec' | 'ylabel_rec': delta_kron_label, |
| python | src/visualization/prepare_plots.py | 950 | 'output_filename' | 'output_filename': 'flux_ratio_combined.png', |
| python | src/visualization/prepare_plots.py | 950 | 'flux_ratio_combined.png' | 'output_filename': 'flux_ratio_combined.png', |
| python | src/visualization/prepare_plots.py | 951 | 'Flux Ratio vs. $\\log$(Flux)' | 'suptitle': r"Flux Ratio vs. $\log$(Flux)", |
| python | src/visualization/prepare_plots.py | 952 | 'rec_cmap' | 'rec_cmap': rec_cmap, |
| python | src/visualization/prepare_plots.py | 953 | 'noise_cmap' | 'noise_cmap': noise_cmap, |
| python | src/visualization/prepare_plots.py | 954 | 'pre_window' | 'pre_window': {'key': 'org_flux', 'positive': True, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 954 | 'org_flux' | 'pre_window': {'key': 'org_flux', 'positive': True, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 954 | 98 | 'pre_window': {'key': 'org_flux', 'positive': True, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 955 | 'mask_mode' | 'mask_mode': 'separate', |
| python | src/visualization/prepare_plots.py | 956 | 'rec_mask' | 'rec_mask': {'positive': ['org_flux'], 'finite': ['rec_flux']}, |
| python | src/visualization/prepare_plots.py | 956 | 'org_flux' | 'rec_mask': {'positive': ['org_flux'], 'finite': ['rec_flux']}, |
| python | src/visualization/prepare_plots.py | 956 | 'rec_flux' | 'rec_mask': {'positive': ['org_flux'], 'finite': ['rec_flux']}, |
| python | src/visualization/prepare_plots.py | 957 | 'noise_mask' | 'noise_mask': {'positive': ['org_flux'], 'finite': ['noise_flux']}, |
| python | src/visualization/prepare_plots.py | 957 | 'org_flux' | 'noise_mask': {'positive': ['org_flux'], 'finite': ['noise_flux']}, |
| python | src/visualization/prepare_plots.py | 957 | 'noise_flux' | 'noise_mask': {'positive': ['org_flux'], 'finite': ['noise_flux']}, |
| python | src/visualization/prepare_plots.py | 958 | 'org_flux' | 'rec': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'rec_flux', 'transform': 'ratio', 'reference': 'org_flux'}}, |
| python | src/visualization/prepare_plots.py | 958 | 'rec_flux' | 'rec': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'rec_flux', 'transform': 'ratio', 'reference': 'org_flux'}}, |
| python | src/visualization/prepare_plots.py | 959 | 'org_flux' | 'noise': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'noise_flux', 'transform': 'ratio', 'reference': 'org_flux'}}, |
| python | src/visualization/prepare_plots.py | 959 | 'noise_flux' | 'noise': {'x': {'key': 'org_flux', 'transform': 'log10'}, 'y': {'key': 'noise_flux', 'transform': 'ratio', 'reference': 'org_flux'}}, |
| python | src/visualization/prepare_plots.py | 960 | 'clip_mode' | 'clip_mode': 'separate', |
| python | src/visualization/prepare_plots.py | 961 | 'norm_mode' | 'norm_mode': 'column', |
| python | src/visualization/prepare_plots.py | 962 | 'norm_key' | 'norm_key': 'org_flux', |
| python | src/visualization/prepare_plots.py | 962 | 'org_flux' | 'norm_key': 'org_flux', |
| python | src/visualization/prepare_plots.py | 963 | 'legend_stat' | 'legend_stat': 'median', |
| python | src/visualization/prepare_plots.py | 964 | 'xlabel_rec' | 'xlabel_rec': flux_label, |
| python | src/visualization/prepare_plots.py | 965 | 'xlabel_noise' | 'xlabel_noise': flux_label, |
| python | src/visualization/prepare_plots.py | 966 | 'ylabel_rec' | 'ylabel_rec': r'Flux Ratio', |
| python | src/visualization/prepare_plots.py | 970 | 'output_filename' | 'output_filename': 'delta_kron_combined.png', |
| python | src/visualization/prepare_plots.py | 970 | 'delta_kron_combined.png' | 'output_filename': 'delta_kron_combined.png', |
| python | src/visualization/prepare_plots.py | 971 | 'Rec., Noisy $\\Delta$Kron Radius vs. Org. Kron Radius' | 'suptitle': r"Rec., Noisy $\Delta$Kron Radius vs. Org. Kron Radius", |
| python | src/visualization/prepare_plots.py | 972 | 'rec_cmap' | 'rec_cmap': rec_cmap, |
| python | src/visualization/prepare_plots.py | 973 | 'noise_cmap' | 'noise_cmap': noise_cmap, |
| python | src/visualization/prepare_plots.py | 974 | 'pre_window' | 'pre_window': {'key': 'org_kron', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 974 | 'org_kron' | 'pre_window': {'key': 'org_kron', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 974 | 98 | 'pre_window': {'key': 'org_kron', 'positive': False, 'percentiles': (1, 98)}, |
| python | src/visualization/prepare_plots.py | 975 | 'mask_mode' | 'mask_mode': 'separate', |
| python | src/visualization/prepare_plots.py | 976 | 'rec_mask' | 'rec_mask': {'finite': ['org_kron', 'rec_kron']}, |
| python | src/visualization/prepare_plots.py | 976 | 'org_kron' | 'rec_mask': {'finite': ['org_kron', 'rec_kron']}, |
| python | src/visualization/prepare_plots.py | 976 | 'rec_kron' | 'rec_mask': {'finite': ['org_kron', 'rec_kron']}, |
| python | src/visualization/prepare_plots.py | 977 | 'noise_mask' | 'noise_mask': {'finite': ['org_kron', 'noise_kron']}, |
| python | src/visualization/prepare_plots.py | 977 | 'org_kron' | 'noise_mask': {'finite': ['org_kron', 'noise_kron']}, |
| python | src/visualization/prepare_plots.py | 977 | 'noise_kron' | 'noise_mask': {'finite': ['org_kron', 'noise_kron']}, |
| python | src/visualization/prepare_plots.py | 978 | 'org_kron' | 'rec': {'x': {'key': 'org_kron'}, 'y': {'key': 'rec_kron', 'transform': 'delta', 'reference': 'org_kron'}}, |
| python | src/visualization/prepare_plots.py | 978 | 'rec_kron' | 'rec': {'x': {'key': 'org_kron'}, 'y': {'key': 'rec_kron', 'transform': 'delta', 'reference': 'org_kron'}}, |
| python | src/visualization/prepare_plots.py | 979 | 'org_kron' | 'noise': {'x': {'key': 'org_kron'}, 'y': {'key': 'noise_kron', 'transform': 'delta', 'reference': 'org_kron'}}, |
| python | src/visualization/prepare_plots.py | 979 | 'noise_kron' | 'noise': {'x': {'key': 'org_kron'}, 'y': {'key': 'noise_kron', 'transform': 'delta', 'reference': 'org_kron'}}, |
| python | src/visualization/prepare_plots.py | 980 | 'clip_mode' | 'clip_mode': 'separate', |
| python | src/visualization/prepare_plots.py | 981 | 'norm_mode' | 'norm_mode': 'column', |
| python | src/visualization/prepare_plots.py | 982 | 'norm_key' | 'norm_key': 'org_kron', |
| python | src/visualization/prepare_plots.py | 982 | 'org_kron' | 'norm_key': 'org_kron', |
| python | src/visualization/prepare_plots.py | 983 | 'legend_stat' | 'legend_stat': 'min', |
| python | src/visualization/prepare_plots.py | 984 | 'xlabel_rec' | 'xlabel_rec': r'Kron Radius $[arcsec]$', |
| python | src/visualization/prepare_plots.py | 985 | 'xlabel_noise' | 'xlabel_noise': arcsec_label, |
| python | src/visualization/prepare_plots.py | 986 | 'ylabel_rec' | 'ylabel_rec': delta_kron_label, |
| python | src/visualization/prepare_plots.py | 997 | 0 | valid = ~(np.isnan(x) \| np.isnan(y)) & np.isfinite(x) & np.isfinite(y) & (x > 0) |
| python | src/visualization/prepare_plots.py | 1002 | 25 | bin_p25, _, _ = binned_statistic(x, y, statistic=lambda y: np.nanpercentile(y, 25), bins=bins) |
| python | src/visualization/prepare_plots.py | 1003 | 75 | bin_p75, _, _ = binned_statistic(x, y, statistic=lambda y: np.nanpercentile(y, 75), bins=bins) |
| python | src/visualization/prepare_plots.py | 1005 | 2 | bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 |
| python | src/visualization/prepare_plots.py | 1010 | 'exp_ratio' | def new_metrics(df, exp_time_col='exp_ratio'): |
| python | src/visualization/prepare_plots.py | 1015 | 'flux_x_org' | tp = group[(~group['flux_x_org'].isna()) & (~group['flux_x_rec'].isna())] |
| python | src/visualization/prepare_plots.py | 1015 | 'flux_x_rec' | tp = group[(~group['flux_x_org'].isna()) & (~group['flux_x_rec'].isna())] |
| python | src/visualization/prepare_plots.py | 1016 | 'flux_x_org' | fp = group[(group['flux_x_org'].isna()) & (~group['flux_x_rec'].isna())] |
| python | src/visualization/prepare_plots.py | 1016 | 'flux_x_rec' | fp = group[(group['flux_x_org'].isna()) & (~group['flux_x_rec'].isna())] |
| python | src/visualization/prepare_plots.py | 1017 | 'flux_x_org' | fn = group[(~group['flux_x_org'].isna()) & (group['flux_x_rec'].isna())] |
| python | src/visualization/prepare_plots.py | 1017 | 'flux_x_rec' | fn = group[(~group['flux_x_org'].isna()) & (group['flux_x_rec'].isna())] |
| python | src/visualization/prepare_plots.py | 1020 | 0 | precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0 |
| python | src/visualization/prepare_plots.py | 1021 | 0 | recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0 |
| python | src/visualization/prepare_plots.py | 1022 | 0 | f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0 |
| python | src/visualization/prepare_plots.py | 1022 | 2 | f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0 |
| python | src/visualization/prepare_plots.py | 1025 | 'exp_ratio' | 'exp_ratio': int(exp_ratio), |
| python | src/visualization/prepare_plots.py | 1040 | 0 | overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0 |
| python | src/visualization/prepare_plots.py | 1041 | 0 | overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0 |
| python | src/visualization/prepare_plots.py | 1042 | 0 | overall_f_measure = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0 |
| python | src/visualization/prepare_plots.py | 1042 | 2 | overall_f_measure = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0 |
| python | src/visualization/prepare_plots.py | 1045 | 'exp_ratio' | 'exp_ratio': 0, |
| python | src/visualization/prepare_plots.py | 1062 | 2 | fig, axes = plt.subplots(2, 4, figsize=(16, 8)) |
| python | src/visualization/prepare_plots.py | 1062 | 4 | fig, axes = plt.subplots(2, 4, figsize=(16, 8)) |
| python | src/visualization/prepare_plots.py | 1062 | 16 | fig, axes = plt.subplots(2, 4, figsize=(16, 8)) |
| python | src/visualization/prepare_plots.py | 1062 | 8 | fig, axes = plt.subplots(2, 4, figsize=(16, 8)) |
| python | src/visualization/prepare_plots.py | 1064 | 0 | j = 0 |
| python | src/visualization/prepare_plots.py | 1066 | 'nsr_org' | df['nsr_org'] = safe_divide(df[flux_org_col], df[flux_org_err_col]) |
| python | src/visualization/prepare_plots.py | 1067 | 'nsr_rec' | df['nsr_rec'] = safe_divide(df[flux_rec_col], df[flux_rec_err_col]) |
| python | src/visualization/prepare_plots.py | 1068 | 'nsr_noise' | df['nsr_noise'] = safe_divide(df[flux_noise_col], df[flux_noise_err_col]) |
| python | src/visualization/prepare_plots.py | 1069 | 'exp_ratio' | for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())): |
| python | src/visualization/prepare_plots.py | 1071 | 2 | i = (index % 2) |
| python | src/visualization/prepare_plots.py | 1074 | 0 | if int(exp_ratio) != 0: |
| python | src/visualization/prepare_plots.py | 1075 | 'exp_ratio' | sub_df = df[df['exp_ratio'] == exp_ratio] |
| python | src/visualization/prepare_plots.py | 1076 | '$\\gamma=' | label = rf'$\gamma={int(exp_ratio)}$' |
| python | src/visualization/prepare_plots.py | 1082 | 'No positive flux values for SNR plot at exp_ratio ' | logging.warning(f"No positive flux values for SNR plot at exp_ratio {exp_ratio}, skipping.") |
| python | src/visualization/prepare_plots.py | 1084 | 20 | bins = np.logspace(np.log10(positive_org_flux.min()), np.log10(positive_org_flux.max()), 20) |
| python | src/visualization/prepare_plots.py | 1087 | 'nsr_org' | gt_x, gt_median, gt_p25, gt_p75 = binned_median(sub_df[flux_org_col], sub_df['nsr_org'], bins) |
| python | src/visualization/prepare_plots.py | 1088 | 'nsr_noise' | noisy_x, noisy_median, noisy_p25, noisy_p75 = binned_median(sub_df[flux_noise_col], sub_df['nsr_noise'], bins) |
| python | src/visualization/prepare_plots.py | 1089 | 'nsr_rec' | rec_x, rec_median, rec_p25, rec_p75 = binned_median(sub_df[flux_rec_col], sub_df['nsr_rec'], bins) |
| python | src/visualization/prepare_plots.py | 1093 | 0.2 | ax.fill_between(gt_x, gt_p25, gt_p75, color="black", alpha=0.2) |
| python | src/visualization/prepare_plots.py | 1097 | 0.2 | ax.fill_between(noisy_x, noisy_p25, noisy_p75, color="darkblue", alpha=0.2) |
| python | src/visualization/prepare_plots.py | 1101 | 0.2 | ax.fill_between(rec_x, rec_p25, rec_p75, color="darkgreen", alpha=0.2) |
| python | src/visualization/prepare_plots.py | 1104 | '--' | ax.grid(True, which="both", linestyle="--", alpha=0.3) |
| python | src/visualization/prepare_plots.py | 1104 | 0.3 | ax.grid(True, which="both", linestyle="--", alpha=0.3) |
| python | src/visualization/prepare_plots.py | 1108 | 0 | if j == 0: |
| python | src/visualization/prepare_plots.py | 1110 | 1 | if i == 1: |
| python | src/visualization/prepare_plots.py | 1111 | 'Flux $[erg\\ s^{-1}\\ cm^{-2}\\ \\AA^{-1}]$' | ax.set_xlabel(r'Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$') |
| python | src/visualization/prepare_plots.py | 1112 | 1 | if i == 1: |
| python | src/visualization/prepare_plots.py | 1115 | 16 | fig.suptitle(r"SNR vs. Flux", fontsize=16) |
| python | src/visualization/prepare_plots.py | 1116 | 0.98 | plt.tight_layout(rect=[0, 0, 1, 0.98]) |
| python | src/visualization/prepare_plots.py | 1117 | 500 | plt.savefig(output_filepath, dpi=500) |
| python | src/visualization/prepare_plots.py | 1148 | 'prepare_plots' | vis_cfg = cfg['prepare_plots'] |
| python | src/visualization/prepare_plots.py | 1150 | 'output_dir' | _output_dir        = vis_cfg['output_dir'] |
| python | src/visualization/prepare_plots.py | 1153 | 'uncropped_output_dir' | vis_cfg['uncropped_output_dir'], |
| python | src/visualization/prepare_plots.py | 1154 | 'photometrical_data_filename' | vis_cfg['photometrical_data_filename'] |
| python | src/visualization/prepare_plots.py | 1157 | 'uncropped_results_csv' | _metrics_filename  = vis_cfg['uncropped_results_csv'] |
| python | src/visualization/prepare_plots.py | 1158 | 'metadata_filepath' | _metadata_filepath = vis_cfg['metadata_filepath'] |
| python | src/visualization/prepare_plots.py | 1159 | 'rec_cmap' | _rec_cmap          = vis_cfg['rec_cmap'] |
| python | src/visualization/prepare_plots.py | 1160 | 'noise_cmap' | _noise_cmap        = vis_cfg['noise_cmap'] |
| python | src/visualization/prepare_plots.py | 1161 | 'norm_quantiles' | _norm_quantiles    = vis_cfg['norm_quantiles'] |
| python | src/visualization/prepare_plots.py | 1163 | 2 | if len(_norm_quantiles) != 2: |
| python | src/visualization/prepare_plots.py | 1164 | 'visualization.prepare_plots.norm_quantiles must contain exactly two values.' | raise ValueError('visualization.prepare_plots.norm_quantiles must contain exactly two values.') |
| python | src/visualization/prepare_plots.py | 1165 | 0 | if _norm_quantiles[0] >= _norm_quantiles[1]: |
| python | src/visualization/prepare_plots.py | 1165 | 1 | if _norm_quantiles[0] >= _norm_quantiles[1]: |
| python | src/visualization/prepare_plots.py | 1166 | 'visualization.prepare_plots.norm_quantiles must be strictly increasing.' | raise ValueError('visualization.prepare_plots.norm_quantiles must be strictly increasing.') |
| python | src/visualization/prepare_plots.py | 1167 | 0 | if _norm_quantiles[0] < 0 or _norm_quantiles[1] > 100: |
| python | src/visualization/prepare_plots.py | 1167 | 1 | if _norm_quantiles[0] < 0 or _norm_quantiles[1] > 100: |
| python | src/visualization/prepare_plots.py | 1167 | 100 | if _norm_quantiles[0] < 0 or _norm_quantiles[1] > 100: |
| python | src/visualization/prepare_plots.py | 1168 | 'visualization.prepare_plots.norm_quantiles must stay within [0, 100].' | raise ValueError('visualization.prepare_plots.norm_quantiles must stay within [0, 100].') |
| python | src/visualization/prepare_plots.py | 1173 | 'edited_' | edited_filepath = os.path.join(os.path.dirname(path_to_data), f'edited_{os.path.basename(path_to_data)}') |
| python | src/visualization/prepare_plots.py | 1184 | 'abmag_cflux_rec' | bright_df = df[(df['abmag_cflux_rec'] <= 25.0) \| (df['abmag_cflux_org'] <= 25.0)].reset_index(drop=True) |
| python | src/visualization/prepare_plots.py | 1184 | 25.0 | bright_df = df[(df['abmag_cflux_rec'] <= 25.0) \| (df['abmag_cflux_org'] <= 25.0)].reset_index(drop=True) |
| python | src/visualization/prepare_plots.py | 1184 | 'abmag_cflux_org' | bright_df = df[(df['abmag_cflux_rec'] <= 25.0) \| (df['abmag_cflux_org'] <= 25.0)].reset_index(drop=True) |
| python | src/visualization/prepare_plots.py | 1185 | 'exp_ratio' | bright_metrics = new_metrics(bright_df, exp_time_col='exp_ratio') |
| python | src/visualization/prepare_plots.py | 1187 | 'bright_' | columns={col: 'bright_' + col for col in bright_metrics.columns if col != 'exp_ratio'} |
| python | src/visualization/prepare_plots.py | 1187 | 'exp_ratio' | columns={col: 'bright_' + col for col in bright_metrics.columns if col != 'exp_ratio'} |
| python | src/visualization/prepare_plots.py | 1190 | 'exp_ratio' | all_metrics = pd.merge(bright_metrics, metrics, on=['exp_ratio']).sort_values(by=['exp_ratio']) |
| python | src/visualization/prepare_plots.py | 1192 | 'abmag_cflux_org' | bright_df = df[df['abmag_cflux_org'] <= 25.0].dropna(subset=['abmag_cflux_org']).reset_index(drop=True) |
| python | src/visualization/prepare_plots.py | 1192 | 25.0 | bright_df = df[df['abmag_cflux_org'] <= 25.0].dropna(subset=['abmag_cflux_org']).reset_index(drop=True) |
| python | src/visualization/prepare_plots.py | 1196 | 'org_flux' | 'org_flux': 'a_cflux_org', |
| python | src/visualization/prepare_plots.py | 1196 | 'a_cflux_org' | 'org_flux': 'a_cflux_org', |
| python | src/visualization/prepare_plots.py | 1197 | 'rec_flux' | 'rec_flux': 'a_cflux_rec', |
| python | src/visualization/prepare_plots.py | 1197 | 'a_cflux_rec' | 'rec_flux': 'a_cflux_rec', |
| python | src/visualization/prepare_plots.py | 1198 | 'noise_flux' | 'noise_flux': 'a_cflux_noise', |
| python | src/visualization/prepare_plots.py | 1198 | 'a_cflux_noise' | 'noise_flux': 'a_cflux_noise', |
| python | src/visualization/prepare_plots.py | 1199 | 'org_kron' | 'org_kron': 'kron_radius_org', |
| python | src/visualization/prepare_plots.py | 1199 | 'kron_radius_org' | 'org_kron': 'kron_radius_org', |
| python | src/visualization/prepare_plots.py | 1200 | 'rec_kron' | 'rec_kron': 'kron_radius_rec', |
| python | src/visualization/prepare_plots.py | 1200 | 'kron_radius_rec' | 'rec_kron': 'kron_radius_rec', |
| python | src/visualization/prepare_plots.py | 1201 | 'noise_kron' | 'noise_kron': 'kron_radius_noise', |
| python | src/visualization/prepare_plots.py | 1201 | 'kron_radius_noise' | 'noise_kron': 'kron_radius_noise', |
| python | src/visualization/prepare_plots.py | 1202 | 'org_mag' | 'org_mag': 'abmag_cflux_org', |
| python | src/visualization/prepare_plots.py | 1202 | 'abmag_cflux_org' | 'org_mag': 'abmag_cflux_org', |
| python | src/visualization/prepare_plots.py | 1203 | 'org_flux_err' | 'org_flux_err': 'a_flux_err_org', |
| python | src/visualization/prepare_plots.py | 1203 | 'a_flux_err_org' | 'org_flux_err': 'a_flux_err_org', |
| python | src/visualization/prepare_plots.py | 1204 | 'noise_flux_err' | 'noise_flux_err': 'a_flux_err_noise', |
| python | src/visualization/prepare_plots.py | 1204 | 'a_flux_err_noise' | 'noise_flux_err': 'a_flux_err_noise', |
| python | src/visualization/prepare_plots.py | 1205 | 'rec_flux_err' | 'rec_flux_err': 'a_flux_err_rec', |
| python | src/visualization/prepare_plots.py | 1205 | 'a_flux_err_rec' | 'rec_flux_err': 'a_flux_err_rec', |
| python | src/visualization/prepare_plots.py | 1209 | 'snr.png' | snr_filename = 'snr.png' |
| python | src/visualization/prepare_plots.py | 1214 | '_' | os.path.join(my_output_dir, my_tag + "_" + spec['output_filename']), |
| python | src/visualization/prepare_plots.py | 1214 | 'output_filename' | os.path.join(my_output_dir, my_tag + "_" + spec['output_filename']), |
| python | src/visualization/prepare_plots.py | 1220 | 'output_filename' | logging.warning(f"Error in {spec['output_filename']}: {e}") |
| python | src/visualization/prepare_plots.py | 1225 | 'org_flux' | columns['org_flux'], |
| python | src/visualization/prepare_plots.py | 1226 | 'noise_flux' | columns['noise_flux'], |
| python | src/visualization/prepare_plots.py | 1227 | 'rec_flux' | columns['rec_flux'], |
| python | src/visualization/prepare_plots.py | 1228 | 'org_flux_err' | columns['org_flux_err'], |
| python | src/visualization/prepare_plots.py | 1229 | 'noise_flux_err' | columns['noise_flux_err'], |
| python | src/visualization/prepare_plots.py | 1230 | 'rec_flux_err' | columns['rec_flux_err'], |
| python | src/visualization/prepare_plots.py | 1231 | '_' | os.path.join(my_output_dir, my_tag + "_" + snr_filename), |
| python | src/visualization/prepare_plots.py | 1234 | 'Error in create_flux_flux_error_diagram: ' | logging.warning(f"Error in create_flux_flux_error_diagram: {e}") |
| python | src/visualization/prepare_plots.py | 1238 | 'all_metrics.csv' | out_csv = os.path.join(_output_dir, 'all_metrics.csv') |
| python | src/visualization/prepare_plots.py | 1244 | '__main__' | if __name__ == "__main__": |
| python | starter.py | 28 | 'config.yaml' | CONFIG_PATH = Path(__file__).resolve().with_name('config.yaml') |
| python | starter.py | 44 | 2 | 'MSE': 2, |
| python | starter.py | 45 | 3 | 'MAE': 3, |
| python | starter.py | 46 | 4 | 'SSIM': 4, |
| python | starter.py | 47 | 5 | 'SIMAE': 5, |
| python | starter.py | 48 | 6 | 'LOGCOSH': 6, |
| python | starter.py | 52 | 'z_scale' | 'z_scale': 1, |
| python | starter.py | 53 | 'min_max' | 'min_max': 2, |
| python | starter.py | 53 | 2 | 'min_max': 2, |
| python | starter.py | 54 | 'log_min_max' | 'log_min_max': 3, |
| python | starter.py | 54 | 3 | 'log_min_max': 3, |
| python | starter.py | 59 | 2 | 'leakyrelu': 2, |
| python | starter.py | 60 | 'sigmoid' | 'sigmoid': 3, |
| python | starter.py | 60 | 3 | 'sigmoid': 3, |
| python | starter.py | 61 | 4 | 'tanh': 4, |
| python | starter.py | 65 | 'data_augment' | 'data_augment': ('src.training.new_train', 'data_augment'), |
| python | starter.py | 65 | 'src.training.new_train' | 'data_augment': ('src.training.new_train', 'data_augment'), |
| python | starter.py | 66 | 'data_augment_pluggable' | 'data_augment_pluggable': ('src.training.new_train', 'data_augment_pluggable'), |
| python | starter.py | 66 | 'src.training.new_train' | 'data_augment_pluggable': ('src.training.new_train', 'data_augment_pluggable'), |
| python | starter.py | 67 | 'log_cosh_loss' | 'log_cosh_loss': ('src.training.utils', 'log_cosh_loss'), |
| python | starter.py | 68 | 'scale_invariant_mae' | 'scale_invariant_mae': ('src.training.utils', 'scale_invariant_mae'), |
| python | starter.py | 69 | 'ssim_loss' | 'ssim_loss': ('src.training.utils', 'ssim_loss'), |
| python | starter.py | 70 | '_simulated_image_from_exposure' | '_simulated_image_from_exposure': ('src.training.math_helpers', '_simulated_image_from_exposure'), |
| python | starter.py | 70 | 'src.training.math_helpers' | '_simulated_image_from_exposure': ('src.training.math_helpers', '_simulated_image_from_exposure'), |
| python | starter.py | 71 | '_simulated_image_from_poisson' | '_simulated_image_from_poisson': ('src.training.math_helpers', '_simulated_image_from_poisson'), |
| python | starter.py | 71 | 'src.training.math_helpers' | '_simulated_image_from_poisson': ('src.training.math_helpers', '_simulated_image_from_poisson'), |
| python | starter.py | 72 | 'wrap_extract_sources' | 'wrap_extract_sources': ('src.evaluation.metrics', 'wrap_extract_sources'), |
| python | starter.py | 73 | 'detect_sources_in_image' | 'detect_sources_in_image': ('src.evaluation.metrics', 'detect_sources_in_image'), |
| python | starter.py | 119 | 'new_train' | return 'gan' if config_data['new_train']['training']['use_gan'] else 'unet' |
| python | starter.py | 119 | 'use_gan' | return 'gan' if config_data['new_train']['training']['use_gan'] else 'unet' |
| python | starter.py | 119 | 'gan' | return 'gan' if config_data['new_train']['training']['use_gan'] else 'unet' |
| python | starter.py | 119 | 'unet' | return 'gan' if config_data['new_train']['training']['use_gan'] else 'unet' |
| python | starter.py | 123 | 'model_type' | model_type = explicit_overrides.get('model_type') |
| python | starter.py | 126 | 'gan' | if str(model_type).lower() == 'gan': |
| python | starter.py | 127 | 'new_train' | return config_data['new_train']['gan']['loss_fn'] |
| python | starter.py | 127 | 'gan' | return config_data['new_train']['gan']['loss_fn'] |
| python | starter.py | 127 | 'loss_fn' | return config_data['new_train']['gan']['loss_fn'] |
| python | starter.py | 128 | 'new_train' | return config_data['new_train']['training']['g_loss_fn'] |
| python | starter.py | 128 | 'g_loss_fn' | return config_data['new_train']['training']['g_loss_fn'] |
| python | starter.py | 132 | 'new_train' | config_data['new_train']['training']['use_gan'] = str(value).lower() == 'gan' |
| python | starter.py | 132 | 'use_gan' | config_data['new_train']['training']['use_gan'] = str(value).lower() == 'gan' |
| python | starter.py | 132 | 'gan' | config_data['new_train']['training']['use_gan'] = str(value).lower() == 'gan' |
| python | starter.py | 138 | '--nsigma' | flag='--nsigma', |
| python | starter.py | 139 | 0 | positional_index=0, |
| python | starter.py | 141 | 'create_dataset' | default_getter=_config_path_getter('create_dataset', 'nsigma'), |
| python | starter.py | 142 | 'create_dataset' | apply=_config_path_setter('create_dataset', 'nsigma'), |
| python | starter.py | 145 | 'footprint_radius' | key='footprint_radius', |
| python | starter.py | 146 | '--footprint-radius' | flag='--footprint-radius', |
| python | starter.py | 147 | 1 | positional_index=1, |
| python | starter.py | 148 | 'footprint_radius' | coerce=lambda value: parse_optional_int(value, 'footprint_radius'), |
| python | starter.py | 149 | 'create_dataset' | default_getter=_config_path_getter('create_dataset', 'footprint_radius'), |
| python | starter.py | 149 | 'footprint_radius' | default_getter=_config_path_getter('create_dataset', 'footprint_radius'), |
| python | starter.py | 150 | 'create_dataset' | apply=_config_path_setter('create_dataset', 'footprint_radius'), |
| python | starter.py | 150 | 'footprint_radius' | apply=_config_path_setter('create_dataset', 'footprint_radius'), |
| python | starter.py | 154 | '--npixels' | flag='--npixels', |
| python | starter.py | 155 | 2 | positional_index=2, |
| python | starter.py | 157 | 'create_dataset' | default_getter=_config_path_getter('create_dataset', 'npixels'), |
| python | starter.py | 158 | 'create_dataset' | apply=_config_path_setter('create_dataset', 'npixels'), |
| python | starter.py | 161 | 'model_type' | key='model_type', |
| python | starter.py | 162 | '--model-type' | flag='--model-type', |
| python | starter.py | 163 | 3 | positional_index=3, |
| python | starter.py | 170 | '--attention' | flag='--attention', |
| python | starter.py | 171 | 4 | positional_index=4, |
| python | starter.py | 173 | 'new_train' | default_getter=_config_path_getter('new_train', 'network', 'attention'), |
| python | starter.py | 174 | 'new_train' | apply=_config_path_setter('new_train', 'network', 'attention'), |
| python | starter.py | 178 | '--scaling' | flag='--scaling', |
| python | starter.py | 179 | 5 | positional_index=5, |
| python | starter.py | 181 | 'new_train' | default_getter=_config_path_getter('new_train', 'training', 'scaling'), |
| python | starter.py | 182 | 'new_train' | apply=_config_path_setter('new_train', 'training', 'scaling'), |
| python | starter.py | 185 | 'loss_name' | key='loss_name', |
| python | starter.py | 186 | '--loss-name' | flag='--loss-name', |
| python | starter.py | 187 | 6 | positional_index=6, |
| python | starter.py | 190 | 'new_train' | apply=_config_path_setter('new_train', 'training', 'g_loss_fn'), |
| python | starter.py | 190 | 'g_loss_fn' | apply=_config_path_setter('new_train', 'training', 'g_loss_fn'), |
| python | starter.py | 193 | 'dropout_rate' | key='dropout_rate', |
| python | starter.py | 194 | '--dropout-rate' | flag='--dropout-rate', |
| python | starter.py | 195 | 7 | positional_index=7, |
| python | starter.py | 196 | 'dropout_rate' | coerce=lambda value: parse_optional_float(value, 'dropout_rate'), |
| python | starter.py | 197 | 'new_train' | default_getter=_config_path_getter('new_train', 'network', 'dropout_rate'), |
| python | starter.py | 197 | 'dropout_rate' | default_getter=_config_path_getter('new_train', 'network', 'dropout_rate'), |
| python | starter.py | 198 | 'new_train' | apply=_config_path_setter('new_train', 'network', 'dropout_rate'), |
| python | starter.py | 198 | 'dropout_rate' | apply=_config_path_setter('new_train', 'network', 'dropout_rate'), |
| python | starter.py | 201 | 'output_activation' | key='output_activation', |
| python | starter.py | 202 | '--output-activation' | flag='--output-activation', |
| python | starter.py | 203 | 8 | positional_index=8, |
| python | starter.py | 205 | 'new_train' | default_getter=_config_path_getter('new_train', 'network', 'output_activation'), |
| python | starter.py | 205 | 'output_activation' | default_getter=_config_path_getter('new_train', 'network', 'output_activation'), |
| python | starter.py | 206 | 'new_train' | apply=_config_path_setter('new_train', 'network', 'output_activation'), |
| python | starter.py | 206 | 'output_activation' | apply=_config_path_setter('new_train', 'network', 'output_activation'), |
| python | starter.py | 209 | 'kernel_initializer' | key='kernel_initializer', |
| python | starter.py | 210 | '--kernel-initializer' | flag='--kernel-initializer', |
| python | starter.py | 211 | 9 | positional_index=9, |
| python | starter.py | 213 | 'new_train' | default_getter=_config_path_getter('new_train', 'network', 'kernel_initializer'), |
| python | starter.py | 213 | 'kernel_initializer' | default_getter=_config_path_getter('new_train', 'network', 'kernel_initializer'), |
| python | starter.py | 214 | 'new_train' | apply=_config_path_setter('new_train', 'network', 'kernel_initializer'), |
| python | starter.py | 214 | 'kernel_initializer' | apply=_config_path_setter('new_train', 'network', 'kernel_initializer'), |
| python | starter.py | 217 | 'activation_name' | key='activation_name', |
| python | starter.py | 218 | '--activation-name' | flag='--activation-name', |
| python | starter.py | 219 | 10 | positional_index=10, |
| python | starter.py | 221 | 'new_train' | default_getter=_config_path_getter('new_train', 'network', 'func'), |
| python | starter.py | 222 | 'new_train' | apply=_config_path_setter('new_train', 'network', 'func'), |
| python | starter.py | 225 | 'discriminator_activation' | key='discriminator_activation', |
| python | starter.py | 226 | '--discriminator-activation' | flag='--discriminator-activation', |
| python | starter.py | 227 | 11 | positional_index=11, |
| python | starter.py | 229 | 'new_train' | default_getter=_config_path_getter('new_train', 'discriminator', 'func'), |
| python | starter.py | 230 | 'new_train' | apply=_config_path_setter('new_train', 'discriminator', 'func'), |
| python | starter.py | 233 | 'discriminator_output_activation' | key='discriminator_output_activation', |
| python | starter.py | 234 | '--discriminator-output-activation' | flag='--discriminator-output-activation', |
| python | starter.py | 235 | 12 | positional_index=12, |
| python | starter.py | 237 | 'new_train' | default_getter=_config_path_getter('new_train', 'discriminator', 'output_activation'), |
| python | starter.py | 237 | 'output_activation' | default_getter=_config_path_getter('new_train', 'discriminator', 'output_activation'), |
| python | starter.py | 238 | 'new_train' | apply=_config_path_setter('new_train', 'discriminator', 'output_activation'), |
| python | starter.py | 238 | 'output_activation' | apply=_config_path_setter('new_train', 'discriminator', 'output_activation'), |
| python | starter.py | 241 | 'filter_surveys' | key='filter_surveys', |
| python | starter.py | 242 | '--filter-surveys' | flag='--filter-surveys', |
| python | starter.py | 243 | 13 | positional_index=13, |
| python | starter.py | 244 | 'filter_surveys' | coerce=lambda value: parse_optional_bool(value, 'filter_surveys'), |
| python | starter.py | 245 | 'create_dataset' | default_getter=_config_path_getter('create_dataset', 'filter_surveys'), |
| python | starter.py | 245 | 'filter_surveys' | default_getter=_config_path_getter('create_dataset', 'filter_surveys'), |
| python | starter.py | 246 | 'create_dataset' | apply=_config_path_setter('create_dataset', 'filter_surveys'), |
| python | starter.py | 246 | 'filter_surveys' | apply=_config_path_setter('create_dataset', 'filter_surveys'), |
| python | starter.py | 249 | 'filter_by_last_name' | key='filter_by_last_name', |
| python | starter.py | 250 | '--filter-by-last-name' | flag='--filter-by-last-name', |
| python | starter.py | 251 | 14 | positional_index=14, |
| python | starter.py | 252 | 'filter_by_last_name' | coerce=lambda value: parse_optional_bool(value, 'filter_by_last_name'), |
| python | starter.py | 253 | 'create_dataset' | default_getter=_config_path_getter('create_dataset', 'filter_by_last_name'), |
| python | starter.py | 253 | 'filter_by_last_name' | default_getter=_config_path_getter('create_dataset', 'filter_by_last_name'), |
| python | starter.py | 254 | 'create_dataset' | apply=_config_path_setter('create_dataset', 'filter_by_last_name'), |
| python | starter.py | 254 | 'filter_by_last_name' | apply=_config_path_setter('create_dataset', 'filter_by_last_name'), |
| python | starter.py | 257 | 'last_name_filter_value' | key='last_name_filter_value', |
| python | starter.py | 258 | '--last-name-filter-value' | flag='--last-name-filter-value', |
| python | starter.py | 259 | 15 | positional_index=15, |
| python | starter.py | 261 | 'create_dataset' | default_getter=_config_path_getter('create_dataset', 'last_name_filter_value'), |
| python | starter.py | 261 | 'last_name_filter_value' | default_getter=_config_path_getter('create_dataset', 'last_name_filter_value'), |
| python | starter.py | 262 | 'create_dataset' | apply=_config_path_setter('create_dataset', 'last_name_filter_value'), |
| python | starter.py | 262 | 'last_name_filter_value' | apply=_config_path_setter('create_dataset', 'last_name_filter_value'), |
| python | starter.py | 276 | 1 | @lru_cache(maxsize=1) |
| python | starter.py | 286 | '--' | if any(token.startswith('--') for token in cli_tokens): |
| python | starter.py | 308 | 0 | if not isinstance(value, int) or value < 0 or value > 65535: |
| python | starter.py | 308 | 65535 | if not isinstance(value, int) or value < 0 or value > 65535: |
| python | starter.py | 310 | 2 | return value.to_bytes(2, 'big') |
| python | starter.py | 313 | 2 | if cursor + 2 > len(raw): |
| python | starter.py | 315 | 2 | return int.from_bytes(raw[cursor:cursor + 2], 'big'), cursor + 2 |
| python | starter.py | 330 | 255 | if len(normalized) > 255: |
| python | starter.py | 359 | 1 | if marker == 1: |
| python | starter.py | 367 | 0 | if marker == 0: |
| python | starter.py | 376 | 7 | if len(payload_obj) == 7: |
| python | starter.py | 377 | 0 | filter_surveys = bool(payload_obj[0]) |
| python | starter.py | 378 | 1 | allowed_survey = [str(item) for item in (payload_obj[1] or [])] |
| python | starter.py | 379 | 2 | filter_by_last_name = bool(payload_obj[2]) |
| python | starter.py | 380 | 3 | last_name_filter_value = [str(item) for item in (payload_obj[3] or [])] if filter_by_last_name else [] |
| python | starter.py | 381 | 4 | nsigma = int(payload_obj[4]) |
| python | starter.py | 382 | 5 | footprint_radius = int(payload_obj[5]) |
| python | starter.py | 383 | 6 | npixels = int(payload_obj[6]) |
| python | starter.py | 385 | 0 | flags = 0 |
| python | starter.py | 389 | 2 | flags \|= 2 |
| python | starter.py | 395 | 4 | flags \|= 4 |
| python | starter.py | 400 | 0 | survey_mask = 0 |
| python | starter.py | 402 | 1 | survey_mask \|= 1 << _KNOWN_SURVEYS.index(survey) |
| python | starter.py | 415 | 9 | compressed = zlib.compress(raw, level=9) |
| python | starter.py | 418 | 10 | if len(payload_obj) == 10: |
| python | starter.py | 419 | 0 | is_gan = str(payload_obj[0]) |
| python | starter.py | 420 | 1 | use_attention = str(payload_obj[1]) |
| python | starter.py | 421 | 2 | loss_function = str(payload_obj[2]) |
| python | starter.py | 422 | 3 | data_alias_enriched_hex = str(payload_obj[3]) |
| python | starter.py | 423 | 4 | scaling_tag = str(payload_obj[4]) |
| python | starter.py | 424 | 5 | dropout_tag = str(payload_obj[5]) |
| python | starter.py | 425 | 6 | activation_tag = str(payload_obj[6]) |
| python | starter.py | 426 | 7 | output_activation_tag = str(payload_obj[7]) |
| python | starter.py | 427 | 8 | discriminator_activation_tag = str(payload_obj[8]) |
| python | starter.py | 428 | 9 | discriminator_output_activation_tag = str(payload_obj[9]) |
| python | starter.py | 430 | 0 | flags = 0 |
| python | starter.py | 431 | 'GAN' | if is_gan == 'GAN': |
| python | starter.py | 434 | 2 | flags \|= 2 |
| python | starter.py | 453 | 9 | compressed = zlib.compress(raw, level=9) |
| python | starter.py | 464 | 4 | padded = token + ('=' * (-len(token) % 4)) |
| python | starter.py | 470 | 3 | if len(raw) < 3: |
| python | starter.py | 473 | 0 | kind = chr(raw[0]) |
| python | starter.py | 474 | 1 | version = raw[1] |
| python | starter.py | 475 | 2 | cursor = 2 |
| python | starter.py | 476 | 1 | if version != 1: |
| python | starter.py | 483 | 1 | filter_surveys = bool(flags & 1) |
| python | starter.py | 484 | 2 | filter_by_last_name = bool(flags & 2) |
| python | starter.py | 485 | 4 | use_mask = bool(flags & 4) |
| python | starter.py | 492 | 1 | if survey_mask & (1 << index): |
| python | starter.py | 521 | 1 | is_gan = 'GAN' if (flags & 1) else 'UNET' |
| python | starter.py | 521 | 'GAN' | is_gan = 'GAN' if (flags & 1) else 'UNET' |
| python | starter.py | 521 | 'UNET' | is_gan = 'GAN' if (flags & 1) else 'UNET' |
| python | starter.py | 522 | 2 | use_attention = 'ATTN' if (flags & 2) else 'NOATTN' |
| python | starter.py | 562 | '_' | normalized = ''.join(ch.lower() if ch.isalnum() else '_' for ch in text) |
| python | starter.py | 563 | '_' | normalized = normalized.strip('_') |
| python | starter.py | 572 | 9 | compressed = zlib.compress(raw, level=9) |
| python | starter.py | 578 | "Invalid data_alias_hex '" | raise ValueError(f"Invalid data_alias_hex '{data_alias_hex}'.") |
| python | starter.py | 583 | 4 | padded = data_alias_hex + ('=' * (-len(data_alias_hex) % 4)) |
| python | starter.py | 587 | "Invalid data_alias_hex '" | raise ValueError(f"Invalid data_alias_hex '{data_alias_hex}'.") from exc |
| python | starter.py | 589 | 3 | if len(raw) < 3 or chr(raw[0]) != 'A' or raw[1] != 1: |
| python | starter.py | 589 | 0 | if len(raw) < 3 or chr(raw[0]) != 'A' or raw[1] != 1: |
| python | starter.py | 589 | 1 | if len(raw) < 3 or chr(raw[0]) != 'A' or raw[1] != 1: |
| python | starter.py | 590 | "Invalid data_alias_hex '" | raise ValueError(f"Invalid data_alias_hex '{data_alias_hex}'.") |
| python | starter.py | 592 | 2 | value, cursor = _read_string(raw, 2) |
| python | starter.py | 594 | "Invalid data_alias_hex '" | raise ValueError(f"Invalid data_alias_hex '{data_alias_hex}'.") |
| python | starter.py | 609 | 'SV_' | data_alias_plain = 'SV_' + '_'.join(allowed_survey) |
| python | starter.py | 609 | '_' | data_alias_plain = 'SV_' + '_'.join(allowed_survey) |
| python | starter.py | 611 | '_' | data_alias_plain = (data_alias_plain + '_' if data_alias_plain else '') + 'LN' + '_'.join(last_name_filter_value) |
| python | starter.py | 615 | '_NS' | data_alias_enriched = f"{data_alias_hex_code}_NS{nsigma}_FP{footprint_radius}_NP{npixels}" |
| python | starter.py | 615 | '_FP' | data_alias_enriched = f"{data_alias_hex_code}_NS{nsigma}_FP{footprint_radius}_NP{npixels}" |
| python | starter.py | 615 | '_NP' | data_alias_enriched = f"{data_alias_hex_code}_NS{nsigma}_FP{footprint_radius}_NP{npixels}" |
| python | starter.py | 627 | 'data_alias_plain' | 'data_alias_plain': data_alias_plain, |
| python | starter.py | 628 | 'data_alias_hex' | 'data_alias_hex': data_alias_hex_code, |
| python | starter.py | 629 | 'filter_surveys' | 'filter_surveys': filter_surveys, |
| python | starter.py | 630 | 'allowed_survey' | 'allowed_survey': allowed_survey, |
| python | starter.py | 631 | 'filter_by_last_name' | 'filter_by_last_name': filter_by_last_name, |
| python | starter.py | 632 | 'last_name_filter_value' | 'last_name_filter_value': last_name_filter_value, |
| python | starter.py | 633 | 'data_alias_enriched' | 'data_alias_enriched': data_alias_enriched, |
| python | starter.py | 634 | 'data_alias_enriched_hex' | 'data_alias_enriched_hex': data_alias_enriched_hex_code, |
| python | starter.py | 640 | "Invalid data_alias_enriched_hex '" | raise ValueError(f"Invalid data_alias_enriched_hex '{data_alias_enriched_hex}'.") |
| python | starter.py | 644 | 7 | if not isinstance(payload, list) or len(payload) != 7: |
| python | starter.py | 645 | 'Decoded data_alias_enriched_hex payload must be a 7-item list.' | raise ValueError('Decoded data_alias_enriched_hex payload must be a 7-item list.') |
| python | starter.py | 647 | 0 | filter_surveys = bool(payload[0]) |
| python | starter.py | 648 | 1 | allowed_survey = [] if payload[1] is None else ([payload[1]] if isinstance(payload[1], str) else [str(item) for item in payload[1]]) |
| python | starter.py | 649 | 2 | filter_by_last_name = bool(payload[2]) |
| python | starter.py | 650 | 3 | last_name_filter_value = [] if payload[3] is None else ([payload[3]] if isinstance(payload[3], str) else [str(item) for item in payload[3]]) |
| python | starter.py | 653 | 4 | nsigma = int(payload[4]) |
| python | starter.py | 654 | 5 | footprint_radius = int(payload[5]) |
| python | starter.py | 655 | 6 | npixels = int(payload[6]) |
| python | starter.py | 657 | 'Decoded data_alias_enriched_hex payload contains non-integer dataset parameters.' | raise ValueError('Decoded data_alias_enriched_hex payload contains non-integer dataset parameters.') from exc |
| python | starter.py | 661 | 'SV_' | data_alias = 'SV_' + '_'.join(allowed_survey) |
| python | starter.py | 661 | '_' | data_alias = 'SV_' + '_'.join(allowed_survey) |
| python | starter.py | 663 | '_' | data_alias = (data_alias + '_' if data_alias else '') + 'LN' + '_'.join(last_name_filter_value) |
| python | starter.py | 668 | 'Decoded data_alias_hex payload is not canonical.' | raise ValueError('Decoded data_alias_hex payload is not canonical.') |
| python | starter.py | 670 | '_NS' | data_alias_enriched = f"{data_alias_hex_code}_NS{nsigma}_FP{footprint_radius}_NP{npixels}" |
| python | starter.py | 670 | '_FP' | data_alias_enriched = f"{data_alias_hex_code}_NS{nsigma}_FP{footprint_radius}_NP{npixels}" |
| python | starter.py | 670 | '_NP' | data_alias_enriched = f"{data_alias_hex_code}_NS{nsigma}_FP{footprint_radius}_NP{npixels}" |
| python | starter.py | 682 | 'Decoded data_alias_enriched_hex payload is not canonical.' | raise ValueError('Decoded data_alias_enriched_hex payload is not canonical.') |
| python | starter.py | 685 | 'data_alias_hex' | 'data_alias_hex': data_alias_hex_code, |
| python | starter.py | 686 | 'data_alias_plain' | 'data_alias_plain': data_alias, |
| python | starter.py | 687 | 'filter_surveys' | 'filter_surveys': filter_surveys, |
| python | starter.py | 688 | 'allowed_survey' | 'allowed_survey': allowed_survey, |
| python | starter.py | 689 | 'filter_by_last_name' | 'filter_by_last_name': filter_by_last_name, |
| python | starter.py | 690 | 'last_name_filter_value' | 'last_name_filter_value': last_name_filter_value, |
| python | starter.py | 692 | 'footprint_radius' | 'footprint_radius': footprint_radius, |
| python | starter.py | 694 | 'data_alias_enriched' | 'data_alias_enriched': data_alias_enriched, |
| python | starter.py | 695 | 'data_alias_enriched_hex' | 'data_alias_enriched_hex': data_alias_enriched_hex, |
| python | starter.py | 703 | '_' | f"{is_gan}_{use_attention}_{loss_function}_" |
| python | starter.py | 703 | '_DO' | f"{is_gan}_{use_attention}_{loss_function}_" |
| python | starter.py | 703 | '_ACT' | f"{is_gan}_{use_attention}_{loss_function}_" |
| python | starter.py | 703 | '_OUT' | f"{is_gan}_{use_attention}_{loss_function}_" |
| python | starter.py | 703 | '_DACT' | f"{is_gan}_{use_attention}_{loss_function}_" |
| python | starter.py | 703 | '_DOUT' | f"{is_gan}_{use_attention}_{loss_function}_" |
| python | starter.py | 721 | 'model_alias_plain' | return {'model_alias_plain': model_alias_plain, 'model_alias_hex': model_alias_hex} |
| python | starter.py | 721 | 'model_alias_hex' | return {'model_alias_plain': model_alias_plain, 'model_alias_hex': model_alias_hex} |
| python | starter.py | 726 | "Invalid model_alias_hex '" | raise ValueError(f"Invalid model_alias_hex '{model_alias_hex}'.") |
| python | starter.py | 730 | 10 | if not isinstance(payload, list) or len(payload) != 10: |
| python | starter.py | 731 | 'Decoded model_alias_hex payload must be a 10-item list.' | raise ValueError('Decoded model_alias_hex payload must be a 10-item list.') |
| python | starter.py | 751 | 'Decoded model_alias_hex payload is not canonical.' | raise ValueError('Decoded model_alias_hex payload is not canonical.') |
| python | starter.py | 754 | '_' | f"{is_gan}_{use_attention}_{loss_function}_" |
| python | starter.py | 754 | '_DO' | f"{is_gan}_{use_attention}_{loss_function}_" |
| python | starter.py | 754 | '_ACT' | f"{is_gan}_{use_attention}_{loss_function}_" |
| python | starter.py | 754 | '_OUT' | f"{is_gan}_{use_attention}_{loss_function}_" |
| python | starter.py | 754 | '_DACT' | f"{is_gan}_{use_attention}_{loss_function}_" |
| python | starter.py | 754 | '_DOUT' | f"{is_gan}_{use_attention}_{loss_function}_" |
| python | starter.py | 767 | 'model_alias_hex' | 'model_alias_hex': model_alias_hex, |
| python | starter.py | 768 | 'model_alias_plain' | 'model_alias_plain': model_alias_plain, |
| python | starter.py | 769 | 'model_type' | 'model_type': 'gan' if is_gan == 'GAN' else 'unet', |
| python | starter.py | 769 | 'GAN' | 'model_type': 'gan' if is_gan == 'GAN' else 'unet', |
| python | starter.py | 769 | 'gan' | 'model_type': 'gan' if is_gan == 'GAN' else 'unet', |
| python | starter.py | 769 | 'unet' | 'model_type': 'gan' if is_gan == 'GAN' else 'unet', |
| python | starter.py | 770 | 'is_gan' | 'is_gan': is_gan, |
| python | starter.py | 771 | 'use_attention' | 'use_attention': use_attention, |
| python | starter.py | 773 | 'loss_function' | 'loss_function': loss_function, |
| python | starter.py | 774 | 'data_alias_enriched_hex' | 'data_alias_enriched_hex': data_alias_enriched_hex, |
| python | starter.py | 775 | 'scaling_tag' | 'scaling_tag': scaling_tag, |
| python | starter.py | 776 | 'dropout_tag' | 'dropout_tag': dropout_tag, |
| python | starter.py | 777 | 'dropout_rate' | 'dropout_rate': dropout_rate, |
| python | starter.py | 778 | 'activation_tag' | 'activation_tag': activation_tag, |
| python | starter.py | 779 | 'output_activation_tag' | 'output_activation_tag': output_activation_tag, |
| python | starter.py | 780 | 'discriminator_activation_tag' | 'discriminator_activation_tag': discriminator_activation_tag, |
| python | starter.py | 781 | 'discriminator_output_activation_tag' | 'discriminator_output_activation_tag': discriminator_output_activation_tag, |
| python | starter.py | 788 | "Invalid models_dir '" | raise ValueError(f"Invalid models_dir '{models_dir}'.") |
| python | starter.py | 790 | '\\' | normalized = models_dir.strip().replace('\\', '/').rstrip('/') |
| python | starter.py | 790 | '/' | normalized = models_dir.strip().replace('\\', '/').rstrip('/') |
| python | starter.py | 792 | '(?P<models_root_dir>.+)/(?P<is_gan>GAN\|UNET)/(?P<use_attention>ATTN\|NOATTN)/(?P<loss_function>[^/]+)/(?P<data_alias_enriched_hex>[^/]+)/(?P<scaling_tag>[^/]+)/DO(?P<dropout_tag>[^/]+)/ACT(?P<activation_tag>[^/]+)/OUT(?P<output_activation_tag>[^/]+)/DACT(?P<discriminator_activation_tag>[^/]+)/DOUT(?P<discriminator_output_activation_tag>[^/]+)' | r'(?P<models_root_dir>.+)/(?P<is_gan>GAN\|UNET)/(?P<use_attention>ATTN\|NOATTN)/(?P<loss_function>[^/]+)/(?P<data_alias_enriched_hex>[^/]+)/(?P<scaling_tag>[^/]+)/DO(?P<dropout_tag>[^/]+)/ACT(?P<activation_tag>[^/]+)/OUT(?P<output_activation_tag>[^/]+)/DACT(?P<discriminator_activation_tag>[^/]+)/DOUT(?P<discriminator_output_activation_tag>[^/]+)', |
| python | starter.py | 796 | "Invalid models_dir '" | raise ValueError(f"Invalid models_dir '{models_dir}'.") |
| python | starter.py | 798 | 'data_alias_enriched_hex' | data_alias_enriched_hex = match.group('data_alias_enriched_hex') |
| python | starter.py | 801 | 'dropout_tag' | dropout_tag = match.group('dropout_tag') |
| python | starter.py | 809 | 'is_gan' | match.group('is_gan'), match.group('use_attention'), match.group('loss_function'), |
| python | starter.py | 809 | 'use_attention' | match.group('is_gan'), match.group('use_attention'), match.group('loss_function'), |
| python | starter.py | 809 | 'loss_function' | match.group('is_gan'), match.group('use_attention'), match.group('loss_function'), |
| python | starter.py | 810 | 'scaling_tag' | data_alias_enriched_hex, match.group('scaling_tag'), dropout_tag, |
| python | starter.py | 811 | 'activation_tag' | match.group('activation_tag'), match.group('output_activation_tag'), |
| python | starter.py | 811 | 'output_activation_tag' | match.group('activation_tag'), match.group('output_activation_tag'), |
| python | starter.py | 812 | 'discriminator_activation_tag' | match.group('discriminator_activation_tag'), match.group('discriminator_output_activation_tag'), |
| python | starter.py | 812 | 'discriminator_output_activation_tag' | match.group('discriminator_activation_tag'), match.group('discriminator_output_activation_tag'), |
| python | starter.py | 814 | 'model_alias_hex' | model_alias_hex = _model_alias_dict['model_alias_hex'] |
| python | starter.py | 815 | 'model_alias_plain' | model_alias_plain = _model_alias_dict['model_alias_plain'] |
| python | starter.py | 818 | 'models_dir' | 'models_dir': normalized, |
| python | starter.py | 819 | 'models_root_dir' | 'models_root_dir': match.group('models_root_dir'), |
| python | starter.py | 820 | 'model_alias_hex' | 'model_alias_hex': model_alias_hex, |
| python | starter.py | 821 | 'model_alias_plain' | 'model_alias_plain': model_alias_plain, |
| python | starter.py | 822 | 'model_type' | 'model_type': 'gan' if match.group('is_gan') == 'GAN' else 'unet', |
| python | starter.py | 822 | 'is_gan' | 'model_type': 'gan' if match.group('is_gan') == 'GAN' else 'unet', |
| python | starter.py | 822 | 'GAN' | 'model_type': 'gan' if match.group('is_gan') == 'GAN' else 'unet', |
| python | starter.py | 822 | 'gan' | 'model_type': 'gan' if match.group('is_gan') == 'GAN' else 'unet', |
| python | starter.py | 822 | 'unet' | 'model_type': 'gan' if match.group('is_gan') == 'GAN' else 'unet', |
| python | starter.py | 823 | 'is_gan' | 'is_gan': match.group('is_gan'), |
| python | starter.py | 824 | 'use_attention' | 'use_attention': match.group('use_attention'), |
| python | starter.py | 825 | 'use_attention' | 'attention': match.group('use_attention') == 'ATTN', |
| python | starter.py | 826 | 'loss_function' | 'loss_function': match.group('loss_function'), |
| python | starter.py | 827 | 'data_alias_enriched_hex' | 'data_alias_enriched_hex': data_alias_enriched_hex, |
| python | starter.py | 828 | 'scaling_tag' | 'scaling_tag': match.group('scaling_tag'), |
| python | starter.py | 829 | 'dropout_tag' | 'dropout_tag': dropout_tag, |
| python | starter.py | 830 | 'dropout_rate' | 'dropout_rate': dropout_rate, |
| python | starter.py | 831 | 'activation_tag' | 'activation_tag': match.group('activation_tag'), |
| python | starter.py | 832 | 'output_activation_tag' | 'output_activation_tag': match.group('output_activation_tag'), |
| python | starter.py | 833 | 'discriminator_activation_tag' | 'discriminator_activation_tag': match.group('discriminator_activation_tag'), |
| python | starter.py | 834 | 'discriminator_output_activation_tag' | 'discriminator_output_activation_tag': match.group('discriminator_output_activation_tag'), |
| python | starter.py | 901 | './' | if result.startswith('./') or result.startswith('../') or result.startswith('/'): |
| python | starter.py | 901 | '../' | if result.startswith('./') or result.startswith('../') or result.startswith('/'): |
| python | starter.py | 901 | '/' | if result.startswith('./') or result.startswith('../') or result.startswith('/'): |
| python | starter.py | 924 | 1 | @lru_cache(maxsize=1) |
| python | starter.py | 994 | 1 | @lru_cache(maxsize=1) |
| python | starter.py | 1005 | 'noise_fn' | if category == 'noise_fn': |
| python | starter.py | 1018 | 'sigma_kernel_fn' | sigma_kernel_name = data_cfg['sigma_kernel_fn'] |
| python | starter.py | 1019 | 'sigma_kernel_requires_fit_data' | data_cfg['sigma_kernel_requires_fit_data'] = sigma_kernel_name == 'fit' |
| python | starter.py | 1020 | 'candidates_fn' | for key in ('candidates_fn', 'post_filter_fn', 'sample_fn', 'sigma_kernel_fn', 'noise_fn', 'stats_name_fn'): |
| python | starter.py | 1020 | 'post_filter_fn' | for key in ('candidates_fn', 'post_filter_fn', 'sample_fn', 'sigma_kernel_fn', 'noise_fn', 'stats_name_fn'): |
| python | starter.py | 1020 | 'sample_fn' | for key in ('candidates_fn', 'post_filter_fn', 'sample_fn', 'sigma_kernel_fn', 'noise_fn', 'stats_name_fn'): |
| python | starter.py | 1020 | 'sigma_kernel_fn' | for key in ('candidates_fn', 'post_filter_fn', 'sample_fn', 'sigma_kernel_fn', 'noise_fn', 'stats_name_fn'): |
| python | starter.py | 1020 | 'noise_fn' | for key in ('candidates_fn', 'post_filter_fn', 'sample_fn', 'sigma_kernel_fn', 'noise_fn', 'stats_name_fn'): |
| python | starter.py | 1020 | 'stats_name_fn' | for key in ('candidates_fn', 'post_filter_fn', 'sample_fn', 'sigma_kernel_fn', 'noise_fn', 'stats_name_fn'): |
| python | starter.py | 1030 | 'output_activation' | if 'output_activation' in network_cfg and network_cfg['output_activation'] is not None: |
| python | starter.py | 1031 | 'output_activation' | network_cfg['output_activation'] = _resolve_output_activation_value(network_cfg['output_activation']) |
| python | starter.py | 1032 | 'kernel_initializer' | if 'kernel_initializer' in network_cfg and network_cfg['kernel_initializer'] is not None: |
| python | starter.py | 1033 | 'kernel_initializer' | network_cfg['kernel_initializer'] = _resolve_initializer_value(network_cfg['kernel_initializer']) |
| python | starter.py | 1043 | 'g_loss_fn' | if 'g_loss_fn' in training_cfg: |
| python | starter.py | 1044 | 'g_loss_fn' | training_cfg['g_loss_fn'] = _resolve_loss_value(training_cfg['g_loss_fn']) |
| python | starter.py | 1045 | 'data_generator' | if 'data_generator' in training_cfg: |
| python | starter.py | 1046 | 'data_generator' | training_cfg['data_generator'] = _resolve_name_from_runtime_registry(training_cfg['data_generator']) |
| python | starter.py | 1048 | 'gan' | if 'gan' in config_data and isinstance(config_data['gan'], dict) and 'loss_fn' in config_data['gan']: |
| python | starter.py | 1048 | 'loss_fn' | if 'gan' in config_data and isinstance(config_data['gan'], dict) and 'loss_fn' in config_data['gan']: |
| python | starter.py | 1049 | 'gan' | gan_cfg = config_data['gan'] |
| python | starter.py | 1050 | 'loss_fn' | gan_cfg['loss_fn'] = _resolve_loss_value(gan_cfg['loss_fn']) |
| python | starter.py | 1075 | 'create_dataset' | dataset_cfg = config_data['create_dataset'] |
| python | starter.py | 1076 | 'new_train' | new_train_cfg = config_data['new_train'] |
| python | starter.py | 1079 | 'gan' | gan_cfg = new_train_cfg['gan'] |
| python | starter.py | 1082 | 'filter_surveys' | filter_surveys = _effective_override_value(config_data, explicit_overrides, 'filter_surveys') |
| python | starter.py | 1083 | 'filter_by_last_name' | filter_by_last_name = _effective_override_value(config_data, explicit_overrides, 'filter_by_last_name') |
| python | starter.py | 1084 | 'last_name_filter_value' | last_name_filter_value = _effective_override_value(config_data, explicit_overrides, 'last_name_filter_value') |
| python | starter.py | 1087 | 'footprint_radius' | footprint_radius = _effective_override_value(config_data, explicit_overrides, 'footprint_radius') |
| python | starter.py | 1092 | 'allowed_survey' | allowed_survey=dataset_cfg['allowed_survey'], |
| python | starter.py | 1099 | 'data_alias_plain' | data_alias_plain = data_dict['data_alias_plain'] |
| python | starter.py | 1100 | 'data_alias_hex' | data_alias_hex = data_dict['data_alias_hex'] |
| python | starter.py | 1101 | 'data_alias_enriched' | data_alias_enriched_plain = data_dict['data_alias_enriched'] |
| python | starter.py | 1102 | 'data_alias_enriched_hex' | data_alias_enriched_hex = data_dict['data_alias_enriched_hex'] |
| python | starter.py | 1104 | 'model_type' | model_type = _effective_override_value(config_data, explicit_overrides, 'model_type') |
| python | starter.py | 1106 | 'gan' | is_gan = 'GAN' if str(model_type).lower() == 'gan' else 'UNET' |
| python | starter.py | 1106 | 'GAN' | is_gan = 'GAN' if str(model_type).lower() == 'gan' else 'UNET' |
| python | starter.py | 1106 | 'UNET' | is_gan = 'GAN' if str(model_type).lower() == 'gan' else 'UNET' |
| python | starter.py | 1109 | 'loss_name' | loss_name = _effective_override_value(config_data, explicit_overrides, 'loss_name') |
| python | starter.py | 1110 | 'dropout_rate' | dropout_rate = _effective_override_value(config_data, explicit_overrides, 'dropout_rate') |
| python | starter.py | 1111 | 'activation_name' | activation_name = _effective_override_value(config_data, explicit_overrides, 'activation_name') |
| python | starter.py | 1112 | 'output_activation' | output_activation = _effective_override_value(config_data, explicit_overrides, 'output_activation') |
| python | starter.py | 1113 | 'discriminator_activation' | discriminator_activation = _effective_override_value(config_data, explicit_overrides, 'discriminator_activation') |
| python | starter.py | 1114 | 'discriminator_output_activation' | discriminator_output_activation = _effective_override_value(config_data, explicit_overrides, 'discriminator_output_activation') |
| python | starter.py | 1118 | 'ssim_loss' | 'ssim_loss':             'SSIM', |
| python | starter.py | 1119 | 'scale_invariant_mae' | 'scale_invariant_mae':   'SIMAE', |
| python | starter.py | 1120 | 'log_cosh_loss' | 'log_cosh_loss':         'LOGCOSH', |
| python | starter.py | 1131 | 'data_dir' | data_dir = f"{config_data['paths']['data_dir']}/{data_alias_enriched_plain}" |
| python | starter.py | 1131 | '/' | data_dir = f"{config_data['paths']['data_dir']}/{data_alias_enriched_plain}" |
| python | starter.py | 1137 | 'model_alias_hex' | model_alias_hex = _model_alias_dict['model_alias_hex'] |
| python | starter.py | 1138 | 'model_alias_plain' | model_alias_plain = _model_alias_dict['model_alias_plain'] |
| python | starter.py | 1139 | 'models_dir' | models_dir = f"""{config_data['paths']['models_dir']}/ |
| python | starter.py | 1139 | '/\n            ' | models_dir = f"""{config_data['paths']['models_dir']}/ |
| python | starter.py | 1139 | '/' | models_dir = f"""{config_data['paths']['models_dir']}/ |
| python | starter.py | 1139 | '/DO' | models_dir = f"""{config_data['paths']['models_dir']}/ |
| python | starter.py | 1139 | '/\n            ACT' | models_dir = f"""{config_data['paths']['models_dir']}/ |
| python | starter.py | 1139 | '/OUT' | models_dir = f"""{config_data['paths']['models_dir']}/ |
| python | starter.py | 1139 | '/\n            DACT' | models_dir = f"""{config_data['paths']['models_dir']}/ |
| python | starter.py | 1139 | '/DOUT' | models_dir = f"""{config_data['paths']['models_dir']}/ |
| python | starter.py | 1147 | 'data_alias_enriched_hex' | 'data_alias_enriched_hex': data_alias_enriched_hex, |
| python | starter.py | 1149 | 'footprint_radius' | 'footprint_radius': footprint_radius, |
| python | starter.py | 1151 | 'data_dir' | 'data_dir': data_dir, |
| python | starter.py | 1152 | 'models_dir' | 'models_dir': models_dir, |
| python | starter.py | 1153 | 'multimodal_metrics_dir' | 'multimodal_metrics_dir': config_data['paths']['multimodal_metrics_dir'], |
| python | starter.py | 1154 | 'singlemodal_metrics_dir' | 'singlemodal_metrics_dir': config_data['paths']['singlemodal_metrics_dir'], |
| python | starter.py | 1155 | 'plots_dir' | 'plots_dir': config_data['paths']['plots_dir'], |
| python | starter.py | 1156 | 'model_type' | 'model_type': model_type, |
| python | starter.py | 1157 | 'is_gan' | 'is_gan': is_gan, |
| python | starter.py | 1158 | 'use_attention' | 'use_attention': use_attention, |
| python | starter.py | 1160 | 'scaling_tag' | 'scaling_tag': scaling_tag, |
| python | starter.py | 1161 | 'loss_name' | 'loss_name': loss_name, |
| python | starter.py | 1162 | 'loss_function' | 'loss_function': loss_function, |
| python | starter.py | 1163 | 'dropout_rate' | 'dropout_rate': dropout_rate, |
| python | starter.py | 1164 | 'dropout_tag' | 'dropout_tag': dropout_tag, |
| python | starter.py | 1165 | 'activation_name' | 'activation_name': activation_name, |
| python | starter.py | 1166 | 'output_activation' | 'output_activation': output_activation, |
| python | starter.py | 1167 | 'discriminator_activation' | 'discriminator_activation': discriminator_activation, |
| python | starter.py | 1168 | 'discriminator_output_activation' | 'discriminator_output_activation': discriminator_output_activation, |
| python | starter.py | 1169 | 'activation_tag' | 'activation_tag': activation_tag, |
| python | starter.py | 1170 | 'output_activation_tag' | 'output_activation_tag': output_activation_tag, |
| python | starter.py | 1171 | 'discriminator_activation_tag' | 'discriminator_activation_tag': discriminator_activation_tag, |
| python | starter.py | 1172 | 'discriminator_output_activation_tag' | 'discriminator_output_activation_tag': discriminator_output_activation_tag, |
| python | starter.py | 1173 | 'model_alias_plain' | 'model_alias_plain': model_alias_plain, |
| python | starter.py | 1174 | 'model_alias_hex' | 'model_alias_hex': model_alias_hex, |
| python | starter.py | 1175 | 'time_tag' | 'time_tag': time_tag, |
| python | starter.py | 1183 | 'data_dir' | for key in ('data_dir', 'models_dir', 'multimodal_metrics_dir', 'singlemodal_metrics_dir', 'plots_dir'): |
| python | starter.py | 1183 | 'models_dir' | for key in ('data_dir', 'models_dir', 'multimodal_metrics_dir', 'singlemodal_metrics_dir', 'plots_dir'): |
| python | starter.py | 1183 | 'multimodal_metrics_dir' | for key in ('data_dir', 'models_dir', 'multimodal_metrics_dir', 'singlemodal_metrics_dir', 'plots_dir'): |
| python | starter.py | 1183 | 'singlemodal_metrics_dir' | for key in ('data_dir', 'models_dir', 'multimodal_metrics_dir', 'singlemodal_metrics_dir', 'plots_dir'): |
| python | starter.py | 1183 | 'plots_dir' | for key in ('data_dir', 'models_dir', 'multimodal_metrics_dir', 'singlemodal_metrics_dir', 'plots_dir'): |
| python | starter.py | 1192 | 'create_dataset' | 'create_dataset', |
| python | starter.py | 1193 | 'new_train' | 'new_train', |
| python | starter.py | 1195 | 'uncropped_metrics' | 'uncropped_metrics', |
| python | starter.py | 1196 | 'merge_catalogs' | 'merge_catalogs', |
| python | starter.py | 1197 | 'prepare_images' | 'prepare_images', |
| python | starter.py | 1198 | 'prepare_plots' | 'prepare_plots', |
| python | starter.py | 1239 | 1 | refs_text = match.group(1).replace(' and ', ',') |
| python | starter.py | 1253 | 0 | while stack and stack[-1][0] >= indent: |
| python | starter.py | 1259 | 1 | key, rest = stripped.split(':', 1) |
| python | starter.py | 1261 | '[A-Za-z_][A-Za-z0-9_]*' | if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key): |
| python | starter.py | 1266 | 1 | value_part, comment = rest.split('#', 1) |
| python | starter.py | 1282 | 8 | @lru_cache(maxsize=8) |
| python | starter.py | 1306 | 1 | leaf = target_path.rsplit('.', 1)[-1] |
| python | starter.py | 1308 | 'create_dataset.split_dirs' | if target_path == 'create_dataset.split_dirs': |
| python | starter.py | 1311 | 'patch_size' | if leaf in {'patch_size', 'uncropped_patch_size'}: |
| python | starter.py | 1311 | 'uncropped_patch_size' | if leaf in {'patch_size', 'uncropped_patch_size'}: |
| python | starter.py | 1312 | 0 | value = source_values[0] |
| python | starter.py | 1317 | 'model_prototype' | if leaf == 'model_prototype': |
| python | starter.py | 1318 | 0 | value = source_values[0] |
| python | starter.py | 1321 | '\\{epoch[^}]*\\}' | return re.sub(r'\{epoch[^}]*\}', '*', value).replace('{prefix}', '*') |
| python | starter.py | 1323 | 1 | if len(source_values) == 1: |
| python | starter.py | 1324 | 0 | return source_values[0] |
| python | starter.py | 1363 | 'checkpoint_filename_pattern' | checkpoint_filename_pattern = training_cfg['checkpoint_filename_pattern'] |
| python | starter.py | 1365 | 'training.checkpoint_filename_pattern must be a non-empty string.' | raise ValueError('training.checkpoint_filename_pattern must be a non-empty string.') |
| python | starter.py | 1367 | 'checkpoint_restore_kwargs' | restore_kwargs = training_cfg['checkpoint_restore_kwargs'] |
| python | starter.py | 1369 | 'training.checkpoint_restore_kwargs must be a mapping.' | raise TypeError('training.checkpoint_restore_kwargs must be a mapping.') |
| python | starter.py | 1372 | 'filename_pattern' | if 'filename_pattern' not in restore_kwargs: |
| python | starter.py | 1373 | "training.checkpoint_restore_kwargs must include 'filename_pattern'." | raise ValueError("training.checkpoint_restore_kwargs must include 'filename_pattern'.") |
| python | starter.py | 1374 | 'filename_pattern' | if restore_kwargs['filename_pattern'] != checkpoint_filename_pattern: |
| python | starter.py | 1375 | 'training.checkpoint_restore_kwargs.filename_pattern must match training.checkpoint_filename_pattern.' | raise ValueError('training.checkpoint_restore_kwargs.filename_pattern must match training.checkpoint_filename_pattern.') |
| python | starter.py | 1377 | 'checkpoint_restore_kwargs' | training_cfg['checkpoint_restore_kwargs'] = restore_kwargs |
| python | starter.py | 1378 | 'checkpoint_restore_kwargs' | data_cfg['checkpoint_restore_kwargs'] = copy.deepcopy(restore_kwargs) |
| python | starter.py | 1379 | 'checkpoint_custom_epoch' | data_cfg['checkpoint_custom_epoch'] = training_cfg['checkpoint_custom_epoch'] |
| python | starter.py | 1388 | 'data_kwargs' | training_cfg['data_kwargs'] = copy.deepcopy(section_cfg['data']) |
| python | starter.py | 1389 | 'network_kwargs' | training_cfg['network_kwargs'] = copy.deepcopy(section_cfg['network']) |
| python | starter.py | 1390 | 'discriminator_kwargs' | training_cfg['discriminator_kwargs'] = copy.deepcopy(section_cfg['discriminator']) |
| python | starter.py | 1391 | 'gan_kwargs' | training_cfg['gan_kwargs'] = copy.deepcopy(section_cfg['gan']) |
| python | starter.py | 1391 | 'gan' | training_cfg['gan_kwargs'] = copy.deepcopy(section_cfg['gan']) |
| python | starter.py | 1395 | 'new_train' | data_cfg = copy.deepcopy(resolved_root['new_train']['data']) |
| python | starter.py | 1396 | 'create_dataset' | create_dataset_cfg = resolved_root['create_dataset'] |
| python | starter.py | 1399 | 'noise_fn' | noise_fn = data_cfg['noise_fn'] |
| python | starter.py | 1401 | 'data_kwargs' | section_cfg['data_kwargs'] = { |
| python | starter.py | 1402 | 'kwargs_data' | 'kwargs_data': data_cfg, |
| python | starter.py | 1404 | 'use_custom_test_images' | if section_cfg['use_custom_test_images']: |
| python | starter.py | 1405 | 'data_kwargs' | section_cfg['data_kwargs']['kwargs_data']['metadata_filepath'] = create_dataset_cfg['noisy_filtered_metadata_output_file'] |
| python | starter.py | 1405 | 'kwargs_data' | section_cfg['data_kwargs']['kwargs_data']['metadata_filepath'] = create_dataset_cfg['noisy_filtered_metadata_output_file'] |
| python | starter.py | 1405 | 'metadata_filepath' | section_cfg['data_kwargs']['kwargs_data']['metadata_filepath'] = create_dataset_cfg['noisy_filtered_metadata_output_file'] |
| python | starter.py | 1405 | 'noisy_filtered_metadata_output_file' | section_cfg['data_kwargs']['kwargs_data']['metadata_filepath'] = create_dataset_cfg['noisy_filtered_metadata_output_file'] |
| python | starter.py | 1407 | 'model_kwargs' | section_cfg['model_kwargs'] = { |
| python | starter.py | 1408 | 'patch_size' | 'patch_size': tuple(section_cfg['patch_size']), |
| python | starter.py | 1411 | 'batch_size' | 'batch_size': section_cfg['batch_size'], |
| python | starter.py | 1412 | 'gaussian_sigma' | 'gaussian_sigma': section_cfg['gaussian_sigma'], |
| python | starter.py | 1413 | 'type_of_image' | 'type_of_image': section_cfg['type_of_image'], |
| python | starter.py | 1414 | 'nan_value' | 'nan_value': data_cfg['nan_value'], |
| python | starter.py | 1415 | 'posinf_value' | 'posinf_value': data_cfg['posinf_value'], |
| python | starter.py | 1416 | 'neginf_value' | 'neginf_value': data_cfg['neginf_value'], |
| python | starter.py | 1417 | 'location_col' | 'location_col': data_cfg['location_col'], |
| python | starter.py | 1418 | 'exp_time_col' | 'exp_time_col': data_cfg['exposure_col'], |
| python | starter.py | 1418 | 'exposure_col' | 'exp_time_col': data_cfg['exposure_col'], |
| python | starter.py | 1419 | 'new_exp_time_col' | 'new_exp_time_col': section_cfg['new_exp_time_col'], |
| python | starter.py | 1420 | 'sigma_key' | 'sigma_key': data_cfg['sigma_key'], |
| python | starter.py | 1421 | 'noise_fn' | 'noise_fn': noise_fn, |
| python | starter.py | 1422 | 'combined_images_dir' | 'combined_images_dir': section_cfg['combined_images_dir'], |
| python | starter.py | 1423 | 'png_dir' | 'png_dir': section_cfg['png_dir'], |
| python | starter.py | 1424 | 'org_dir' | 'org_dir': section_cfg['org_dir'], |
| python | starter.py | 1425 | 'noisy_dir' | 'noisy_dir': section_cfg['noisy_dir'], |
| python | starter.py | 1426 | 'rec_dir' | 'rec_dir': section_cfg['rec_dir'], |
| python | starter.py | 1427 | 'use_mosaic' | 'use_mosaic': section_cfg['use_mosaic'], |
| python | starter.py | 1432 | 'footprint_radius' | 'footprint_radius', 'distance_threshold', 'deblend', 'deblend_timeout', |
| python | starter.py | 1432 | 'distance_threshold' | 'footprint_radius', 'distance_threshold', 'deblend', 'deblend_timeout', |
| python | starter.py | 1432 | 'deblend_timeout' | 'footprint_radius', 'distance_threshold', 'deblend', 'deblend_timeout', |
| python | starter.py | 1433 | 'win_size' | 'alpha', 'beta', 'gamma', 'k1', 'k2', 'win_size', 'win_sigma', |
| python | starter.py | 1433 | 'win_sigma' | 'alpha', 'beta', 'gamma', 'k1', 'k2', 'win_size', 'win_sigma', |
| python | starter.py | 1434 | 'org_thresh' | 'func', 'thresh', 'org_thresh', 'radius_factor', 'PHOT_FLUXFRAC', |
| python | starter.py | 1434 | 'radius_factor' | 'func', 'thresh', 'org_thresh', 'radius_factor', 'PHOT_FLUXFRAC', |
| python | starter.py | 1434 | 'PHOT_FLUXFRAC' | 'func', 'thresh', 'org_thresh', 'radius_factor', 'PHOT_FLUXFRAC', |
| python | starter.py | 1435 | 'r_min' | 'r_min', 'elongation_fraction', 'PHOT_AUTOPARAMS', 'maskthresh', |
| python | starter.py | 1435 | 'elongation_fraction' | 'r_min', 'elongation_fraction', 'PHOT_AUTOPARAMS', 'maskthresh', |
| python | starter.py | 1435 | 'PHOT_AUTOPARAMS' | 'r_min', 'elongation_fraction', 'PHOT_AUTOPARAMS', 'maskthresh', |
| python | starter.py | 1436 | 'org_minarea' | 'minarea', 'org_minarea', 'filter_type', 'deblend_nthresh', 'deblend_cont', |
| python | starter.py | 1436 | 'filter_type' | 'minarea', 'org_minarea', 'filter_type', 'deblend_nthresh', 'deblend_cont', |
| python | starter.py | 1436 | 'deblend_nthresh' | 'minarea', 'org_minarea', 'filter_type', 'deblend_nthresh', 'deblend_cont', |
| python | starter.py | 1436 | 'deblend_cont' | 'minarea', 'org_minarea', 'filter_type', 'deblend_nthresh', 'deblend_cont', |
| python | starter.py | 1437 | 'clean_param' | 'clean', 'clean_param', |
| python | starter.py | 1439 | 'kwargs_source' | section_cfg['kwargs_source'] = {key: copy.deepcopy(section_cfg[key]) for key in source_keys if key in section_cfg} |
| python | starter.py | 1440 | 'kwargs_source' | section_cfg['kwargs_source']['sigma_key'] = data_cfg['sigma_key'] |
| python | starter.py | 1440 | 'sigma_key' | section_cfg['kwargs_source']['sigma_key'] = data_cfg['sigma_key'] |
| python | starter.py | 1441 | 'kwargs_source' | section_cfg['kwargs_source']['noise_fn'] = noise_fn |
| python | starter.py | 1441 | 'noise_fn' | section_cfg['kwargs_source']['noise_fn'] = noise_fn |
| python | starter.py | 1442 | 'kwargs_source' | section_cfg['kwargs_source']['type_of_image'] = section_cfg['type_of_image'] |
| python | starter.py | 1442 | 'type_of_image' | section_cfg['kwargs_source']['type_of_image'] = section_cfg['type_of_image'] |
| python | starter.py | 1443 | 'kwargs_source' | section_cfg['kwargs_source']['nan_value'] = data_cfg['nan_value'] |
| python | starter.py | 1443 | 'nan_value' | section_cfg['kwargs_source']['nan_value'] = data_cfg['nan_value'] |
| python | starter.py | 1444 | 'kwargs_source' | section_cfg['kwargs_source']['posinf_value'] = data_cfg['posinf_value'] |
| python | starter.py | 1444 | 'posinf_value' | section_cfg['kwargs_source']['posinf_value'] = data_cfg['posinf_value'] |
| python | starter.py | 1445 | 'kwargs_source' | section_cfg['kwargs_source']['neginf_value'] = data_cfg['neginf_value'] |
| python | starter.py | 1445 | 'neginf_value' | section_cfg['kwargs_source']['neginf_value'] = data_cfg['neginf_value'] |
| python | starter.py | 1449 | 'kwargs_source' | kwargs_source = copy.deepcopy(section_cfg['kwargs_source']) |
| python | starter.py | 1450 | 'uncropped_patch_size' | kwargs_source['uncropped_patch_size'] = copy.deepcopy(section_cfg['uncropped_patch_size']) |
| python | starter.py | 1451 | 'uncropped_stride' | kwargs_source['uncropped_stride'] = copy.deepcopy(section_cfg['uncropped_stride']) |
| python | starter.py | 1452 | 'uncropped_weighting' | kwargs_source['uncropped_weighting'] = section_cfg['uncropped_weighting'] |
| python | starter.py | 1453 | 'uncropped_batch_size' | kwargs_source['uncropped_batch_size'] = section_cfg['uncropped_batch_size'] |
| python | starter.py | 1454 | 'uncropped_use_mosaic' | kwargs_source['uncropped_use_mosaic'] = section_cfg['uncropped_use_mosaic'] |
| python | starter.py | 1455 | 'kwargs_source' | section_cfg['kwargs_source'] = kwargs_source |
| python | starter.py | 1456 | 'data_kwargs' | section_cfg['data_kwargs'] = copy.deepcopy(section_cfg['data_kwargs']) |
| python | starter.py | 1460 | 'new_train' | 'new_train': _finalize_new_train_section, |
| python | starter.py | 1462 | 'uncropped_metrics' | 'uncropped_metrics': _finalize_uncropped_metrics_section, |
| python | starter.py | 1466 | 'new_train' | 'new_train': ( |
| python | starter.py | 1467 | 'training.data_kwargs' | 'training.data_kwargs', |
| python | starter.py | 1468 | 'training.network_kwargs' | 'training.network_kwargs', |
| python | starter.py | 1469 | 'training.discriminator_kwargs' | 'training.discriminator_kwargs', |
| python | starter.py | 1470 | 'training.gan_kwargs' | 'training.gan_kwargs', |
| python | starter.py | 1473 | 'data_kwargs' | 'data_kwargs', |
| python | starter.py | 1474 | 'model_kwargs' | 'model_kwargs', |
| python | starter.py | 1475 | 'kwargs_source' | 'kwargs_source', |
| python | starter.py | 1477 | 'uncropped_metrics' | 'uncropped_metrics': ( |
| python | starter.py | 1478 | 'kwargs_source' | 'kwargs_source', |
| python | starter.py | 1479 | 'data_kwargs' | 'data_kwargs', |
