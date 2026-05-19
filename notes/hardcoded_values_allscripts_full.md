# Hardcoded Values Audit (All Scripts, Full)

| Category | Lang | File | Line | Hardcoded value | Context |
| --- | --- | --- | ---: | --- | --- |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 18 | 2 | pool_size = 2 |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 19 | 0.02 | deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02)) |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 32 | 1 | conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 32 | 'g_conv1_1' | conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 33 | 1 | conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 33 | 'g_conv1_2' | conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 36 | 1 | conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 36 | 'g_conv2_1' | conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 37 | 1 | conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 37 | 'g_conv2_2' | conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 40 | 1 | conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 40 | 'g_conv3_1' | conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 41 | 1 | conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 41 | 'g_conv3_2' | conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 44 | 1 | conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 44 | 'g_conv4_1' | conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 45 | 1 | conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 45 | 'g_conv4_2' | conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 48 | 1 | conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 48 | 'g_conv5_1' | conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 49 | 1 | conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 49 | 'g_conv5_2' | conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 52 | 1 | conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 52 | 'g_conv6_1' | conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 53 | 1 | conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 53 | 'g_conv6_2' | conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 56 | 1 | conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 56 | 'g_conv7_1' | conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 57 | 1 | conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 57 | 'g_conv7_2' | conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 60 | 1 | conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 60 | 'g_conv8_1' | conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 61 | 1 | conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 61 | 'g_conv8_2' | conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 64 | 1 | conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 64 | 'g_conv9_1' | conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 65 | 1 | conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2') |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 65 | 'g_conv9_2' | conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2') |
| literal | python | legacy/create_new_image.py | 67 | 1 | conv10 = slim.conv2d(conv9, 1, [1, 1], rate=1, activation_fn=None, scope='g_conv10') |
| literal | python | legacy/create_new_image.py | 67 | 'g_conv10' | conv10 = slim.conv2d(conv9, 1, [1, 1], rate=1, activation_fn=None, scope='g_conv10') |
| literal | python | legacy/create_new_image.py | 135 | 256 | ps = 256 |
| literal | python | legacy/create_new_image.py | 144 | 32 | step = 32  # you can change your step, max_step = ps (256) |
| literal | python | legacy/create_new_image.py | 162 | 2 | ratio = 2 |
| hyperparameter-or-threshold | python | legacy/create_new_image.py | 166 | 0 | in_patch = np.expand_dims(in_patch,  axis=0) |
| literal | python | legacy/create_new_image.py | 178 | 'Greys_r' | plt.imshow(output, cmap='Greys_r', origin='lower', norm=norm) |
| literal | python | legacy/create_new_image.py | 180 | 600 | plt.savefig(save_name + ".png", dpi=600, bbox_inches='tight', pad_inches=0) |
| literal | python | legacy/create_new_image.py | 197 | 2 | ratio = 2 |
| literal | python | legacy/down_data.py | 14 | 3 | obsTable = Observations.query_criteria(calib_level= 3, dataproduct_type = 'image', |
| path-or-filename | python | legacy/original_data_split.py | 75 | './original_dataset' | txt_filepath = './original_dataset' |
| path-or-filename | python | legacy/original_data_split.py | 76 | './data' | image_filepath = './data' |
| path-or-filename | python | legacy/original_down_data.py | 7 | './original_dataset' | directory = './original_dataset' |
| path-or-filename | python | legacy/original_images.py | 185 | './original_dataset' | save_dir = './original_dataset' |
| literal | python | legacy/original_images.py | 188 | 'sci_data_set_name' | column = 'sci_data_set_name' |
| literal | python | legacy/original_images.py | 190 | 10000 | max_requests = 10000 |
| literal | python | legacy/original_images.py | 191 | 10 | reset_after = 10 |
| hyperparameter-or-threshold | python | legacy/original_images.py | 192 | 32 | max_workers = 32 |
| literal | python | legacy/original_images.py | 194 | 60 | period = 60 |
| literal | python | legacy/original_images.py | 195 | 10 | step = 10 |
| path-or-filename | python | legacy/original_images.py | 213 | './hst_wfc3_f160W_metadata.csv' | output_filepath = './hst_wfc3_f160W_metadata.csv' |
| path-or-filename | python | legacy/train_test_val_split.py | 8 | './data' | data_directory = './data' |
| path-or-filename | python | legacy/train_test_val_split.py | 9 | './original_dataset' | destination = './original_dataset' |
| scheduler-resource | bash | src/bash/check_gpu_hpc.sh | 2 | #SBATCH --job-name=check-gpu | #SBATCH --job-name=check-gpu |
| scheduler-resource | bash | src/bash/check_gpu_hpc.sh | 3 | #SBATCH --output=../logs/check_gpu_%j.out | #SBATCH --output=../logs/check_gpu_%j.out |
| scheduler-resource | bash | src/bash/check_gpu_hpc.sh | 4 | #SBATCH --error=../errs/check_gpu_%j.err | #SBATCH --error=../errs/check_gpu_%j.err |
| scheduler-resource | bash | src/bash/check_gpu_hpc.sh | 5 | #SBATCH --partition=gpu-single | #SBATCH --partition=gpu-single |
| scheduler-resource | bash | src/bash/check_gpu_hpc.sh | 6 | #SBATCH --gres=gpu:1 | #SBATCH --gres=gpu:1 |
| scheduler-resource | bash | src/bash/check_gpu_hpc.sh | 7 | #SBATCH --mem=2gb | #SBATCH --mem=2gb |
| scheduler-resource | bash | src/bash/check_gpu_hpc.sh | 8 | #SBATCH --time=00:30:00 | #SBATCH --time=00:30:00 |
| literal | bash | src/bash/check_gpu_hpc.sh | 68 | importlib.import_module("tensorflow") | tf = importlib.import_module("tensorflow") |
| literal | bash | src/bash/check_gpu_hpc.sh | 70 | tf.config.list_physical_devices("GPU") | gpus = tf.config.list_physical_devices("GPU") |
| literal | bash | src/bash/check_gpu_hpc.sh | 77 | importlib.import_module("torch") | torch = importlib.import_module("torch") |
| literal | bash | src/bash/check_gpu_hpc.sh | 86 | len(result["tensorflow_visible_gpus"]) > 0 | tf_ok = len(result["tensorflow_visible_gpus"]) > 0 |
| literal | bash | src/bash/check_gpu_hpc.sh | 87 | result["torch_cuda_available"] and result["torch_device_count"] > 0 | torch_ok = result["torch_cuda_available"] and result["torch_device_count"] > 0 |
| scheduler-resource | bash | src/bash/mast_and_create_dataset.sh | 2 | #SBATCH --job-name=mast-create-dataset | #SBATCH --job-name=mast-create-dataset |
| scheduler-resource | bash | src/bash/mast_and_create_dataset.sh | 3 | #SBATCH --output=../logs/mast_and_create_dataset_%j.out | #SBATCH --output=../logs/mast_and_create_dataset_%j.out |
| scheduler-resource | bash | src/bash/mast_and_create_dataset.sh | 4 | #SBATCH --error=../errs/mast_and_create_dataset_%j.err | #SBATCH --error=../errs/mast_and_create_dataset_%j.err |
| scheduler-resource | bash | src/bash/mast_and_create_dataset.sh | 5 | #SBATCH --partition=cpu-single | #SBATCH --partition=cpu-single |
| scheduler-resource | bash | src/bash/mast_and_create_dataset.sh | 6 | #SBATCH --cpus-per-task=16 | #SBATCH --cpus-per-task=16 |
| scheduler-resource | bash | src/bash/mast_and_create_dataset.sh | 7 | #SBATCH --mem=64gb | #SBATCH --mem=64gb |
| scheduler-resource | bash | src/bash/mast_and_create_dataset.sh | 8 | #SBATCH --time=96:00:00 | #SBATCH --time=96:00:00 |
| scheduler-resource | bash | src/bash/mast_and_create_dataset.sh | 9 | #SBATCH --array=1-3 | #SBATCH --array=1-3 |
| hyperparameter-or-threshold | bash | src/bash/mast_and_create_dataset.sh | 37 | (                          ""        2      2      2      ) | NSIGMA=(                          ""        2      2      2      ) |
| hyperparameter-or-threshold | bash | src/bash/mast_and_create_dataset.sh | 38 | (                       ""        10     10     10     ) | FP_RADIUS=(                       ""        10     10     10     ) |
| hyperparameter-or-threshold | bash | src/bash/mast_and_create_dataset.sh | 39 | (                         ""        8      8      8      ) | NPIXELS=(                         ""        8      8      8      ) |
| literal | bash | src/bash/mast_and_create_dataset.sh | 40 | (                  ""        true   true   true   ) | FILTER_SURVEYS=(                  ""        true   true   true   ) |
| literal | bash | src/bash/mast_and_create_dataset.sh | 41 | (             ""        false  true   true   ) | FILTER_BY_LAST_NAME=(             ""        false  true   true   ) |
| literal | bash | src/bash/mast_and_create_dataset.sh | 52 | "$(date +%Y%m%d_%H%M%S)" | LOG_TS="$(date +%Y%m%d_%H%M%S)" |
| scheduler-resource | bash | src/bash/mast_and_create_dataset.sh | 53 | "${SLURM_JOB_ID:-local}" | LOG_JOB_ID="${SLURM_JOB_ID:-local}" |
| path-or-filename | bash | src/bash/mast_and_create_dataset.sh | 54 | "${AUN_LOG_DIR}/mast_and_create_dataset_${LOG_JOB_ID}_${LOG_TS}.log" | LOG_FILE="${AUN_LOG_DIR}/mast_and_create_dataset_${LOG_JOB_ID}_${LOG_TS}.log" |
| literal | bash | src/bash/setup_hpc_environment.sh | 32 | "$(cd .. && pwd)" | PROJECT_ROOT="$(cd .. && pwd)" |
| path-or-filename | bash | src/bash/setup_hpc_environment.sh | 33 | "$PROJECT_ROOT/logs" | LOGS_DIR="$PROJECT_ROOT/logs" |
| path-or-filename | bash | src/bash/setup_hpc_environment.sh | 36 | "${HOME}/.local/miniconda3" | CONDA_INSTALL_PREFIX="${HOME}/.local/miniconda3" |
| path-or-filename | bash | src/bash/setup_hpc_environment.sh | 38 | "${PROJECT_ROOT}/.venv"  # Use .venv for compatibility with existing scripts | ENV_PATH="${PROJECT_ROOT}/.venv"  # Use .venv for compatibility with existing scripts |
| literal | bash | src/bash/setup_hpc_environment.sh | 39 | 0 | CLEAN_START=0 |
| literal | bash | src/bash/setup_hpc_environment.sh | 42 | "latest"  # or use specific version like "24.11.2-0" | MINICONDA_VERSION="latest"  # or use specific version like "24.11.2-0" |
| literal | bash | src/bash/setup_hpc_environment.sh | 44 | "x86_64" | MINICONDA_ARCH="x86_64" |
| path-or-filename | bash | src/bash/setup_hpc_environment.sh | 45 | "Miniconda3-${MINICONDA_VERSION}-${MINICONDA_OS}-${MINICONDA_ARCH}.sh" | MINICONDA_INSTALLER="Miniconda3-${MINICONDA_VERSION}-${MINICONDA_OS}-${MINICONDA_ARCH}.sh" |
| path-or-filename | bash | src/bash/setup_hpc_environment.sh | 46 | "https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}" | MINICONDA_URL="https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}" |
| path-or-filename | bash | src/bash/setup_hpc_environment.sh | 50 | "$LOGS_DIR/setup_hpc_$(date +%Y%m%d_%H%M%S).log" | LOG_FILE="$LOGS_DIR/setup_hpc_$(date +%Y%m%d_%H%M%S).log" |
| literal | bash | src/bash/setup_hpc_environment.sh | 56 | "$2" | CONDA_INSTALL_PREFIX="$2" |
| literal | bash | src/bash/setup_hpc_environment.sh | 60 | 1 | CLEAN_START=1 |
| scheduler-resource | bash | src/bash/setup_hpc_environment_slurm.sh | 20 | #SBATCH --job-name=setup-astro-hpc | #SBATCH --job-name=setup-astro-hpc |
| scheduler-resource | bash | src/bash/setup_hpc_environment_slurm.sh | 21 | #SBATCH --output=../logs/setup_astro_hpc_%j.out | #SBATCH --output=../logs/setup_astro_hpc_%j.out |
| scheduler-resource | bash | src/bash/setup_hpc_environment_slurm.sh | 22 | #SBATCH --error=../errs/setup_astro_hpc_%j.err | #SBATCH --error=../errs/setup_astro_hpc_%j.err |
| scheduler-resource | bash | src/bash/setup_hpc_environment_slurm.sh | 23 | #SBATCH --partition=cpu-single | #SBATCH --partition=cpu-single |
| scheduler-resource | bash | src/bash/setup_hpc_environment_slurm.sh | 24 | #SBATCH --nodes=1 | #SBATCH --nodes=1 |
| scheduler-resource | bash | src/bash/setup_hpc_environment_slurm.sh | 25 | #SBATCH --cpus-per-task=16 | #SBATCH --cpus-per-task=16 |
| scheduler-resource | bash | src/bash/setup_hpc_environment_slurm.sh | 26 | #SBATCH --mem=16GB | #SBATCH --mem=16GB |
| scheduler-resource | bash | src/bash/setup_hpc_environment_slurm.sh | 27 | #SBATCH --time=02:00:00 | #SBATCH --time=02:00:00 |
| literal | bash | src/bash/setup_hpc_environment_slurm.sh | 43 | "$(cd .. && pwd)" | PROJECT_ROOT="$(cd .. && pwd)" |
| scheduler-resource | bash | src/bash/train_model_scenarios.sh | 2 | #SBATCH --job-name=train-scenarios | #SBATCH --job-name=train-scenarios |
| scheduler-resource | bash | src/bash/train_model_scenarios.sh | 3 | #SBATCH --output=../logs/train_model_scenarios_%j.out | #SBATCH --output=../logs/train_model_scenarios_%j.out |
| scheduler-resource | bash | src/bash/train_model_scenarios.sh | 4 | #SBATCH --error=../errs/train_model_scenarios_%j.err | #SBATCH --error=../errs/train_model_scenarios_%j.err |
| scheduler-resource | bash | src/bash/train_model_scenarios.sh | 5 | #SBATCH --partition=gpu-single | #SBATCH --partition=gpu-single |
| scheduler-resource | bash | src/bash/train_model_scenarios.sh | 6 | #SBATCH --nodes=1 | #SBATCH --nodes=1 |
| scheduler-resource | bash | src/bash/train_model_scenarios.sh | 7 | #SBATCH --ntasks=1 | #SBATCH --ntasks=1 |
| scheduler-resource | bash | src/bash/train_model_scenarios.sh | 8 | #SBATCH --cpus-per-task=8 | #SBATCH --cpus-per-task=8 |
| scheduler-resource | bash | src/bash/train_model_scenarios.sh | 9 | #SBATCH --gres=gpu:A40:1 | #SBATCH --gres=gpu:A40:1 |
| scheduler-resource | bash | src/bash/train_model_scenarios.sh | 10 | #SBATCH --mem=32gb | #SBATCH --mem=32gb |
| scheduler-resource | bash | src/bash/train_model_scenarios.sh | 11 | #SBATCH --time=96:00:00 | #SBATCH --time=96:00:00 |
| scheduler-resource | bash | src/bash/train_model_scenarios.sh | 12 | #SBATCH --array=1-10%10 | #SBATCH --array=1-10%10 |
| literal | bash | src/bash/train_model_scenarios.sh | 17 | 8 | N=8 |
| hyperparameter-or-threshold | bash | src/bash/train_model_scenarios.sh | 18 | 10 | TOTAL_WORKERS=10 |
| scheduler-resource | bash | src/bash/train_model_scenarios.sh | 29 | "${SLURM_CPUS_PER_TASK}" | N="${SLURM_CPUS_PER_TASK}" |
| literal | bash | src/bash/train_model_scenarios.sh | 56 | "$(date +%Y%m%d_%H%M%S)" | LOG_TS="$(date +%Y%m%d_%H%M%S)" |
| scheduler-resource | bash | src/bash/train_model_scenarios.sh | 57 | "${SLURM_JOB_ID:-local}" | LOG_JOB_ID="${SLURM_JOB_ID:-local}" |
| path-or-filename | bash | src/bash/train_model_scenarios.sh | 58 | "${AUN_LOG_DIR}/train_model_scenarios_${LOG_JOB_ID}_${LOG_TS}.log" | LOG_FILE="${AUN_LOG_DIR}/train_model_scenarios_${LOG_JOB_ID}_${LOG_TS}.log" |
| literal | bash | src/bash/train_model_scenarios.sh | 76 | ("false" "true") | ATTENTION_OPTIONS=("false" "true") |
| literal | bash | src/bash/train_model_scenarios.sh | 218 | 1 | ran=1 |
| literal | python | src/data/create_dataset.py | 60 | 30 | plt.hist(df[exp_column], bins=30, color='skyblue', edgecolor='black', label=f'N: {len(df)}') |
| literal | python | src/data/create_dataset.py | 66 | 2 | plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val}') |
| literal | python | src/data/create_dataset.py | 67 | 2 | plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val}') |
| literal | python | src/data/create_dataset.py | 68 | 2 | plt.axvline(mean_val + std_val, color='orange', linestyle='dashed', linewidth=2, label=f'Std Dev: {std_val}') |
| literal | python | src/data/create_dataset.py | 69 | 2 | plt.axvline(mean_val - std_val, color='orange', linestyle='dashed', linewidth=2) |
| literal | python | src/data/mast.py | 43 | 5000 | limit = 5000 |
| hyperparameter-or-threshold | python | src/data/mast.py | 136 | 0.75 | counts, bins_edges, patches = plt.hist(data, bins=bins, density=False, alpha=0.75, color=c, edgecolor='black', label=label) |
| literal | python | src/data/mast.py | 141 | '--' | plt.axvline(mean, color='r', linestyle='--', label='mean') |
| literal | python | src/data/mast.py | 142 | '--' | plt.axvline(median, color='purple', linestyle='--', label='median') |
| literal | python | src/data/mast.py | 150 | 300 | plt.savefig(output_filename, dpi=300) |
| scheduler-resource | python | src/data/mast.py | 186 | 15 | session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) |
| scheduler-resource | python | src/data/mast.py | 231 | 15 | session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) |
| hyperparameter-or-threshold | python | src/data/mast.py | 250 | 1 | chunk_size = 1 |
| literal | python | src/data/mast.py | 437 | 1 | for idx, id_chunk in enumerate(chunks, start=1): |
| literal | python | src/data/mast.py | 459 | 1 | for idx, id_chunk in enumerate(chunks, start=1) |
| literal | python | src/evaluation/metrics.py | 471 | 300 | plt.savefig(filepath, dpi=300, bbox_inches='tight') |
| literal | python | src/evaluation/metrics.py | 546 | 1.5 | edgecolor='blue', facecolor='none', linewidth=1.5) |
| literal | python | src/evaluation/metrics.py | 553 | 1.5 | edgecolor='red', facecolor='none', linewidth=1.5) |
| literal | python | src/evaluation/metrics.py | 571 | 1.5 | edgecolor='green', facecolor='none', linewidth=1.5) |
| literal | python | src/evaluation/metrics.py | 578 | 1.5 | edgecolor='red', facecolor='none', linewidth=1.5) |
| literal | python | src/evaluation/metrics.py | 585 | 2 | Line2D([0], [0], color='red', lw=2, label='Matched Sources'), |
| literal | python | src/evaluation/metrics.py | 586 | 2 | Line2D([0], [0], color='blue', lw=2, label='Unmatched Sources (Original)'), |
| literal | python | src/evaluation/metrics.py | 587 | 2 | Line2D([0], [0], color='green', lw=2, label='Unmatched Sources (Reconstructed)') |
| literal | python | src/evaluation/metrics.py | 591 | 3 | ncol=3, frameon=False) |
| literal | python | src/evaluation/metrics.py | 594 | 300 | plt.savefig(filepath, dpi=300, bbox_inches='tight') |
| hyperparameter-or-threshold | python | src/evaluation/metrics.py | 882 | 1 | with ThreadPoolExecutor(max_workers=1) as executor: |
| hyperparameter-or-threshold | python | src/evaluation/metrics.py | 1017 | 1 | PHOT_AUTOPARAMS*kronrad, subpix=1, err=rms_map) |
| hyperparameter-or-threshold | python | src/evaluation/metrics.py | 1028 | 5 | r, rflag = sep.flux_radius(data_sub, x, y, radius_factor * a, PHOT_FLUXFRAC, normflux=flux, subpix=5) |
| literal | python | src/evaluation/metrics.py | 1033 | 3.0 | sep.mask_ellipse(mask, x, y, a, b, theta, r=3.) |
| hyperparameter-or-threshold | python | src/evaluation/metrics.py | 2346 | 3 | overrides = parse_config_overrides(start_index=3)  # sys.argv[1]=index, sys.argv[2]=concurrent_workers, flags start at 3 |
| literal | python | src/evaluation/uncropped_metrics.py | 284 | 0.5 | alpha=0.5, |
| literal | python | src/evaluation/uncropped_metrics.py | 307 | 300 | plt.savefig(hist_png, dpi=300) |
| hyperparameter-or-threshold | python | src/evaluation/uncropped_metrics.py | 342 | 1 | concurrent_workers=1 |
| hyperparameter-or-threshold | python | src/training/callback.py | 275 | 0 | selected_validation_size = 0 |
| hyperparameter-or-threshold | python | src/training/callback.py | 276 | 0 | n_batches_to_use = 0 |
| hyperparameter-or-threshold | python | src/training/callback.py | 295 | 0 | num_batches = 0 |
| literal | python | src/training/math_helpers.py | 703 | 80.0 | log_domain_data = np.clip(log_domain_data, a_min=None, a_max=80.0) |
| hyperparameter-or-threshold | python | src/training/new_train.py | 685 | 2 | batch_size = 2 |
| literal | python | src/training/new_train.py | 703 | 0 | start_epoch = 0 |
| literal | python | src/training/new_train.py | 733 | 0 | start_epoch = 0 |
| hyperparameter-or-threshold | python | src/training/new_train.py | 793 | 0 | train_dataset_size = 0 |
| hyperparameter-or-threshold | python | src/training/new_train.py | 794 | 0 | train_batches = 0 |
| hyperparameter-or-threshold | python | src/training/new_train.py | 799 | 0 | validation_dataset_size = 0 |
| hyperparameter-or-threshold | python | src/training/new_train.py | 800 | 0 | valid_batches = 0 |
| path-or-filename | python | src/training/utils.py | 229 | 'checkpoint_info.json' | CHECKPOINT_INFO_FILENAME = 'checkpoint_info.json' |
| literal | python | src/training/utils.py | 262 | 2 | json.dumps(checkpoint_info, indent=2, default=str), |
| literal | python | src/training/utils.py | 393 | 0 | start_epoch = 0 |
| hyperparameter-or-threshold | python | src/training/utils.py | 550 | 1 | return tf.nn.conv2d(t, kernel, strides=1, padding='SAME') |
| literal | python | src/training/utils.py | 604 | 7 | metadata_shape = 7 |
| literal | python | src/training/utils.py | 606 | 5 | metadata_shape = 5 |
| literal | python | src/training/utils.py | 985 | 9 | stop = 9 |
| literal | python | src/training/utils.py | 1198 | 'exponent_diff' | base='base', exponent='exponent_diff', noise_ratio=None) |
| literal | python | src/training/utils.py | 1217 | 'exp_ratio' | base=None, exponent=None, noise_ratio='exp_ratio') |
| literal | python | src/training/utils.py | 1237 | 'exp_ratio' | return filtering_df_v2(info, x, col_A='exp_ratio', |
| literal | python | src/training/utils.py | 1238 | 'sm_peak_NSR' | col_B='sm_peak_NSR', col_C='dataset', col_D='location', |
| literal | python | src/visualization/prepare_images.py | 38 | 4 | columns = 4 |
| hyperparameter-or-threshold | python | src/visualization/prepare_images.py | 141 | 16 | ax0.set_title(f"{label}", fontsize=16) |
| literal | python | src/visualization/prepare_images.py | 143 | 500 | fig.savefig(output_filepath, dpi=500) |
| literal | python | src/visualization/prepare_images.py | 201 | 1.5 | edgecolor='blue', facecolor='none', linewidth=1.5) |
| literal | python | src/visualization/prepare_images.py | 208 | 1.5 | edgecolor='red', facecolor='none', linewidth=1.5) |
| literal | python | src/visualization/prepare_images.py | 216 | 1.5 | edgecolor='green', facecolor='none', linewidth=1.5) |
| literal | python | src/visualization/prepare_images.py | 223 | 1.5 | edgecolor='red', facecolor='none', linewidth=1.5) |
| literal | python | src/visualization/prepare_images.py | 231 | 1.5 | edgecolor='yellow', facecolor='none', linewidth=1.5) |
| literal | python | src/visualization/prepare_images.py | 238 | 1.5 | edgecolor='red', facecolor='none', linewidth=1.5) |
| hyperparameter-or-threshold | python | src/visualization/prepare_images.py | 402 | 16 | ax0.set_title(f"{label}", fontsize=16) |
| literal | python | src/visualization/prepare_images.py | 404 | 2 | Line2D([0], [0], color='red', lw=2, label='Matched Sources'), |
| literal | python | src/visualization/prepare_images.py | 405 | 2 | Line2D([0], [0], color='blue', lw=2, label='Unmatched Sources (Original)'), |
| literal | python | src/visualization/prepare_images.py | 406 | 2 | Line2D([0], [0], color='green', lw=2, label='Unmatched Sources (Reconstructed)'), |
| literal | python | src/visualization/prepare_images.py | 407 | 2 | Line2D([0], [0], color='yellow', lw=2, label='Unmatched Sources (Noisy)') |
| literal | python | src/visualization/prepare_images.py | 411 | 3 | ncol=3, frameon=False) |
| literal | python | src/visualization/prepare_images.py | 413 | 500 | fig.savefig(output_filepath, dpi=500) |
| hyperparameter-or-threshold | python | src/visualization/prepare_images.py | 508 | 1 | concurrent_workers=1 |
| literal | python | src/visualization/prepare_plots.py | 143 | 33356.4 | factor = 3.33564e4 |
| literal | python | src/visualization/prepare_plots.py | 311 | 0.5 | ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5) |
| literal | python | src/visualization/prepare_plots.py | 319 | 2 | mincnt=2, |
| literal | python | src/visualization/prepare_plots.py | 320 | 0.1 | linewidths=0.1, |
| hyperparameter-or-threshold | python | src/visualization/prepare_plots.py | 338 | 4 | gradient_line = mlines.Line2D([0], [0], color=color, lw=4, label=legend_label) |
| hyperparameter-or-threshold | python | src/visualization/prepare_plots.py | 354 | 20 | fig.suptitle(suptitle, fontsize=20) |
| literal | python | src/visualization/prepare_plots.py | 356 | 0.2 | fig.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92) |
| literal | python | src/visualization/prepare_plots.py | 356 | 0.3 | fig.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92) |
| literal | python | src/visualization/prepare_plots.py | 356 | 0.92 | fig.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92) |
| path-or-filename | python | src/visualization/prepare_plots.py | 654 | 'Org. Flux $[erg\\ s^{-1}\\ cm^{-2}\\ \\AA^{-1}]$' | flux_label = r'Org. Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$' |
| path-or-filename | python | src/visualization/prepare_plots.py | 655 | 'Rec. Flux $[erg\\ s^{-1}\\ cm^{-2}\\ \\AA^{-1}]$' | rec_flux_label = r'Rec. Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$' |
| path-or-filename | python | src/visualization/prepare_plots.py | 656 | 'Noisy Flux $[erg\\ s^{-1}\\ cm^{-2}\\ \\AA^{-1}]$' | noisy_flux_label = r'Noisy Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$' |
| hyperparameter-or-threshold | python | src/visualization/prepare_plots.py | 657 | 'Org. Kron Radius $[arcsec]$' | kron_label = r'Org. Kron Radius $[arcsec]$' |
| path-or-filename | python | src/visualization/prepare_plots.py | 660 | '$\\Delta$Kron Radius $[arcsec]$' | delta_kron_label = r'$\Delta$Kron Radius $[arcsec]$' |
| literal | python | src/visualization/prepare_plots.py | 1093 | 0.2 | ax.fill_between(gt_x, gt_p25, gt_p75, color="black", alpha=0.2) |
| literal | python | src/visualization/prepare_plots.py | 1097 | 0.2 | ax.fill_between(noisy_x, noisy_p25, noisy_p75, color="darkblue", alpha=0.2) |
| literal | python | src/visualization/prepare_plots.py | 1101 | 0.2 | ax.fill_between(rec_x, rec_p25, rec_p75, color="darkgreen", alpha=0.2) |
| literal | python | src/visualization/prepare_plots.py | 1104 | '--' | ax.grid(True, which="both", linestyle="--", alpha=0.3) |
| literal | python | src/visualization/prepare_plots.py | 1104 | 0.3 | ax.grid(True, which="both", linestyle="--", alpha=0.3) |
| hyperparameter-or-threshold | python | src/visualization/prepare_plots.py | 1115 | 16 | fig.suptitle(r"SNR vs. Flux", fontsize=16) |
| literal | python | src/visualization/prepare_plots.py | 1117 | 500 | plt.savefig(output_filepath, dpi=500) |
| scheduler-resource | python | src/visualization/prepare_plots.py | 1185 | 'exp_ratio' | bright_metrics = new_metrics(bright_df, exp_time_col='exp_ratio') |
| path-or-filename | python | src/visualization/prepare_plots.py | 1209 | 'snr.png' | snr_filename = 'snr.png' |
| hyperparameter-or-threshold | python | starter.py | 138 | '--nsigma' | flag='--nsigma', |
| hyperparameter-or-threshold | python | starter.py | 145 | 'footprint_radius' | key='footprint_radius', |
| hyperparameter-or-threshold | python | starter.py | 146 | '--footprint-radius' | flag='--footprint-radius', |
| hyperparameter-or-threshold | python | starter.py | 154 | '--npixels' | flag='--npixels', |
| literal | python | starter.py | 155 | 2 | positional_index=2, |
| literal | python | starter.py | 161 | 'model_type' | key='model_type', |
| literal | python | starter.py | 162 | '--model-type' | flag='--model-type', |
| literal | python | starter.py | 163 | 3 | positional_index=3, |
| literal | python | starter.py | 170 | '--attention' | flag='--attention', |
| literal | python | starter.py | 171 | 4 | positional_index=4, |
| literal | python | starter.py | 178 | '--scaling' | flag='--scaling', |
| literal | python | starter.py | 179 | 5 | positional_index=5, |
| literal | python | starter.py | 185 | 'loss_name' | key='loss_name', |
| literal | python | starter.py | 186 | '--loss-name' | flag='--loss-name', |
| literal | python | starter.py | 187 | 6 | positional_index=6, |
| hyperparameter-or-threshold | python | starter.py | 193 | 'dropout_rate' | key='dropout_rate', |
| hyperparameter-or-threshold | python | starter.py | 194 | '--dropout-rate' | flag='--dropout-rate', |
| literal | python | starter.py | 195 | 7 | positional_index=7, |
| literal | python | starter.py | 201 | 'output_activation' | key='output_activation', |
| literal | python | starter.py | 202 | '--output-activation' | flag='--output-activation', |
| literal | python | starter.py | 203 | 8 | positional_index=8, |
| literal | python | starter.py | 209 | 'kernel_initializer' | key='kernel_initializer', |
| literal | python | starter.py | 210 | '--kernel-initializer' | flag='--kernel-initializer', |
| literal | python | starter.py | 211 | 9 | positional_index=9, |
| literal | python | starter.py | 217 | 'activation_name' | key='activation_name', |
| literal | python | starter.py | 218 | '--activation-name' | flag='--activation-name', |
| literal | python | starter.py | 219 | 10 | positional_index=10, |
| literal | python | starter.py | 225 | 'discriminator_activation' | key='discriminator_activation', |
| literal | python | starter.py | 226 | '--discriminator-activation' | flag='--discriminator-activation', |
| literal | python | starter.py | 227 | 11 | positional_index=11, |
| literal | python | starter.py | 233 | 'discriminator_output_activation' | key='discriminator_output_activation', |
| literal | python | starter.py | 234 | '--discriminator-output-activation' | flag='--discriminator-output-activation', |
| literal | python | starter.py | 235 | 12 | positional_index=12, |
| literal | python | starter.py | 241 | 'filter_surveys' | key='filter_surveys', |
| literal | python | starter.py | 242 | '--filter-surveys' | flag='--filter-surveys', |
| literal | python | starter.py | 243 | 13 | positional_index=13, |
| literal | python | starter.py | 249 | 'filter_by_last_name' | key='filter_by_last_name', |
| literal | python | starter.py | 250 | '--filter-by-last-name' | flag='--filter-by-last-name', |
| literal | python | starter.py | 251 | 14 | positional_index=14, |
| literal | python | starter.py | 257 | 'last_name_filter_value' | key='last_name_filter_value', |
| literal | python | starter.py | 258 | '--last-name-filter-value' | flag='--last-name-filter-value', |
| literal | python | starter.py | 259 | 15 | positional_index=15, |
| hyperparameter-or-threshold | python | starter.py | 276 | 1 | @lru_cache(maxsize=1) |
| literal | python | starter.py | 415 | 9 | compressed = zlib.compress(raw, level=9) |
| literal | python | starter.py | 453 | 9 | compressed = zlib.compress(raw, level=9) |
| literal | python | starter.py | 475 | 2 | cursor = 2 |
| literal | python | starter.py | 572 | 9 | compressed = zlib.compress(raw, level=9) |
| hyperparameter-or-threshold | python | starter.py | 924 | 1 | @lru_cache(maxsize=1) |
| hyperparameter-or-threshold | python | starter.py | 994 | 1 | @lru_cache(maxsize=1) |
| hyperparameter-or-threshold | python | starter.py | 1282 | 8 | @lru_cache(maxsize=8) |
