#!/bin/bash

AUN_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


init_aun_paths() {
    : "${AUN_PROJECT_ROOT:=$(cd "${AUN_UTILS_DIR}/../.." && pwd)}"
    : "${AUN_ENV_PATH:=${AUN_PROJECT_ROOT}/.venv}"
    : "${AUN_CONDA_BASE:=${HOME}/.local/miniconda3}"
    : "${AUN_LOG_DIR:=${AUN_PROJECT_ROOT}/logs}"
}


setup_aun_logging() {
    local log_name="$1"

    init_aun_paths
    mkdir -p "${AUN_LOG_DIR}"

    local log_ts
    local log_job_id
    local log_file
    log_ts="$(date +%Y%m%d_%H%M%S)"
    log_job_id="${SLURM_JOB_ID:-local}"
    log_file="${AUN_LOG_DIR}/${log_name}_${log_job_id}_${log_ts}.log"
    AUN_LOG_FILE="${log_file}"
    exec > >(tee -a "${log_file}") 2>&1
}


activate_aun_env() {
    init_aun_paths

    if [[ -f "${AUN_CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        source "${AUN_CONDA_BASE}/etc/profile.d/conda.sh"
        conda activate "${AUN_ENV_PATH}"
    else
        echo "ERROR: Conda base not found at ${AUN_CONDA_BASE}"
        exit 1
    fi
}


require_aun_env_active() {
    init_aun_paths

    if [[ "${CONDA_PREFIX:-}" != "${AUN_ENV_PATH}" ]]; then
        echo "ERROR: Expected active env ${AUN_ENV_PATH}, got ${CONDA_PREFIX:-<none>}"
        echo "Run: bash bash/setup_hpc_environment.sh"
        exit 1
    fi
}


enter_aun_src() {
    init_aun_paths
    cd "${AUN_PROJECT_ROOT}/src"

    if [[ ! -f "${AUN_PROJECT_ROOT}/src/starter.py" ]]; then
        echo "ERROR: Missing required file ${AUN_PROJECT_ROOT}/src/starter.py"
        exit 1
    fi
    if [[ ! -f "${AUN_PROJECT_ROOT}/src/config_parsing.py" ]]; then
        echo "ERROR: Missing required file ${AUN_PROJECT_ROOT}/src/config_parsing.py"
        exit 1
    fi

    export PYTHONPATH="${AUN_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

    python - <<'PY'
import inspect
import os

import starter
import config_parsing

starter_file = os.path.abspath(inspect.getfile(starter)).replace('\\', '/')
config_parsing_file = os.path.abspath(inspect.getfile(config_parsing)).replace('\\', '/')

if not starter_file.endswith('/src/starter.py'):
    raise SystemExit(
        f"ERROR: starter resolved to {starter_file}, expected .../src/starter.py"
    )

if not config_parsing_file.endswith('/src/config_parsing.py'):
    raise SystemExit(
        f"ERROR: config_parsing resolved to {config_parsing_file}, expected .../src/config_parsing.py"
    )

required_starter_symbols = ('load_config', 'parse_config_overrides', '_decode_models_dir')
missing = [name for name in required_starter_symbols if not hasattr(starter, name)]
if missing:
    raise SystemExit(
        f"ERROR: {starter_file} is missing required symbols: {', '.join(missing)}"
    )
PY
}


configure_aun_threads() {
    local requested_threads="$1"
    local effective_threads="${requested_threads}"

    if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${SLURM_CPUS_PER_TASK}" != "${requested_threads}" ]]; then
        echo "WARNING: N=${requested_threads} but SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}. Using SLURM_CPUS_PER_TASK."
        effective_threads="${SLURM_CPUS_PER_TASK}"
    fi

    export OMP_NUM_THREADS="${effective_threads}"
    export MKL_NUM_THREADS="${effective_threads}"
    export OPENBLAS_NUM_THREADS="${effective_threads}"
    AUN_EFFECTIVE_THREADS="${effective_threads}"
}


build_data_args() {
    local filter_by_last_name="$1"
    local last_name_filter_value="${2:-}"

    DATA_ARGS=(
        --nsigma "2"
        --footprint-radius "10"
        --npixels "8"
        --filter-surveys "true"
        --filter-by-last-name "${filter_by_last_name}"
    )

    if [[ "${filter_by_last_name}" == "true" ]]; then
        DATA_ARGS+=(--last-name-filter-value "${last_name_filter_value}")
    fi
}


build_default_data_args() {
    build_data_args "true" "MOMCHEVA"
}


build_uncropped_data_args() {
    build_data_args "true" "FABER"
}


build_model_scenario_args() {
    local model="$1"
    local attention="$2"
    local scaling="$3"
    local loss="$4"
    local output_activation="null"

    if [[ "${loss}" == "ssim_loss" ]]; then
        output_activation="sigmoid"
    fi

    MODEL_ARGS=(
        --model-type "${model}"
        --attention "${attention}"
        --scaling "${scaling}"
        --loss-name "${loss}"
        --dropout-rate "0.2"
        --output-activation "${output_activation}"
        --kernel-initializer "he_normal"
        --activation-name "LeakyReLU"
        --discriminator-activation "LeakyReLU"
        --discriminator-output-activation "sigmoid"
    )
}


init_aun_runtime_selectors() {
    : "${AUN_DATA_ALIAS_HEX:=eNpzYWRl-M_AxMDFwAEAClcBXg}"
    : "${AUN_MODEL_ALIAS_HEX:=null}"
    : "${AUN_EPOCH_SELECTOR:=null}"
}


assert_last_name_scenario_matches_training_matrix() {
    local attention="$1"
    local loss="$2"
    local filter_by_last_name="$3"
    local last_name_filter_value="$4"
    local data_alias_selector="$5"
    local model_alias_selector="$6"

    python - "${attention}" "${loss}" "${filter_by_last_name}" "${last_name_filter_value}" "${data_alias_selector}" "${model_alias_selector}" <<'PY'
import sys
import starter

attention = sys.argv[1]
loss = sys.argv[2]
filter_by_last_name = sys.argv[3]
last_name_filter_value = sys.argv[4]
data_alias_selector = sys.argv[5]
model_alias_selector = sys.argv[6]

CANONICAL_ATTENTION = ('true', 'false')
CANONICAL_LOSS = ('scale_invariant_mae', 'log_cosh_loss')
CANONICAL_DATA = (
    ('false', ''),
    ('true', 'FABER'),
    ('true', 'MOMCHEVA'),
)

expected = {}
for attn in CANONICAL_ATTENTION:
    for loss_name in CANONICAL_LOSS:
        for data_filter, data_last_name in CANONICAL_DATA:
            kwargs = {
                'nsigma': '2',
                'footprint_radius': '10',
                'npixels': '8',
                'model_type': 'unet',
                'attention': attn,
                'scaling': 'z_scale',
                'loss_name': loss_name,
                'filter_surveys': 'true',
                'filter_by_last_name': data_filter,
            }
            if data_filter == 'true':
                kwargs['last_name_filter_value'] = data_last_name

            cfg = starter.load_config(**kwargs)
            data_alias = cfg['metrics']['data_kwargs']['kwargs_data']['data_alias_enriched_hex']
            model_alias = cfg['metrics']['data_kwargs']['kwargs_data']['model_alias_hex']
            if not data_alias or not model_alias:
                raise SystemExit(
                    f'ERROR: failed to resolve aliases for canonical scenario '
                    f'attn={attn} loss={loss_name} filter_by_last_name={data_filter} '
                    f'last_name={data_last_name or "off"}.'
                )

            key = (attn, loss_name, data_filter, data_last_name)
            if key in expected:
                raise SystemExit(f'ERROR: duplicate canonical key generated: {key}')
            expected[key] = (data_alias, model_alias)

if len(expected) != 12:
    raise SystemExit(f'ERROR: canonical scenario size mismatch: expected 12, got {len(expected)}')

input_key = (
    attention,
    loss,
    filter_by_last_name,
    last_name_filter_value if filter_by_last_name == 'true' else '',
)

if input_key not in expected:
    raise SystemExit(
        f'ERROR: scenario not in canonical training matrix: '
        f'attn={attention} loss={loss} filter_by_last_name={filter_by_last_name} '
        f'last_name={last_name_filter_value or "off"}'
    )

expected_data_alias, expected_model_alias = expected[input_key]
if expected_data_alias != data_alias_selector or expected_model_alias != model_alias_selector:
    raise SystemExit(
        'ERROR: selector mismatch against canonical training matrix. '
        f'Expected data_alias={expected_data_alias} model_alias={expected_model_alias}; '
        f'got data_alias={data_alias_selector} model_alias={model_alias_selector}'
    )
PY
}