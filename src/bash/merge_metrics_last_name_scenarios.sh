#!/bin/bash
#SBATCH --job-name=merge-metrics-last-name-scenarios
#SBATCH --output=../logs/merge_metrics_last_name_scenarios_%j.out
#SBATCH --error=../errs/merge_metrics_last_name_scenarios_%j.err
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=01:00:00
#SBATCH --array=1-6%6

# Array-only script. 12 merge-metrics scenarios matching train_model_scenarios_last_name.sh:
#   4 model combos (attention x loss) x 3 data scenarios (last_name off / FABER / MOMCHEVA)
# Divided across 6 array jobs (2 scenarios each, deterministic sharding).

set -euo pipefail

MERGE_CONCURRENT_WORKERS=6

source bash/utils.sh

init_aun_paths
setup_aun_logging "merge_metrics_last_name_scenarios"
activate_aun_env
enter_aun_src
init_aun_runtime_selectors

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: This script is array-only. Submit with sbatch so SLURM_ARRAY_TASK_ID is set."
    echo "Example: sbatch bash/merge_metrics_last_name_scenarios.sh"
    exit 1
fi

MERGE_INDEX="${SLURM_ARRAY_TASK_ID}"
EPOCH_SELECTOR="${AUN_EPOCH_SELECTOR}"

ATTENTION_OPTS=("true" "false")
LOSS_OPTS=("scale_invariant_mae" "log_cosh_loss")
DATA_FILTER_OPTS=("false" "true" "true")
DATA_LAST_NAME_OPTS=("" "FABER" "MOMCHEVA")

run_one() {
    local attention="$1"
    local loss="$2"
    local filter_by_last_name="$3"
    local last_name_filter_value="$4"
    local data_alias_selector
    local model_alias_selector

    build_data_args "${filter_by_last_name}" "${last_name_filter_value}"
    build_model_scenario_args "unet" "${attention}" "z_scale" "${loss}"

    read -r data_alias_selector model_alias_selector < <(
        python - "${attention}" "${loss}" "${filter_by_last_name}" "${last_name_filter_value}" <<'PY'
import sys
import starter

attention = sys.argv[1]
loss = sys.argv[2]
filter_by_last_name = sys.argv[3]
last_name_filter_value = sys.argv[4]

kwargs = {
    'nsigma': '2',
    'footprint_radius': '10',
    'npixels': '8',
    'model_type': 'unet',
    'attention': attention,
    'scaling': 'z_scale',
    'loss_name': loss,
    'filter_surveys': 'true',
    'filter_by_last_name': filter_by_last_name,
}

if filter_by_last_name == 'true':
    kwargs['last_name_filter_value'] = last_name_filter_value

cfg = starter.load_config(**kwargs)
data_alias = cfg['metrics']['data_kwargs']['kwargs_data']['data_alias_enriched_hex']
model_alias = cfg['metrics']['data_kwargs']['kwargs_data']['model_alias_hex']

if not data_alias or not model_alias:
    raise SystemExit('ERROR: failed to resolve scenario data/model aliases from config.')

print(f"{data_alias} {model_alias}")
PY
    )

    local data_label
    if [[ "${filter_by_last_name}" == "true" ]]; then
        data_label="last_name=${last_name_filter_value}"
    else
        data_label="last_name=off"
    fi

    assert_last_name_scenario_matches_training_matrix \
        "${attention}" "${loss}" "${filter_by_last_name}" "${last_name_filter_value}" \
        "${data_alias_selector}" "${model_alias_selector}"

    echo "[merge_metrics] idx=${MERGE_INDEX} cw=${MERGE_CONCURRENT_WORKERS} model=unet attn=${attention} scaling=z_scale loss=${loss} ${data_label} data_alias=${data_alias_selector} model_alias=${model_alias_selector}"

    python -m src.evaluation.merge_metrics \
        "${data_alias_selector}" "${model_alias_selector}" "${EPOCH_SELECTOR}" \
        "${DATA_ARGS[@]}" "${MODEL_ARGS[@]}"
}

scenario_idx=0
for attention in "${ATTENTION_OPTS[@]}"; do
    for loss in "${LOSS_OPTS[@]}"; do
        for i in "${!DATA_FILTER_OPTS[@]}"; do
            (( ++scenario_idx ))
            worker=$(( ((scenario_idx - 1) % MERGE_CONCURRENT_WORKERS) + 1 ))
            if (( worker == MERGE_INDEX )); then
                run_one "${attention}" "${loss}" "${DATA_FILTER_OPTS[$i]}" "${DATA_LAST_NAME_OPTS[$i]}"
            fi
        done
    done
done
