#!/bin/bash
#SBATCH --job-name=train-scenarios-last-name
#SBATCH --output=../logs/train_model_scenarios_last_name_%j.out
#SBATCH --error=../errs/train_model_scenarios_last_name_%j.err
#SBATCH --partition=gpu-single
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=234gb
#SBATCH --time=96:00:00
#SBATCH --array=1-6%6

set -euo pipefail

N=64
TOTAL_WORKERS=6

source bash/utils.sh

init_aun_paths
setup_aun_logging "train_model_scenarios_last_name"
configure_aun_threads "${N}"
N="${AUN_EFFECTIVE_THREADS}"
activate_aun_env
echo "Activated conda environment: ${CONDA_PREFIX}"
require_aun_env_active
enter_aun_src

# 4 model/loss scenarios x 3 data scenarios = 12 total:
#  1) attention=true,  loss=scale_invariant_mae, model=unet
#  2) attention=true,  loss=log_cosh_loss,       model=unet
#  3) attention=false, loss=scale_invariant_mae, model=unet
#  4) attention=false, loss=log_cosh_loss,       model=unet
# Data scenario variants:
#  - filter_by_last_name=false
#  - filter_by_last_name=true,  last_name_filter_value=FABER
#  - filter_by_last_name=true,  last_name_filter_value=MOMCHEVA

MODEL_TYPE=("")
ATTENTION=("")
LOSS_NAME=("")
FILTER_BY_LAST_NAME=("")
LAST_NAME_FILTER_VALUE=("")

generate_scenario_matrix() {
    local attention
    local loss

    local data_filter_options=("false" "true" "true")
    local data_last_name_options=("" "FABER" "MOMCHEVA")

    for attention in "true" "false"; do
        for loss in "scale_invariant_mae" "log_cosh_loss"; do
            for i in "${!data_filter_options[@]}"; do
                MODEL_TYPE+=("unet")
                ATTENTION+=("${attention}")
                LOSS_NAME+=("${loss}")
                FILTER_BY_LAST_NAME+=("${data_filter_options[$i]}")
                LAST_NAME_FILTER_VALUE+=("${data_last_name_options[$i]}")
            done
        done
    done
}

scenario_data_label() {
    local idx="$1"
    if [[ "${FILTER_BY_LAST_NAME[$idx]}" == "true" ]]; then
        echo "last_name=${LAST_NAME_FILTER_VALUE[$idx]}"
    else
        echo "last_name=off"
    fi
}

print_scenario_catalog() {
    local idx
    local total_scenarios=$(( ${#MODEL_TYPE[@]} - 1 ))

    echo "Scenario catalog (${total_scenarios} total):"
    for idx in $(seq 1 "${total_scenarios}"); do
        echo "  ${idx}) model=${MODEL_TYPE[$idx]} attention=${ATTENTION[$idx]} loss=${LOSS_NAME[$idx]} $(scenario_data_label "${idx}")"
    done
}

validate_scenario_matrix() {
    local idx
    local total_scenarios=$(( ${#MODEL_TYPE[@]} - 1 ))
    local count_true_simae=0
    local count_true_logcosh=0
    local count_false_simae=0
    local count_false_logcosh=0

    for idx in $(seq 1 "${total_scenarios}"); do
        if [[ "${MODEL_TYPE[$idx]}" != "unet" ]]; then
            echo "Invalid model at scenario ${idx}: ${MODEL_TYPE[$idx]}"
            exit 1
        fi

        if [[ "${ATTENTION[$idx]}" == "true" && "${LOSS_NAME[$idx]}" == "scale_invariant_mae" ]]; then
            ((++count_true_simae))
        elif [[ "${ATTENTION[$idx]}" == "true" && "${LOSS_NAME[$idx]}" == "log_cosh_loss" ]]; then
            ((++count_true_logcosh))
        elif [[ "${ATTENTION[$idx]}" == "false" && "${LOSS_NAME[$idx]}" == "scale_invariant_mae" ]]; then
            ((++count_false_simae))
        elif [[ "${ATTENTION[$idx]}" == "false" && "${LOSS_NAME[$idx]}" == "log_cosh_loss" ]]; then
            ((++count_false_logcosh))
        else
            echo "Invalid attention/loss pair at scenario ${idx}: attention=${ATTENTION[$idx]} loss=${LOSS_NAME[$idx]}"
            exit 1
        fi

        if [[ "${FILTER_BY_LAST_NAME[$idx]}" == "true" ]]; then
            if [[ "${LAST_NAME_FILTER_VALUE[$idx]}" != "FABER" && "${LAST_NAME_FILTER_VALUE[$idx]}" != "MOMCHEVA" ]]; then
                echo "Invalid last-name filter at scenario ${idx}: ${LAST_NAME_FILTER_VALUE[$idx]}"
                exit 1
            fi
        elif [[ "${FILTER_BY_LAST_NAME[$idx]}" == "false" ]]; then
            if [[ -n "${LAST_NAME_FILTER_VALUE[$idx]}" ]]; then
                echo "Unexpected last-name value when filter is off at scenario ${idx}: ${LAST_NAME_FILTER_VALUE[$idx]}"
                exit 1
            fi
        else
            echo "Invalid filter_by_last_name at scenario ${idx}: ${FILTER_BY_LAST_NAME[$idx]}"
            exit 1
        fi
    done

    if (( total_scenarios != 12 )); then
        echo "Scenario matrix mismatch. Expected total 12 scenarios, found ${total_scenarios}."
        exit 1
    fi

    if (( count_true_simae != 3 || count_true_logcosh != 3 || count_false_simae != 3 || count_false_logcosh != 3 )); then
        echo "Scenario matrix mismatch. Each attention/loss pair must appear exactly 3 times (one per data scenario)."
        echo "Counts: true+simae=${count_true_simae}, true+logcosh=${count_true_logcosh}, false+simae=${count_false_simae}, false+logcosh=${count_false_logcosh}"
        exit 1
    fi
}

run_job() {
    local idx="$1"

    if [[ -z "${MODEL_TYPE[$idx]}" ]]; then
        echo "Invalid job index: ${idx}"
        exit 1
    fi

    local args=(
        --model-type "${MODEL_TYPE[$idx]}"
        --attention "${ATTENTION[$idx]}"
        --loss-name "${LOSS_NAME[$idx]}"
        --filter-by-last-name "${FILTER_BY_LAST_NAME[$idx]}"
    )

    if [[ "${FILTER_BY_LAST_NAME[$idx]}" == "true" ]]; then
        args+=(--last-name-filter-value "${LAST_NAME_FILTER_VALUE[$idx]}")
    fi

    echo "Project root : ${AUN_PROJECT_ROOT}"
    echo "Conda env    : ${AUN_ENV_PATH}"
    echo "Cores/job (N): ${N}"
    echo "GPU request  : gpu:1"
    echo "Log file     : ${AUN_LOG_FILE}"
    echo "[Job ${idx}] model=${MODEL_TYPE[$idx]} attention=${ATTENTION[$idx]} loss=${LOSS_NAME[$idx]} $(scenario_data_label "${idx}")"
    echo "[Job ${idx}] srun --ntasks=1 --cpus-per-task=${N} --cpu-bind=cores python -m src.training.new_train ${args[*]}"

    srun --ntasks=1 --cpus-per-task="${N}" --cpu-bind=cores \
        python -m src.training.new_train "${args[@]}"
}

run_worker() {
    local worker_id="$1"
    local total_scenarios=$(( ${#MODEL_TYPE[@]} - 1 ))
    local ran=0

    if (( worker_id < 1 || worker_id > TOTAL_WORKERS )); then
        echo "Invalid worker id: ${worker_id}. Expected 1..${TOTAL_WORKERS}."
        exit 1
    fi

    echo "[Worker ${worker_id}] deterministic sharding across ${TOTAL_WORKERS} workers"
    for idx in $(seq 1 "${total_scenarios}"); do
        if (( ((idx - 1) % TOTAL_WORKERS) + 1 == worker_id )); then
            run_job "${idx}"
            ran=1
        fi
    done

    if (( ran == 0 )); then
        echo "[Worker ${worker_id}] no assigned scenarios"
    fi
}

generate_scenario_matrix
validate_scenario_matrix

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: This script is array-only. Submit with sbatch so SLURM_ARRAY_TASK_ID is set."
    echo "Example: sbatch bash/train_model_scenarios_last_name.sh"
    exit 1
fi

if [[ "${SLURM_ARRAY_TASK_ID}" == "1" ]]; then
    print_scenario_catalog
fi

run_worker "${SLURM_ARRAY_TASK_ID}"