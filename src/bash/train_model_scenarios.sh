#!/bin/bash
#SBATCH --job-name=train-scenarios
#SBATCH --output=../logs/train_model_scenarios_%j.out
#SBATCH --error=../errs/train_model_scenarios_%j.err
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=32gb
#SBATCH --time=96:00:00
#SBATCH --array=1-10%10

# Number of CPU cores per job (N). Keep this in sync with
# '#SBATCH --cpus-per-task' above, or override at submit time with e.g.
# sbatch --cpus-per-task=16 bash/train_model_scenarios.sh
N=8
TOTAL_WORKERS=10

# Project root. Default assumes current working directory is src/.
: "${AUN_PROJECT_ROOT:=$(cd .. && pwd)}"

# Conda environment path (created by setup_hpc_environment.sh)
: "${AUN_ENV_PATH:=${AUN_PROJECT_ROOT}/.venv}"
: "${AUN_CONDA_BASE:=${HOME}/.local/miniconda3}"

if [[ -n "${SLURM_CPUS_PER_TASK}" && "${SLURM_CPUS_PER_TASK}" != "${N}" ]]; then
    echo "WARNING: N=${N} but SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}. Using SLURM_CPUS_PER_TASK."
    N="${SLURM_CPUS_PER_TASK}"
fi

# Activate conda environment from a prefix path
if [[ -f "${AUN_CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${AUN_CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${AUN_ENV_PATH}"
    echo "Activated conda environment: ${CONDA_PREFIX}"
else
    echo "ERROR: Conda base not found at ${AUN_CONDA_BASE}"
    echo "Set AUN_CONDA_BASE or run: bash bash/setup_hpc_environment.sh"
    exit 1
fi

if [[ "${CONDA_PREFIX:-}" != "${AUN_ENV_PATH}" ]]; then
    echo "ERROR: Expected active env ${AUN_ENV_PATH}, got ${CONDA_PREFIX:-<none>}"
    echo "Run: bash bash/setup_hpc_environment.sh"
    exit 1
fi

#module use /gpfs/bwfor/home/hd/hd_hd/hd_nk194/modules/modulefiles
#module load lib/cudnn/9.4.0-cuda-12.6

# Logging destination.
: "${AUN_LOG_DIR:=${AUN_PROJECT_ROOT}/logs}"
mkdir -p "${AUN_LOG_DIR}"

LOG_TS="$(date +%Y%m%d_%H%M%S)"
LOG_JOB_ID="${SLURM_JOB_ID:-local}"
LOG_FILE="${AUN_LOG_DIR}/train_model_scenarios_${LOG_JOB_ID}_${LOG_TS}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

# Full experiment set (cartesian product):
#   model (2) x attention (2) x scaling (4) x loss (4) = 64 scenarios.
#
# Fixed per scenario:
#   - dropout_rate=0.2
#   - kernel_initializer=he_normal
#   - activation_name=LeakyReLU
#   - discriminator_activation=LeakyReLU
#   - discriminator_output_activation=sigmoid
#
# Output activation rule:
#   - ssim_loss => sigmoid
#   - otherwise => null

MODEL_OPTIONS=("unet" "gan")
ATTENTION_OPTIONS=("false" "true")
SCALING_OPTIONS=("z_scale" "min_max" "log_min_max" "null")
LOSS_OPTIONS=("log_cosh_loss" "scale_invariant_mae" "MeanAbsoluteError" "ssim_loss")

MODEL_TYPE=("")
ATTENTION=("")
SCALING=("")
LOSS_NAME=("")
DROPOUT_RATE=("")
OUTPUT_ACTIVATION=("")
KERNEL_INITIALIZER=("")
ACTIVATION_NAME=("")
DISCRIMINATOR_ACTIVATION=("")
DISCRIMINATOR_OUTPUT_ACTIVATION=("")

generate_scenario_matrix() {
    local model
    local attn
    local scaling
    local loss
    local out_act

    for model in "${MODEL_OPTIONS[@]}"; do
        for attn in "${ATTENTION_OPTIONS[@]}"; do
            for scaling in "${SCALING_OPTIONS[@]}"; do
                for loss in "${LOSS_OPTIONS[@]}"; do
                    out_act="null"
                    if [[ "$loss" == "ssim_loss" ]]; then
                        out_act="sigmoid"
                    fi

                    MODEL_TYPE+=("$model")
                    ATTENTION+=("$attn")
                    SCALING+=("$scaling")
                    LOSS_NAME+=("$loss")
                    DROPOUT_RATE+=("0.2")
                    OUTPUT_ACTIVATION+=("$out_act")
                    KERNEL_INITIALIZER+=("he_normal")
                    ACTIVATION_NAME+=("LeakyReLU")
                    DISCRIMINATOR_ACTIVATION+=("LeakyReLU")
                    DISCRIMINATOR_OUTPUT_ACTIVATION+=("sigmoid")
                done
            done
        done
    done
}

print_scenario_catalog() {
    local idx
    local total_scenarios=$(( ${#MODEL_TYPE[@]} - 1 ))
    echo "Scenario catalog (${total_scenarios} total):"
    for idx in $(seq 1 "${total_scenarios}"); do
        echo "  ${idx}) model=${MODEL_TYPE[$idx]} attention=${ATTENTION[$idx]} scaling=${SCALING[$idx]} loss=${LOSS_NAME[$idx]} out_act=${OUTPUT_ACTIVATION[$idx]}"
    done
}

validate_scenario_matrix() {
    local unet_false=0
    local unet_true=0
    local gan_false=0
    local gan_true=0
    local total_scenarios=$(( ${#MODEL_TYPE[@]} - 1 ))

    for idx in $(seq 1 "${total_scenarios}"); do
        if [[ "${MODEL_TYPE[$idx]}" == "unet" && "${ATTENTION[$idx]}" == "false" ]]; then
            ((unet_false++))
        elif [[ "${MODEL_TYPE[$idx]}" == "unet" && "${ATTENTION[$idx]}" == "true" ]]; then
            ((unet_true++))
        elif [[ "${MODEL_TYPE[$idx]}" == "gan" && "${ATTENTION[$idx]}" == "false" ]]; then
            ((gan_false++))
        elif [[ "${MODEL_TYPE[$idx]}" == "gan" && "${ATTENTION[$idx]}" == "true" ]]; then
            ((gan_true++))
        else
            echo "Invalid scenario at index ${idx}: model=${MODEL_TYPE[$idx]} attention=${ATTENTION[$idx]}"
            exit 1
        fi
    done

    if (( unet_false != 16 || unet_true != 16 || gan_false != 16 || gan_true != 16 )); then
        echo "Scenario matrix mismatch. Expected 16 runs for each (model, attention) pair."
        echo "Counts: unet/false=${unet_false}, unet/true=${unet_true}, gan/false=${gan_false}, gan/true=${gan_true}"
        exit 1
    fi

    if (( total_scenarios != 64 )); then
        echo "Scenario matrix mismatch. Expected total 64 scenarios, found ${total_scenarios}."
        exit 1
    fi
}

run_job() {
    local idx=$1

    if [[ -z "${MODEL_TYPE[$idx]}" ]]; then
        echo "Invalid job index: ${idx}"
        exit 1
    fi

    local args=(
        --model-type "${MODEL_TYPE[$idx]}"
        --attention "${ATTENTION[$idx]}"
        --scaling "${SCALING[$idx]}"
        --loss-name "${LOSS_NAME[$idx]}"
        --dropout-rate "${DROPOUT_RATE[$idx]}"
        --output-activation "${OUTPUT_ACTIVATION[$idx]}"
        --kernel-initializer "${KERNEL_INITIALIZER[$idx]}"
        --activation-name "${ACTIVATION_NAME[$idx]}"
        --discriminator-activation "${DISCRIMINATOR_ACTIVATION[$idx]}"
        --discriminator-output-activation "${DISCRIMINATOR_OUTPUT_ACTIVATION[$idx]}"
    )

    echo "Project root : ${AUN_PROJECT_ROOT}"
    echo "Conda env    : ${AUN_ENV_PATH}"
    echo "Cores/job (N): ${N}"
    echo "GPU request  : gpu:1 (constraint=gpu4)"
    echo "Log file     : ${LOG_FILE}"
    echo "[Job ${idx}] model=${MODEL_TYPE[$idx]} attention=${ATTENTION[$idx]} scaling=${SCALING[$idx]} loss=${LOSS_NAME[$idx]} dropout=${DROPOUT_RATE[$idx]} out_act=${OUTPUT_ACTIVATION[$idx]} init=${KERNEL_INITIALIZER[$idx]} act=${ACTIVATION_NAME[$idx]} d_act=${DISCRIMINATOR_ACTIVATION[$idx]} d_out=${DISCRIMINATOR_OUTPUT_ACTIVATION[$idx]}"
    echo "[Job ${idx}] srun --ntasks=1 --cpus-per-task=${N} --cpu-bind=cores python -m src.training.new_train ${args[*]}"

    cd "${AUN_PROJECT_ROOT}/src"
    export PYTHONPATH="${AUN_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    export OMP_NUM_THREADS="${N}"
    export MKL_NUM_THREADS="${N}"
    export OPENBLAS_NUM_THREADS="${N}"
    srun --ntasks=1 --cpus-per-task="${N}" --cpu-bind=cores \
        python -m src.training.new_train "${args[@]}"
}

run_worker() {
    local worker_id=$1
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

if [[ -z "${SLURM_ARRAY_TASK_ID}" || "${SLURM_ARRAY_TASK_ID}" == "1" ]]; then
    print_scenario_catalog
fi

# Dispatch
if [[ -n "${SLURM_ARRAY_TASK_ID}" ]]; then
    run_worker "${SLURM_ARRAY_TASK_ID}"
elif [[ -n "$1" ]]; then
    run_worker "$1"
else
    for worker in $(seq 1 "${TOTAL_WORKERS}"); do
        run_worker "${worker}"
    done
fi
