# Bash Orchestration

`src/bash/` contains cluster-oriented wrappers for environment setup, dataset preparation jobs, training scenario sweeps, and GPU checks.

These scripts do not replace Python entrypoints; they orchestrate them with reproducible job tables and scheduler integration.

## Scripts

| Script | Purpose |
| --- | --- |
| `setup_hpc_environment.sh` | Install Miniconda, create/update project conda env at project `.venv` |
| `setup_hpc_environment_slurm.sh` | Submit environment setup as a SLURM batch job |
| `check_gpu_hpc.sh` | Validate GPU visibility (`nvidia-smi`, TensorFlow, PyTorch) |
| `migrate_models_dir_layout.sh` | Dry-run or execute the models directory layout migration under SLURM |
| `mast_and_create_dataset.sh` | Run stage 1+2 job table (`mast` and `create_dataset`) |
| `train_model_scenarios.sh` | Execute deterministic training scenario matrix across workers |
| `metrics_single_run.sh` | Single-run `metrics` (`index=0`, `concurrent_workers=1`) |
| `uncropped_metrics_single_run.sh` | Single-run `uncropped_metrics` |
| `merge_catalogs_single_run.sh` | Single-run `merge_catalogs` |
| `prepare_images_single_run.sh` | Single-run `prepare_images` |
| `prepare_plots_single_run.sh` | Single-run `prepare_plots` |
| `metrics_scenarios.sh` | 10-array model-parameter sweep for `metrics` |
| `uncropped_metrics_scenarios.sh` | 10-array model-parameter sweep for `uncropped_metrics` |
| `merge_catalogs_scenarios.sh` | 10-array model-parameter sweep for `merge_catalogs` |
| `prepare_images_scenarios.sh` | 10-array model-parameter sweep for `prepare_images` |
| `prepare_plots_scenarios.sh` | 10-array model-parameter sweep for `prepare_plots` |
| `utils.sh` | Shared bash helpers for bootstrap, logging, thread env, and fixed stage args |

## Orchestration Flow

```mermaid
flowchart LR
	setup[setup_hpc_environment.sh] --> env[project conda env]
	env --> check[check_gpu_hpc.sh]
	env --> data[mast_and_create_dataset.sh]
	data --> train[train_model_scenarios.sh]
	train --> metrics[metrics_scenarios.sh]
	metrics --> uncropped[uncropped_metrics_scenarios.sh]
	uncropped --> merge[merge_catalogs_scenarios.sh]
	merge --> prepimg[prepare_images_scenarios.sh]
	prepimg --> prepplots[prepare_plots_scenarios.sh]
	slurm[setup_hpc_environment_slurm.sh] --> setup
```

## Execution Modes

### Local mode
Most scripts can run locally (without SLURM array variables). Example: `mast_and_create_dataset.sh` loops all job rows if no task ID is provided.

### SLURM mode
Scripts with `#SBATCH` headers can be submitted directly and use scheduler variables such as:

- `SLURM_ARRAY_TASK_ID`
- `SLURM_CPUS_PER_TASK`
- `SLURM_JOB_ID`

## Examples

From `src/`:

```bash
bash bash/setup_hpc_environment.sh --clean-start
sbatch bash/setup_hpc_environment_slurm.sh
sbatch bash/check_gpu_hpc.sh
sbatch bash/migrate_models_dir_layout.sh /gpfs/lsdf02/sd17f001/eren/network/models
AUN_MIGRATE_EXECUTE=true sbatch bash/migrate_models_dir_layout.sh /gpfs/lsdf02/sd17f001/eren/network/models
bash bash/mast_and_create_dataset.sh
sbatch bash/mast_and_create_dataset.sh
sbatch bash/train_model_scenarios.sh
```

## Environment Variables

Common runtime overrides used by scripts:

- `AUN_PROJECT_ROOT`
- `AUN_ENV_PATH`
- `AUN_CONDA_BASE`
- `AUN_LOG_DIR`
- `AUN_RUN_MAST`
- `AUN_RUN_CREATE_DATASET`

Example:

```bash
export AUN_PROJECT_ROOT=/path/to/astroUnets
export AUN_CONDA_BASE=$HOME/.local/miniconda3
bash bash/mast_and_create_dataset.sh
```
