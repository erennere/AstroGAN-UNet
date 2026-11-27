# Notes for coding agents

- Package: `src/astrounets`; one command: `astrounets` (see `cli.py`). Steps: fetch, dataset, train,
  evaluate, plots. The study's stages (models, tune, datasets), ranking and tuning choice are in
  `workflow.py`; `workflow/Snakefile` only schedules them; `launch.py` starts Snakemake in the
  background and implements `status` / `stop`. Profiles: `workflow/profiles/{local,slurm}`. On Helix
  everything is a job: `workflow/helix/setup.sbatch` and `workflow/helix/run.sbatch`.
- Configuration: `configs/default.yaml` holds every value; the schema and all validation rules are in
  `config.py`. A new option = a field in the dataclass + a line in `default.yaml` (+ `smoke.yaml` /
  `pilot.yaml` if it should differ there) + a rule in `validate()` if some values are impossible. No
  defaults in code. `dataset.name` selects an entry of `datasets:`, which completes the dataset section.
- Run folders are named by `Config.run_name` and `identity_hash`; settings that do not change what a
  model learns go in `_IDENTITY_IGNORED`.
- Write results with `files.py` (temporary file + rename); never write a result file in place.
- Tests: `python -m pytest` (about ten minutes, CPU). They run on the synthetic sky from
  `data/mock.py`; never on real data. Science invariants live in `tests/test_science.py`; the slow
  tests run the whole workflow locally, through the SLURM profile and as run.sbatch (`tests/fake_slurm.py`).
- Keep the noise, scaling and evaluation paths shared (`data/pairs.py`): training, validation,
  evaluation and plots must use the same functions. Evaluation and `astrounets denoise` reconstruct
  through the same `TrainedModel.reconstruct` (`infer.py`) and fill no-data pixels with `fill_no_data`.
- `infer.py` reads a model's config.yaml key by key, so trained models keep loading when the
  configuration gains fields; keep it that way.
- Figures (`plots.py`, `report.py`) are drawn only from files the stages wrote; never compute a measure
  there that the evaluation does not store. A running study loads its frozen `runs/_workflow/config.yaml`
  in every job: a new config field breaks it, so such changes are pulled on the cluster between studies.
- Code that changes what a run sees (data, noise, scaling) bumps `config.DATA_VERSION`.
- A study can be stopped, updated and started again: new grid values only add runs (existing runs keep
  their identity and resume; `test_a_new_scaling_mode_adds_runs_and_keeps_the_others`), a run's saved
  config may lack fields added since (`train._check_run_folder` fills them in), and the profiles use
  `rerun-triggers: [mtime, input]` so a summary reruns when its set of runs grows. A new scaling mode must
  leave the existing modes' training data byte-identical (`Scaler.fit`'s `valid` is for robust_z only).
