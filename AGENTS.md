# Notes for coding agents

- Package: `src/astrounets`; one command: `astrounets` (see `cli.py`). Steps: fetch, dataset, train,
  evaluate, plots. The study's stages (models, tune, datasets), ranking and tuning choice are in
  `workflow.py`; `workflow/Snakefile` only schedules them; `launch.py` starts Snakemake in the
  background and implements `status` / `stop`. Profiles: `workflow/profiles/{local,slurm}`.
- Configuration: `configs/default.yaml` holds every value; the schema and all validation rules are in
  `config.py`. A new option = a field in the dataclass + a line in `default.yaml` (+ `smoke.yaml` /
  `pilot.yaml` if it should differ there) + a rule in `validate()` if some values are impossible. No
  defaults in code. `dataset.name` selects an entry of `datasets:`, which completes the dataset section.
- Run folders are named by `Config.run_name` and `identity_hash`; settings that do not change what a
  model learns go in `_IDENTITY_IGNORED`.
- Write results with `files.py` (temporary file + rename); never write a result file in place.
- Tests: `python -m pytest` (about eight minutes, CPU). They run on the synthetic sky from
  `data/mock.py`; never on real data. Science invariants live in `tests/test_science.py`; the two slow
  tests run the whole workflow locally and through the SLURM profile (`tests/fake_slurm.py`).
- Keep the noise, scaling and evaluation paths shared (`data/pairs.py`): training, validation,
  evaluation and plots must use the same functions.
