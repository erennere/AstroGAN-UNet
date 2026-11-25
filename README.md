# astroUnets

Denoising of HST WFC3/IR F160W drizzled images with a U-Net or a GAN (U-Net generator + conditional
PatchGAN). Deep exposures are the targets; shorter exposures are simulated by adding noise. Models are
judged on what matters for photometry: flux accuracy, source shapes, image structure, completeness and purity.

## Install

| Where | Command |
| --- | --- |
| Windows (CPU) | `powershell -ExecutionPolicy Bypass -File setup_env.ps1` |
| Linux / WSL | `bash setup_env.sh` (`CPU_ONLY=1` for a CPU-only TensorFlow) |
| bwForCluster Helix | `module load devel/python/<3.11 or 3.12>` then `bash setup_env.sh` on a login node |

Each creates `.venv/` in the repository and installs the `astrounets` command with pinned versions
(TensorFlow 2.20, Snakemake 9). Activate it (`source .venv/bin/activate` or `.\.venv\Scripts\Activate.ps1`).

## Run the study

One command, the same on the laptop and on Helix:

```
astrounets workflow      # start (or continue) everything; runs in the background
astrounets status        # progress of each stage + the end of the log
astrounets stop          # stop; `astrounets workflow` later continues where it stopped
```

`astrounets workflow` runs [Snakemake](https://snakemake.readthedocs.io) in the background, so you
can close the console (on Windows the laptop must stay awake). If `sbatch` is available (Helix login
node) every training run is its own SLURM GPU job and dataset builds are CPU jobs; otherwise everything
runs on this machine, one job at a time (`--local` forces that). Starting again skips finished runs and
resumes interrupted ones from their last epoch. Useful options: `--dry-run` (list the jobs),
`--foreground` (stay attached), `--until data|models|tune|datasets` (stop after a stage),
`-c <config>` and `--set key=value` (as everywhere).

The study, defined in the `workflow:` block of `configs/default.yaml`:

1. **fetch**: one MAST query, sky tiles, splits, downloads (on Helix it runs on the login node, which
   has internet).
2. **datasets**: `all`, `faber`, `momcheva` (defined under `datasets:`), each built once.
3. **models** (stage 1): the 56 valid combinations of model type, attention, scaling and loss, 20 epochs
   each on `all`, ranked on the validation split (see *Ranking*).
4. **tune**: the best `workflow.tune.top` setups (3), each trained with every rate in
   `workflow.tune.learning_rates` at a constant rate until the validation loss has not improved for
   `patience` epochs. Per setup the rate with the lowest validation loss wins; its stage-2 epoch budget
   is `epoch_margin` x the epoch of that best loss, kept within `min_epochs` and `max_epochs`.
5. **datasets** (stage 2): each tuned setup trained on `all`, `faber` and `momcheva` with its own rate
   and epochs (the rate halves after 75 % of the epochs, `train.lr_drops`), tested on the test split of
   all three. Final table: `runs/_workflow/datasets/results.csv`.

Cluster resources (partition, `gpu:A40:1`, time, memory, how many jobs at once) are in
`workflow/profiles/slurm/config.yaml`; the local profile is next to it. Inside SLURM jobs the images a
run reads are first copied to the node-local `$TMPDIR` (`paths.node_scratch`). Every result is written
to a temporary file and renamed when complete, so a killed job never leaves half a file behind.

## Checks

```
python -m pytest                                          # all tests, about eight minutes on CPU
astrounets smoke                                          # the whole workflow on a synthetic sky
astrounets workflow -c configs/pilot.yaml --foreground    # the whole workflow on 24 real HST images
```

The tests use a synthetic sky; two of them run the complete workflow, once locally and once through
the SLURM profile against a stand-in for SLURM (`tests/fake_slurm.py`).

Single steps, for debugging: `astrounets fetch`, `astrounets dataset --set dataset.name=faber`,
`astrounets train`, `astrounets evaluate --on own:test --on faber:test`, `astrounets plots`, or
train + evaluate + plots with `astrounets run`. `astrounets info` prints the resolved config and the run folder.

## Where things go

```
data/mast/metadata.parquet    every image: MAST metadata, sky tile, split (shared by all datasets)
data/mast/originals/          downloaded *_drz.fits
data/<dataset>/crops.parquet  every usable 256 px crop: position, split, exposure, pixel sigma
data/<dataset>/eval_<split>.parquet   frozen evaluation inputs (crop, exposure ratio, noise seed)
runs/<dataset>/<model>_<attn>_<scaling>_<loss>-<hash8>/
    config.yaml  history.csv  train.json  checkpoints/best.keras  previews/  plots/
    eval/<dataset>_<split>/   summary.json  samples.parquet  sources.parquet  plots/
runs/_workflow/
    config.yaml               the configuration the running workflow uses
    models/ranking.csv  tune/choice.csv  datasets/results.csv
    logs/                     background workflow logs
    .snakemake/slurm_logs/    one log per SLURM job
```

The hash covers everything that changes what a model learns, so two different setups never share a
folder. Evaluation and plotting settings are not part of it: changing a detection threshold re-evaluates
the same model.

## Method in brief

- **Splits by sky tile.** The sky is cut into tiles of `mast.tile_arcmin` (20'); whole tiles go to train,
  validation or test, with one assignment shared by every dataset. Validation and test images whose
  centre lies within `mast.overlap_arcmin` (3.1', a WFC3/IR footprint diagonal) of an image in an earlier
  split are dropped, so no patch of sky is in two splits. (Grouping whole contiguous fields instead
  would put all MOMCHEVA F160W images, which share one field, into a single split.)
- **Noise.** Pixel noise sigma per crop is the median `background_rms` of photutils `Background2D`.
  A shorter exposure t/r is simulated as clean + N(0, sigma * sqrt(r - 1)); r is drawn log-uniformly
  between `noise.ratio_min` and the largest ratio keeping t/r >= `noise.min_new_exposure`.
  Limitation: drizzled noise is spatially correlated; the added noise is white.
- **Scaling** (`none`, `z_scale`, `min_max`, `log_min_max`) is fitted on the noisy input only and the
  same parameters are used for the target and to invert the prediction.
- **Validation** uses a fixed set (`train.val_samples`) that is identical every epoch; the best
  generator by validation reconstruction loss is kept. Runs whose losses turn NaN stop as "diverged"
  and are never ranked.
- **Evaluation** uses frozen noisy inputs per dataset split (identical for every model). Sources are
  detected with SEP in the deep image, the noisy input and the reconstruction with one absolute
  threshold (`eval.detect_nsigma` x noise of the noisy input), matched one-to-one, and measured with
  Kron fluxes, half-light radii and ellipticities. AB magnitudes come from the `PHOTFLAM`/`PHOTPLAM` headers.
- **Ranking** (stage 1): weighted mean of the ranks in flux scatter, |flux bias|, SSIM, radius error and
  ellipticity error (`ranking.weights`), over runs whose completeness and purity are at least those of
  the noisy input (`ranking.guard`).

## Changes from the thesis code (tag `pre-cleanup`)

The old pipeline (1) took the noise level from the spread of the smooth background model, about 70x too
small, (2) scaled noisy and clean images with their own statistics but inverted predictions with the
noisy ones, biasing fluxes by sigma_noisy/sigma_clean, (3) added the sky level a second time when adding
noise, (4) evaluated "uncropped" metrics on training images, (5) compared models on different random
noise, (6) built every dataset from only the first 20 metadata rows (debug line), and (7) matched
sources many-to-one. Results from before the cleanup are not comparable with results from now.
