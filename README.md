# AstroGAN-UNet

Denoising of HST WFC3/IR F160W drizzled images with a U-Net or a GAN (U-Net generator + conditional
PatchGAN). Deep exposures are the targets; shorter exposures are simulated by adding noise. Models are
judged on what matters for photometry: flux accuracy, source shapes, image structure, completeness and purity.
The Python package and command are called `astrounets`.

## On bwForCluster Helix (everything as SLURM jobs)

The repository is [erennere/AstroGAN-UNet](https://github.com/erennere/AstroGAN-UNet). Clone it where
there is room (the download is about 155 GB), for example into your SDS@hd space, which Helix mounts
under /mnt/sds-hd/ on login and compute nodes. Everything is run from the repository folder; `data/`,
`runs/` and `.venv/` are created inside it.

```
git clone https://github.com/erennere/AstroGAN-UNet.git && cd AstroGAN-UNet
sbatch workflow/helix/setup.sbatch       # once: .venv (no modules), GPU check, tests -> setup-<id>.log
sbatch workflow/helix/run.sbatch         # the study -> workflow-<id>.log
.venv/bin/astrounets status              # progress of each stage
.venv/bin/astrounets stop                # stop; submitting run.sbatch again continues where it stopped
```

`setup.sbatch` builds `.venv` with `setup_env.sh`: it uses a Python 3.11-3.13 if one is on the PATH,
otherwise uv downloads one into `.tools/` (the job needs internet). `run.sbatch` is a small CPU job that
runs [Snakemake](https://snakemake.readthedocs.io); Snakemake downloads the data inside that job and
submits every training run as its own GPU job (`gpu:A40:1`) and every dataset build as a CPU job.
15 minutes before the 120 h limit the job stops cleanly and submits itself again; the new job starts
when the old one has ended and continues where it stopped. Another config: `sbatch
workflow/helix/run.sbatch configs/pilot.yaml`. Cluster resources (partition, GPU, time, memory, how many
jobs at once) are in `workflow/profiles/slurm/config.yaml`.

## On a laptop or workstation

| Where | Install |
| --- | --- |
| Windows (CPU) | `powershell -ExecutionPolicy Bypass -File setup_env.ps1` |
| Linux / WSL | `bash setup_env.sh` (`CPU_ONLY=1` for a CPU-only TensorFlow) |

Both create `.venv/` in the repository with pinned versions (TensorFlow 2.20, Snakemake 9). Then:

```
astrounets workflow      # start (or continue) everything; runs in the background
astrounets status        # progress of each stage + the end of the log
astrounets stop          # stop; `astrounets workflow` later continues where it stopped
```

`astrounets workflow` runs Snakemake in the background, so you can close the console (the computer must
stay awake). Jobs run one at a time on this machine. Starting again skips finished runs and resumes
interrupted ones from their last epoch. Useful options: `--dry-run` (list the jobs), `--foreground`
(stay attached), `--until data|models|tune|datasets` (stop after a stage), `-c <config>` and
`--set key=value` (as everywhere). Every result is written to a temporary file and renamed when
complete, so a killed job never leaves half a file behind.

## The study

Defined in the `workflow:` block of `configs/default.yaml`:

1. **fetch**: one MAST query, sky tiles, splits, downloads. Only images some dataset can use are
   downloaded: in November 2026 that was about 9,000 of 13,459 F160W images, roughly 155 GB.
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

## Checks

```
python -m pytest                                          # all tests, about ten minutes on CPU
astrounets smoke                                          # the whole workflow on a synthetic sky
astrounets workflow -c configs/pilot.yaml --foreground    # the whole workflow on 24 real HST images
```

The tests use a synthetic sky; three of them run the complete workflow: locally, through the SLURM
profile, and as `workflow/helix/run.sbatch` including its resubmission before the time limit, the last
two against a stand-in for SLURM (`tests/fake_slurm.py`).

Single steps, for debugging: `astrounets fetch`, `astrounets dataset --set dataset.name=faber`,
`astrounets train`, `astrounets evaluate --on own:test --on faber:test`, `astrounets plots`, or
train + evaluate + plots with `astrounets run`. `astrounets info` prints the resolved config and the run folder.

## Where things go

```
data/mast/metadata.parquet    every image: MAST metadata, sky tile, split (shared by all datasets)
data/mast/originals/          downloaded *_drz.fits
data/<dataset>/crops.parquet  every usable 256 px crop: position, split, exposure, pixel sigma, crowded
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
- **No-data pixels.** Empty and non-finite pixels, and pixels whose drizzle weight (`WHT`) is below
  `dataset.min_weight_fraction` (0.5) of the image's median weight, count as no data: the WFC3/IR
  "death star" and partly covered edges would otherwise enter training and evaluation as sky. No-data
  pixels get no noise, and reconstructions are set to 0 there.
- **Noise.** The pixel noise sigma of a crop is *measured* as the robust spread (1.4826 x MAD) of its
  empty-sky pixels around the photutils `Background2D` background; sources are masked first (pixels
  2 sigma above the background after 2-px smoothing, grown by 5 px). It is checked against the noise
  *expected* from photons, dark current and read noise, sigma_exp^2 = (B + dark)/t + n * RN^2 / t^2, with
  B the local background (sky plus unresolved light), t the pixel's exposure from the drizzle weight
  map and n the exposures per pixel. Drizzling correlates neighbouring pixels and lowers the measured
  spread by a factor k, the median of measured/expected over the dataset (0.74 on the 22 pilot images;
  0.62-0.82 per image in uncrowded fields). Where a crop has too little empty sky, or its
  measured noise is more than `dataset.max_noise_excess` (1.3) x k x expected, the spread is structure
  (unresolved stars, nebulae, the body of a galaxy), not noise, and k x sigma_exp is used instead.
  On the pilot images this leaves sparse fields within 2-8 % of the previous estimate and lowers it
  by 3-24x in M31 (PHAT), Orion, M59 and 47 Tuc fields, where the previous estimate (the clipped spread
  of all pixels) mistook unresolved stars for noise. The crop's exposure is the drizzle weight, not
  the header `EXPTIME`, which for mosaics is the total over all pointings (up to 8x more).
  A shorter exposure t/r is simulated as clean + N(0, sigma * sqrt(r - 1)); r is drawn log-uniformly
  between `noise.ratio_min` and the largest ratio keeping t/r >= `noise.min_new_exposure`.
  Limitations: drizzled noise is spatially correlated, the added noise is white; the photon noise of
  sources themselves is not added (background-limited approximation).
- **Crowded crops.** A crop is crowded if less than `dataset.crowded_sky_fraction` (half) of it is
  empty sky or its noise came from the expectation. Crowded crops stay in training and evaluation.
  Evaluation sets draw up to `eval.n_crops` crops from each group, so the crowded numbers do not rest
  on a handful of crops, and every evaluation reports both groups and all crops together (`groups` in
  summary.json, `<measure>_<group>` columns in the result tables; `overall` mixes the groups as drawn).
- **Scaling** (`none`, `z_scale`, `min_max`, `log_min_max`) is fitted on the noisy input only and the
  same parameters are used for the target and to invert the prediction.
- **Validation** uses a fixed set (`train.val_samples`) that is identical every epoch; the best
  generator by validation reconstruction loss is kept. Runs whose losses turn NaN stop as "diverged"
  and are never ranked.
- **Evaluation** uses frozen noisy inputs per dataset split (identical for every model). Sources are
  detected with SEP in the deep image, the noisy input and the reconstruction with one absolute
  threshold (`eval.detect_nsigma` x noise of the noisy input) and one background (estimated on the
  deep image; separate estimates absorb more source light the noisier an image is, which made
  noisy-input fluxes up to 0.3 mag too faint in crowded fields), matched one-to-one, and measured with
  Kron fluxes, half-light radii and ellipticities. AB magnitudes come from the `PHOTFLAM`/`PHOTPLAM` headers.
  PSNR is computed per crop (with that crop's deep-image range) and reported as the median over crops;
  until November 2026 it came from the mean error over all crops, where a few bright crops dominated.
- **Ranking** (stage 1): weighted mean of the ranks in flux scatter, |flux bias|, SSIM, radius error and
  ellipticity error (`ranking.weights`), over runs whose completeness and purity are at least those of
  the noisy input (`ranking.guard`). Each measure is ranked within sparse and within crowded crops, and
  the two ranks are averaged with the weights in `ranking.groups` (sparse 3, crowded 1): the study is
  about deep extragalactic fields first, but a setup that fails in crowded fields drops, and the guard
  must hold in both. Pooling all crops instead would let the crowded ones decide, because they hold
  most of the sources, and by how many crowded tiles happen to land in the validation split.

## Changes from the thesis code (tag `pre-cleanup`)

The old pipeline (1) took the noise level from the spread of the smooth background model, about 70x too
small, (2) scaled noisy and clean images with their own statistics but inverted predictions with the
noisy ones, biasing fluxes by sigma_noisy/sigma_clean, (3) added the sky level a second time when adding
noise, (4) evaluated "uncropped" metrics on training images, (5) compared models on different random
noise, (6) built every dataset from only the first 20 metadata rows (debug line), and (7) matched
sources many-to-one. Results from before the cleanup are not comparable with results from now.
