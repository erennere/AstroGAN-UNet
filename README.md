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

Changing the study while it runs: `astrounets stop`, `git pull` (or edit the config), `sbatch
workflow/helix/run.sbatch`. Finished runs are kept, interrupted ones resume from their last epoch, new grid
entries (say another scaling mode) become new runs, and the ranking is redone once they are done. A run
keeps its folder as long as nothing it learns from changes (the hash below); configuration fields added
since a run started are filled in from the new configuration. Never pull while the workflow runs: every
job reads the configuration frozen at the start.

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
3. **models** (stage 1): the 68 valid combinations of model type, attention, scaling and loss, 20 epochs
   each on `all`, ranked on the validation split (see *Ranking*).
4. **tune**: the best `workflow.tune.top` setups (3), each trained with every rate in
   `workflow.tune.learning_rates` at a constant rate until the validation loss has not improved for
   `patience` epochs. Per setup the rate with the lowest validation loss wins; its stage-2 epoch budget
   is `epoch_margin` x the epoch of that best loss, kept within `min_epochs` and `max_epochs`.
5. **datasets** (stage 2): each tuned setup trained on `all`, `faber` and `momcheva` with its own rate
   and epochs (the rate halves after 75 % of the epochs, `train.lr_drops`), tested on the test split of
   all three. Final table: `runs/_workflow/datasets/results.csv`.
6. **final**: the final model, which `astrounets denoise` uses by default: of the stage-2 models
   trained on `all` (`workflow.final.train_on`), one per tuned setup, the best by the ranking rule on
   the validation split (`workflow.final.select_on`; evaluated there first if needed), so the test
   numbers of the chosen model stay unbiased. `runs/_workflow/final/final.json` and `ranking.csv`.

## Using a trained model on real images

```
astrounets denoise image_drz.fits                     # the study's final model
astrounets denoise data/new/ --out denoised/          # every FITS file in a folder
astrounets denoise "data/new/*_drz.fits" --model runs/_workflow/report/models/all/<run> --zoom 812,640
```

On Helix: `sbatch workflow/helix/denoise.sbatch <the same arguments>` (a GPU job; log
`denoise-<id>.log`).

For each image: `<out>/<image>_rec.fits`, shaped like the input (its headers and WCS, the reconstruction
in place of the science image, the weight map; `DNMODEL`, `DNEPOCH`, `DNSCALE`, `DNOVERLP`, ... say how
it was made), and `<image>_compare.png`: original | reconstruction | difference with the same display
range, the whole image and full-resolution cutouts (the centre, or each `--zoom X,Y`). `--show` also
opens both images in a window where zoom and pan move both; or open both files in DS9 with
`ds9 image_drz.fits denoised/image_drz_rec.fits -tile -lock frame wcs -lock scale -lock colorbar`.

The model (`--model`, default `denoise.model: best`) is a run folder or a copy under
`runs/_workflow/report/models/`; `best` is the final model of the study, also when only the report
folder has been copied from the cluster. Images of any size are reconstructed in overlapping tiles of
the model's patch size (256 px), each scaled on its own as in training and blended with a cosine taper
over `denoise.overlap` (64 px); the image edge is mirrored, so it never lies at a tile edge. No-data
pixels (empty, non-finite, or below the training weight cut) are filled with the local sky for the model,
as in training, and keep their input values. With an identity model the input comes back unchanged at
any size (`tests/test_infer.py`). `--tta` averages over the 8 rotations and flips (8x the time). A full-size model needs about 3 GB of
memory on a CPU at the default `--batch-size 4` (8: 5 GB, 16: more than 7 GB) and is not faster with larger
batches; a WFC3/IR frame of 1000 x 1100 pixels takes about 25 s on two CPU cores.
Settings: the `denoise:` section of `configs/default.yaml` and the options of `astrounets denoise --help`;
`--out` and `--model` are relative to the current folder, the config values to the repository.

The models know WFC3/IR F160W drizzled images in e-/s (0.128″ pixels). Images in electrons are divided
by EXPTIME and written back in electrons; another filter, instrument, pixel scale or unit is processed
but warned about, since the model has not seen such data.

From Python:

```python
from astrounets.infer import TrainedModel
model = TrainedModel.load("runs/_workflow/report/models/all/<run>")
rec = model.denoise(image, weight)           # 2-D arrays in e-/s; any size
```

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
`astrounets train`, `astrounets evaluate --on own:test --on faber:test`, `astrounets plots [--detailed]`, or
train + evaluate + plots with `astrounets run`. `astrounets info` prints the resolved config and the run folder.

## Where things go

```
data/mast/metadata.parquet    every image: MAST metadata, sky tile, split (shared by all datasets)
data/mast/originals/          downloaded *_drz.fits
data/<dataset>/crops.parquet  every usable 256 px crop: position, split, exposure, pixel sigma, crowded
data/<dataset>/eval_<split>.parquet   frozen evaluation inputs (crop, exposure ratio, noise seed)
denoised/<image>_rec.fits, <image>_compare.png    astrounets denoise
runs/<dataset>/<model>_<attn>_<scaling>_<loss>-<hash8>/
    config.yaml  history.csv  train.json  checkpoints/best.keras  previews/  plots/
    eval/<dataset>_<split>/   summary.json  samples.parquet  sources.parquet  plots/
runs/_workflow/
    config.yaml               the configuration the running workflow uses
    models/ranking.csv  tune/choice.csv  datasets/results.csv  final/final.json
    report/                   the study's figures, tables and final models (see Figures)
    logs/                     background workflow logs
    .snakemake/slurm_logs/    one log per SLURM job
```

## Figures

Every figure is drawn only from files the stages wrote, so it can be redrawn at any time without
training or evaluating again: `astrounets plots --run <run folder> [--detailed]` for one run,
`astrounets report` for the study (the workflow's last step does this). Sparse and crowded crops are
always shown separately, and the reconstruction always next to the noisy input on the same samples.

- **Per run** (`plots/`): `history.png`, the training and validation loss.
- **Per evaluation** (`eval/<dataset>_<split>/plots/`):
  - `examples.png`: noisy | reconstruction | deep image, crops of both groups at low to high ratios;
  - `completeness.png`: completeness against true AB magnitude;
  - `dmag_vs_mag.png`: measured − true magnitude against true magnitude;
  - `by_ratio.png`: completeness, flux scatter, SSIM and PSNR against exposure ratio, on the crops
    evaluated at every ratio up to 16, so the curves show the effect of noise, not of which crops allow
    which ratio (high ratios are possible only for long exposures).
- **Hexbins** (the thesis figures; for the final models, `plots.hexbins`): `hexbin/<group>/<subset>/`,
  group overall / sparse / crowded, subset all sources or AB < 25. One panel per exposure ratio,
  reconstruction next to noisy input, log counts, running median and reference line:
  flux ratio against true AB magnitude and true flux, ΔKron radius against true AB magnitude and true
  Kron radius, Kron radius and flux against their true values; each for sources found in each image
  (`*_combined`), in both (`*_combined_both`) and only in the reconstruction (`*_rec`). Fluxes are Kron
  fluxes in e⁻/s, radii circularised Kron radii in arcsec (0.12825″ pixels).
- **Study** (`runs/_workflow/report/`): `ranking.png` (stage-1 scores, top setups marked),
  `tuning.png` (validation loss per learning rate, the choice), `results_<measure>.png` (stage 2,
  trained on × tested on, sparse and crowded: flux scatter, |flux bias|, completeness gain, SSIM),
  `tables/` and `models/` (each final generator with its config), listed in `index.txt` with the final
  model (`tables/final.json`).
- **Denoised images** (`astrounets denoise`): `<image>_compare.png`, see *Using a trained model*.

The hash covers everything that changes what a model learns, so two different setups never share a
folder. Evaluation and plotting settings are not part of it: changing a detection threshold re-evaluates
the same model.

## Method in brief

- **Splits by sky tile.** The sky is cut into tiles of `mast.tile_arcmin` (20'); whole tiles go to train,
  validation or test, with one assignment shared by every dataset. Validation and test images whose
  centre lies within `mast.overlap_arcmin` (3.1', a WFC3/IR footprint diagonal) of an image in an earlier
  split are dropped, so no patch of sky is in two splits. (Grouping whole contiguous fields instead
  would put all MOMCHEVA F160W images, which share one field, into a single split.) Tiles are shared
  out within each observing program (`mast.split_per_program`): a tile belongs to the PI with most
  images in it, and every program with 3+ tiles is split 60/20/20 on its own (the others together),
  so big programs such as PHAT, Orion, SH0ES, CLASH, CANDELS and 3D-HST appear in all three splits
  (with one global draw PHAT was 458/31/174 train/val/test). Programs on a single tile, such as the
  WFC3 calibration star clusters, stay in one split.
- **No-data pixels.** Empty and non-finite pixels, and pixels whose drizzle weight (`WHT`) is below
  `dataset.min_weight_fraction` (0.5) of the image's median weight, count as no data: the WFC3/IR
  "death star", partly covered edges and pixels where exposures were rejected. They are set to the
  crop's background level in both the deep and the noisy image, without noise, so the network sees
  flat empty sky there; in evaluation they hold that same value in all three images. (Kept at 0, as
  until November 2026, they lay ~20 sigma below the sky: the network smeared them into dark blobs and
  they set the minimum of the min_max and log_min_max scalings. 73 % of all crops contain some.)
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
- **Training mix.** Training draws crops at random, at most `train.max_crowded_share` (25 %) of them
  crowded and the rest sparse, each crop equally likely within its group (the natural share is used
  when it is lower, e.g. for FABER and MOMCHEVA). The fixed validation set that picks the best
  checkpoint is drawn the same way. Without the cap, crowded fields, large mosaics and repeatedly
  observed star clusters would take whatever share their crop counts give them (56 % in the pilot).
- **Scaling** (`none`, `z_scale`, `robust_z`, `min_max`, `log_min_max`) is fitted on the noisy input only and
  the same parameters are used for the target and to invert the prediction. `robust_z` uses the median and
  1.4826 x MAD of the crop's valid pixels instead of mean and standard deviation: with `z_scale` a bright star
  can raise a crop's standard deviation some 70x above the sky noise, so the noise nearly vanishes in scaled
  units and the whole crop (on whole images: the whole tile, a visible square) is barely denoised.
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
