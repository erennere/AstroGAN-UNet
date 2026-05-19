# Training Stage

The training stage converts curated FITS crops into learned reconstruction models.
It is the bridge between data preparation and evaluation, and it is responsible for producing reproducible checkpoint trees that downstream metrics and visualization consume.

The stage supports two runtime modes:

- U-Net generator-only training
- GAN training (generator + discriminator)

## Stage Components

| File | Role |
| --- | --- |
| `new_train.py` | Main training entrypoint and orchestration |
| `callback.py` | Checkpoint, logging, and training callback lifecycle |
| `utils.py` | Dataset assembly, sampling, and data utilities |
| `math_helpers.py` | Numeric helpers for exposure/noise transforms |

## Training Architecture

```mermaid
flowchart TD
	cfg[new_train section from config] --> main[src.training.new_train.main]
	main --> kwargs[data/network/discriminator/gan kwargs]
	kwargs --> data[data pipeline and tf datasets]
	kwargs --> net[generator build]
	kwargs --> disc[discriminator build when GAN]
	net --> fit[model.fit]
	disc --> fit
	data --> fit
	fit --> artifacts[checkpoints + history + logs + sampled csv caches]
```

## Runtime Data Flow

1. Load resolved `new_train` config from `starter.py`.
2. Materialize runtime dictionaries:
   - `data_kwargs`
   - `network_kwargs`
   - `discriminator_kwargs`
   - `gan_kwargs`
3. Build datasets from metadata/statistics and split image directories.
4. Build training graph (U-Net only or GAN).
5. Execute training loop and persist artifacts.

## Artifact Flow

```mermaid
flowchart LR
	in1[cropped_images_stats_*.csv] --> train[new_train.py]
	in2[training/eval/test image splits] --> train
	train --> ckpt[*.keras checkpoints]
	train --> hist[training_history_*.csv and *.json]
	train --> logs[validation and metrics text logs]
	train --> caches[sampled_training/eval/test_*.csv]
```

## Typical Runs

Default:

```bash
python -m src.training.new_train
```

GAN run example:

```bash
python -m src.training.new_train --model-type gan --attention true --scaling log_min_max --loss-name ssim_loss --dropout-rate 0.3 --output-activation sigmoid
```

U-Net run example:

```bash
python -m src.training.new_train --model-type unet --attention false --scaling z_scale --loss-name log_cosh_loss
```

## Key Config Sections

Training behavior is configured under `new_train`:

- `new_train.data`
- `new_train.network`
- `new_train.discriminator`
- `new_train.gan`
- `new_train.training`

Inherited values come from earlier owners:

- `create_dataset` (dataset schema and geometry)
- `paths` (path roots for data and model artifacts)

CLI overrides parsed in `starter.py` take precedence over YAML-owned values.

## Downstream Consumers

Training outputs are consumed by:

- `src.evaluation.metrics`
- `src.evaluation.uncropped_metrics`
- `src.visualization.prepare_images`

This makes checkpoint naming and output directory stability critical for reproducible full-pipeline runs.
