# Test Prompt: rigorous coverage follow-up

## Observed coverage already present

The current suite already covers the main training/evaluation path with synthetic data:

- `starter.py`: config loading, alias determinism, and shared runtime bindings
- `src/training/utils.py`: custom losses, registry resolution, `tf.data` creation, FITS round-trips, checkpoint metadata helpers
- `src/training/math_helpers.py`: scaling/normalization helpers, ratios, sigma helpers, poisson simulation
- `src/models/network.py`: U-Net/discriminator tensor shapes, GAN train-step behavior, reconstruction/adversarial weighting
- `src/data/create_dataset.py`: metadata filtering, post-filter invariants, train/eval/test splitting
- `src/evaluation/metrics.py` + `src/evaluation/merge_catalogs.py`: sliding-window inference, source detection chain, catalog merge behavior
- integration flows for callback outputs, data augmentation caches, U-Net/GAN single-step training, and parquet round-trips

## Prioritized issues: high to low

### 1. Critical: scaling inversion and core filesystem/runtime helpers

These paths can silently corrupt outputs or fail at runtime across many workflows.

- `src/training/math_helpers.py`
  - `inverse_min_max_normalization`
  - `inverse_zscore_normalization`
  - `inverse_adaptive_log_transform_and_denormalize`

- `src/training/utils.py`
  - `_normalize_runtime_filepath`
  - `ensure_directory_exists`
  - `is_not_nan`
  - `open_fits` ratio / bounds / EXPTIME edge cases

### 2. High: distribution and numeric helper behavior

- `src/training/math_helpers.py`
  - `linear_function`
  - `power_law`
  - `evenly_spaced_numbers`
  - `find_distribution`
  - `find_distribution_only_exp`

### 3. High: MAST and visualization helper logic

- `src/data/mast.py`
   - `_chunked`
   - `_choose_best_product`
   - `merge_products_with_metadata`
   - bulk-resolution retry/merge logic

- `src/visualization/prepare_images.py`
   - `build_composite_axes`
   - `plot_source_comparison_sep`
   - `coordinate_detect_source`
   - composite plotting/filtering helpers

### 4. Medium: uncropped evaluation aggregation

- `src/evaluation/uncropped_metrics.py`
   - `log_range`
   - `process_data`
   - CSV/histogram aggregation logic in `main`

## Constraints to preserve

- use only synthetic arrays/DataFrames and pytest temp paths
- never contact MAST or any remote service
- mock network/process-boundary behavior with `pytest-mock`
- keep tests CPU-only and fast
- avoid duplicating checkpoint serialization tests
- prefer randomized synthetic arrays/DataFrames with deterministic seeds

## Immediate next testing task

Add focused pytest coverage for the highest-priority gaps first:

- `src/training/math_helpers.py`
  - randomized round-trip tests for all inverse scaling helpers
  - distribution helper invariants across random log-spaced data
  - spacing/math helper properties

- `src/training/utils.py`
  - runtime path normalization
  - directory creation behavior
  - numeric validation behavior
  - FITS ratio/bounds/error handling
  - checkpoint filename / checkpoint-info edge cases

Then continue with:

- `src/evaluation/uncropped_metrics.py`
  - isolate pure helpers and aggregation paths with mocks

After each wave, run `python -m pytest --tb=short -q` and fix any failures.
