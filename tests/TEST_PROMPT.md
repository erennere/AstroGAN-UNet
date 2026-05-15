# Test Prompt: coverage-guided follow-up

## Observed coverage already present

The current suite already covers the main training/evaluation path with synthetic data:

- `starter.py`: config loading, alias determinism, and shared runtime bindings
- `src/training/utils.py`: custom losses, registry resolution, `tf.data` creation, FITS round-trips, checkpoint metadata helpers
- `src/training/math_helpers.py`: scaling/normalization helpers, ratios, sigma helpers, poisson simulation
- `src/models/network.py`: U-Net/discriminator tensor shapes, GAN train-step behavior, reconstruction/adversarial weighting
- `src/data/create_dataset.py`: metadata filtering, post-filter invariants, train/eval/test splitting
- `src/evaluation/metrics.py` + `src/evaluation/merge_catalogs.py`: sliding-window inference, source detection chain, catalog merge behavior
- integration flows for callback outputs, data augmentation caches, U-Net/GAN single-step training, and parquet round-trips

## Under-tested or untested areas discovered

The following modules still contain logic with little or no direct coverage:

1. `src/data/mast.py`
   - `_chunked`
   - `_choose_best_product`
   - `merge_products_with_metadata`
   - bulk-resolution retry/merge logic

2. `src/visualization/prepare_images.py`
   - `build_composite_axes`
   - `plot_source_comparison_sep`
   - `coordinate_detect_source`
   - composite plotting/filtering helpers

3. `src/evaluation/uncropped_metrics.py`
   - `log_range`
   - `process_data`
   - CSV/histogram aggregation logic in `main`

## Constraints to preserve

- use only synthetic arrays/DataFrames and pytest temp paths
- never contact MAST or any remote service
- mock network/process-boundary behavior with `pytest-mock`
- keep tests CPU-only and fast
- avoid duplicating checkpoint serialization tests

## Immediate next testing task

Add a focused pytest module that covers pure helper logic in:

- `src/data/mast.py`
  - chunk splitting
  - best-product selection policy
  - metadata URL merge behavior while preserving existing URLs

- `src/visualization/prepare_images.py`
  - composite axes construction
  - ellipse grouping for matched/unmatched detections
  - filtering behavior in `coordinate_detect_source`

After that, run `python -m pytest --tb=short -q` and fix any failures.
