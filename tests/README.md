# Test Suite

The tests use only synthetic in-memory fixtures and temporary files. They do not read from the repository `data/` or `models/` directories and do not require a GPU.

## Install test dependencies

Use the repository environment or install the test dependencies directly:

```bash
pip install pytest pytest-mock numpy pandas scipy scikit-learn astropy photutils astroquery sep pillow matplotlib pyarrow aiohttp tensorflow-cpu
```

## Run tests

Run the full suite from the repository root:

```bash
pytest --tb=short -q
```

Run unit tests only:

```bash
pytest -m unit --tb=short -q
```

Run integration tests only:

```bash
pytest -m integration --tb=short -q
```

Run slow tests only:

```bash
pytest -m slow --tb=short -q
```

## GPU requirements

No test requires a GPU. All tests are designed to pass on CPU-only environments.

## Synthetic fixtures

Shared fixtures live in `tests/conftest.py`.

- `tiny_fits_array`: 64x64 float32 synthetic astronomy-like image with Gaussian point sources.
- `tiny_fits_file`: temporary FITS file backed by `tiny_fits_array`.
- `tiny_metadata_df`: synthetic metadata table plus temporary FITS files for training/eval/test scenarios.
- `mock_cfg`: temporary config dictionary shaped like `load_config()` output.
- `tiny_unet`, `tiny_discriminator`, `tiny_gan`: tiny TensorFlow models for fast CPU tests.
- `tiny_catalog_df`: small synthetic source catalog.
- `tiny_checkpoint_path`: temporary saved Keras checkpoint with embedded checkpoint metadata.

To extend the fixtures, prefer adding new synthetic columns or temporary files inside `conftest.py` rather than reading external datasets.
