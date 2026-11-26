"""Configuration: one YAML file, a strict schema, `--set` overrides and the run identity.

Every key must be present in the YAML (no defaults in code), unknown keys are errors, and
impossible combinations are rejected before anything runs. `dataset.name` picks one entry of
`datasets`, whose keys (at least pi_names) complete the `dataset` section.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import types
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised for missing, unknown, mistyped or contradictory configuration values."""


SCALING_MODES = ("none", "z_scale", "min_max", "log_min_max")
BOUNDED_SCALINGS = ("min_max", "log_min_max")
LOSSES = ("log_cosh", "simae", "mae", "mse", "ssim")
MODEL_TYPES = ("unet", "gan")
ACTIVATIONS = ("leakyrelu", "relu", "elu", "gelu")
OUTPUT_ACTIVATIONS = ("none", "sigmoid")
RANKING_MEASURES = ("flux_scatter", "flux_bias", "ssim", "radius_error", "ellipticity_error")
CROP_GROUPS = ("overall", "sparse", "crowded")  # evaluation groups: every crop, or by dataset.crowded_sky_fraction
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------------------- schema
@dataclass(frozen=True)
class Paths:
    data: str
    runs: str


@dataclass(frozen=True)
class MastColumns:
    id: str
    exposure: str
    ra: str
    dec: str
    aperture: str
    pi: str
    target: str


@dataclass(frozen=True)
class Split:
    train: int
    val: int
    test: int


@dataclass(frozen=True)
class Mast:
    mock: bool
    mock_images: int
    mock_image_size: int
    mission: str
    query: dict[str, str]
    columns: MastColumns
    page_size: int
    product_suffix: str
    resolve_chunk: int
    retries: int
    download: bool
    download_limit: int | None
    download_workers: int
    http_timeout: float
    tile_arcmin: float
    overlap_arcmin: float
    split: Split


@dataclass(frozen=True)
class Dataset:
    name: str
    apertures: list[str]
    pi_names: list[str]
    exposure_min: float
    exposure_max: float
    max_images: int | None
    filter: str
    hdu: str
    weight_hdu: str
    min_weight_fraction: float
    patch_size: int
    min_valid_fraction: float
    bkg_box_size: int
    bkg_filter_size: int
    bkg_sigma: float
    bkg_exclude_percentile: float
    source_nsigma: float
    source_grow: int
    read_noise: float
    dark_current: float
    max_noise_excess: float
    crowded_sky_fraction: float
    workers: int


@dataclass(frozen=True)
class Noise:
    ratio_min: float
    ratio_max: float
    min_new_exposure: float


@dataclass(frozen=True)
class Scaling:
    mode: str
    log_clip_max: float


@dataclass(frozen=True)
class Model:
    type: str
    depth: int
    filters: int
    growth: int
    kernel_size: int
    pool_size: int
    block_depth: int
    batch_norm: bool
    dropout: float
    dropout_from_layer: int
    attention: bool
    activation: str
    output_activation: str
    initializer: str
    use_bias: bool


@dataclass(frozen=True)
class Gan:
    disc_filters: int
    disc_layers: int
    d_lr_ratio: float
    adversarial_weight: float
    reconstruction_weight: float
    label_smoothing: float


@dataclass(frozen=True)
class Train:
    epochs: int
    batch_size: int
    samples_per_epoch: int
    val_samples: int
    loss: str
    beta_1: float
    learning_rate: float
    lr_drops: list[float]
    lr_drop_factor: float
    patience: int | None
    augment: bool
    preview_every: int
    preview_samples: int
    loader_workers: int
    resume: bool


@dataclass(frozen=True)
class Eval:
    split: str
    n_crops: int
    ratios: list[float]
    batch_size: int
    detect_nsigma: float
    detect_minarea: int
    bkg_box_size: int
    deblend_nthresh: int
    deblend_cont: float
    kron_radius_factor: float
    kron_factor: float
    min_radius: float
    match_radius: float
    mag_bins: list[float]
    n_examples: int


@dataclass(frozen=True)
class Plots:
    dpi: int


@dataclass(frozen=True)
class Ranking:
    guard: bool
    groups: dict[str, float]
    weights: dict[str, float]


@dataclass(frozen=True)
class ModelsStage:
    set: dict
    grid: dict
    evaluate: list[str]


@dataclass(frozen=True)
class TuneStage:
    top: int
    learning_rates: list[float]
    max_epochs: int
    patience: int
    epoch_margin: float
    min_epochs: int


@dataclass(frozen=True)
class DatasetsStage:
    train_on: list[str]
    evaluate: list[str]


@dataclass(frozen=True)
class Workflow:
    models: ModelsStage
    tune: TuneStage
    datasets: DatasetsStage


@dataclass(frozen=True)
class Config:
    seed: int
    paths: Paths
    mast: Mast
    dataset: Dataset
    datasets: dict[str, dict]
    noise: Noise
    scaling: Scaling
    model: Model
    gan: Gan
    train: Train
    eval: Eval
    plots: Plots
    ranking: Ranking
    workflow: Workflow
    root: str  # repository root; set by the loader, never written in YAML

    # ---- derived locations ------------------------------------------------------------------
    def path(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else Path(self.root) / p

    @property
    def data_dir(self) -> Path:
        return self.path(self.paths.data)

    @property
    def mast_dir(self) -> Path:
        return self.data_dir / "mast"

    def dataset_dir(self, name: str | None = None) -> Path:
        return self.data_dir / (name or self.dataset.name)

    @property
    def runs_dir(self) -> Path:
        return self.path(self.paths.runs)

    @property
    def run_name(self) -> str:
        attn = "attn" if self.model.attention else "noattn"
        return f"{self.model.type}_{attn}_{self.scaling.mode}_{self.train.loss}"

    @property
    def run_hash(self) -> str:
        return identity_hash(self)

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.dataset.name / f"{self.run_name}-{self.run_hash}"


# ------------------------------------------------------------------------------- strict builder
def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", str(tp))


def _build(tp: Any, value: Any, where: str) -> Any:
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        options = get_args(tp)
        if value is None and type(None) in options:
            return None
        errors = []
        for option in options:
            if option is type(None):
                continue
            try:
                return _build(option, value, where)
            except ConfigError as err:
                errors.append(str(err))
        raise ConfigError(errors[0] if errors else f"{where}: invalid value {value!r}")
    if is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {value!r}")
        hints = get_type_hints(tp)
        names = [f.name for f in fields(tp) if f.name != "root"]
        unknown = sorted(set(value) - set(names) - ({"root"} if tp is Config else set()))
        missing = [n for n in names if n not in value]
        if unknown:
            raise ConfigError(f"{where or 'config'}: unknown key(s) {unknown}")
        if missing:
            raise ConfigError(f"{where or 'config'}: missing key(s) {missing}")
        kwargs = {n: _build(hints[n], value[n], f"{where}.{n}" if where else n) for n in names}
        if any(f.name == "root" for f in fields(tp)):
            kwargs["root"] = value.get("root", "")
        return tp(**kwargs)
    if origin is list:
        (item,) = get_args(tp)
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        return [_build(item, v, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        _, item = get_args(tp)
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {value!r}")
        return {str(k): _build(item, v, f"{where}.{k}") for k, v in value.items()}
    if tp is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {value!r}")
        return copy.deepcopy(value)
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected text, got {value!r}")
        return value
    raise ConfigError(f"{where}: unsupported type {_type_name(tp)}")


# ------------------------------------------------------------------------------------- loading
# Mappings that an overriding file replaces as a whole instead of merging key by key.
_REPLACED = {("workflow", "models", "grid"), ("workflow", "models", "set"), ("ranking", "groups")}


def _deep_merge(base: dict, extra: dict, _at: tuple = ()) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        here = (*_at, key)
        if isinstance(value, dict) and isinstance(out.get(key), dict) and here not in _REPLACED:
            out[key] = _deep_merge(out[key], value, here)
        else:
            out[key] = copy.deepcopy(value)
    return out


def read_yaml(path: Path) -> dict:
    """Read a YAML file, following a top-level `base:` chain (paths relative to the file)."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    base = data.pop("base", None)
    if base is not None:
        data = _deep_merge(read_yaml(path.parent / base), data)
    return data


def parse_overrides(items: list[str] | None) -> dict:
    """Turn ['train.epochs=5', 'model.attention=false'] into a nested dict (values parsed as YAML)."""
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ConfigError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        if not all(parts):
            raise ConfigError(f"--set: invalid key {key!r}")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"--set: {key!r} conflicts with another override")
        node[parts[-1]] = yaml.safe_load(raw) if raw.strip() else ""
    return out


def find_root(start: Path) -> Path:
    """Repository root: ASTROUNETS_ROOT, else the nearest folder above `start` with pyproject.toml."""
    env = os.environ.get("ASTROUNETS_ROOT")
    if env:
        return Path(env).resolve()
    start = Path(start).resolve()
    for folder in [start, *start.parents]:
        if (folder / "pyproject.toml").is_file():
            return folder
    return start


def _resolve_dataset(data: dict) -> dict:
    """Complete the `dataset` section with the entry of `datasets` named by dataset.name."""
    dataset, presets = data.get("dataset"), data.get("datasets")
    if not isinstance(dataset, dict) or not isinstance(presets, dict):
        return data
    allowed = {f.name for f in fields(Dataset)} - {"name"}
    for preset_name, preset in presets.items():
        if not isinstance(preset, dict):
            raise ConfigError(f"datasets.{preset_name}: expected a mapping")
        unknown = sorted(set(preset) - allowed)
        if unknown:
            raise ConfigError(f"datasets.{preset_name}: {unknown} is not a dataset setting")
    name = dataset.get("name")
    if name not in presets:
        raise ConfigError(f"dataset.name: '{name}' is not defined under datasets (defined: {sorted(presets)})")
    preset = presets[name]
    for key, value in preset.items():
        if key in dataset and dataset[key] != value:
            raise ConfigError(f"dataset.{key} conflicts with datasets.{name}.{key}; change it under datasets")
    return {**data, "dataset": {**dataset, **copy.deepcopy(preset)}}


def from_dict(data: dict, root: Path | str) -> Config:
    data = _resolve_dataset(dict(data))
    data["root"] = str(Path(root).resolve())
    cfg = _build(Config, data, "")
    validate(cfg)
    return cfg


def load_config(path: str | Path, overrides: list[str] | dict | None = None) -> Config:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    data = read_yaml(path)
    extra = overrides if isinstance(overrides, dict) else parse_overrides(overrides)
    data = _deep_merge(data, extra)
    return from_dict(data, find_root(path.parent))


def default_config_path() -> Path:
    """configs/default.yaml of the repository we are in, else of the repository astrounets is installed from."""
    path = find_root(Path.cwd()) / "configs" / "default.yaml"
    return path if path.is_file() else find_root(Path(__file__).parent) / "configs" / "default.yaml"


def to_dict(cfg: Config) -> dict:
    """The config as written in YAML: keys that come from the chosen `datasets` entry are left out."""
    data = asdict(cfg)
    data.pop("root")
    for key in cfg.datasets.get(cfg.dataset.name) or {}:
        data["dataset"].pop(key, None)
    return data


def _resolved(cfg: Config) -> dict:
    data = asdict(cfg)
    data.pop("root")
    return data


def save_config(cfg: Config, path: Path) -> None:
    from astrounets.files import write_text

    write_text(path, yaml.safe_dump(to_dict(cfg), sort_keys=False))


def replace(cfg: Config, overrides: dict) -> Config:
    """A new validated Config with nested overrides applied."""
    return from_dict(_deep_merge(to_dict(cfg), overrides), cfg.root)


def with_dataset(cfg: Config, name: str) -> Config:
    """The same config pointed at another entry of `datasets`."""
    return cfg if name == cfg.dataset.name else replace(cfg, {"dataset": {"name": name}})


# ---------------------------------------------------------------------------------- run identity
# Settings that never change what a model learns; changing them keeps the same run folder.
_IDENTITY_SECTIONS = ("seed", "mast", "dataset", "noise", "scaling", "model", "gan", "train")
_IDENTITY_IGNORED = {
    "mast": ("download", "download_limit", "download_workers", "http_timeout", "retries",
             "resolve_chunk", "page_size"),
    "dataset": ("workers",),
    "train": ("loader_workers", "preview_every", "preview_samples", "resume"),
}


def identity_hash(cfg: Config) -> str:
    data = _resolved(cfg)
    ident = {key: data[key] for key in _IDENTITY_SECTIONS}
    if cfg.model.type != "gan":
        ident.pop("gan")
    for section, keys in _IDENTITY_IGNORED.items():
        for key in keys:
            ident[section].pop(key, None)
    return _hash(ident)


def _hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def dataset_key(cfg: Config) -> str:
    """Fingerprint of everything that decides a dataset's crop table (stored in its summary.json)."""
    data = _resolved(cfg)
    dataset = {k: v for k, v in data["dataset"].items() if k != "workers"}
    mast = {k: data["mast"][k] for k in ("mock", "mock_images", "mock_image_size", "query", "product_suffix",
                                         "tile_arcmin", "overlap_arcmin", "split")}
    return _hash({"dataset": dataset, "mast": mast, "seed": data["seed"]})


def lr_at(train: Train, epoch: int) -> float:
    """Generator learning rate in (0-based) `epoch`: learning_rate, times lr_drop_factor per passed drop."""
    passed = sum(epoch >= round(f * train.epochs) for f in train.lr_drops)
    return train.learning_rate * train.lr_drop_factor ** passed


# ------------------------------------------------------------------------------------ validation
def validate(cfg: Config) -> None:
    errors: list[str] = []

    def need(ok: bool, message: str) -> None:
        if not ok:
            errors.append(message)

    m, d, n, t, e, s = cfg.mast, cfg.dataset, cfg.noise, cfg.train, cfg.eval, cfg.scaling
    need(m.split.train > 0 and m.split.val > 0 and m.split.test > 0, "mast.split: every split needs a positive share")
    need(m.split.train + m.split.val + m.split.test == 100, "mast.split: shares must sum to 100")
    need(m.tile_arcmin > 0 and m.overlap_arcmin >= 0, "mast: tile_arcmin must be positive, overlap_arcmin >= 0")
    need(m.download_limit is None or m.download_limit >= 3, "mast.download_limit must be null or >= 3")
    need(not m.mock or (m.mock_images >= 6 and m.mock_image_size >= 2 * d.patch_size),
         "mast.mock needs mock_images >= 6 and mock_image_size >= 2 * dataset.patch_size")
    for name in cfg.datasets:
        need(re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) is not None and name != "mast",
             f"datasets.{name}: names use lowercase letters, digits, _ or - (and 'mast' is taken)")
    need(d.exposure_max > d.exposure_min > 0, "dataset: need 0 < exposure_min < exposure_max")
    need(0 < d.min_valid_fraction <= 1, "dataset.min_valid_fraction must be in (0, 1]")
    need(0 <= d.min_weight_fraction < 1, "dataset.min_weight_fraction must be in [0, 1) (0 = no weight cut)")
    need(d.source_nsigma > 0 and d.source_grow >= 0, "dataset: source_nsigma must be positive, source_grow >= 0")
    need(d.read_noise >= 0 and d.dark_current >= 0, "dataset: read_noise and dark_current must be >= 0")
    need(d.max_noise_excess > 1, "dataset.max_noise_excess must be > 1")
    need(0 <= d.crowded_sky_fraction <= 1, "dataset.crowded_sky_fraction must be in [0, 1]")
    need(d.patch_size % (cfg.model.pool_size ** (cfg.model.depth - 1)) == 0,
         "dataset.patch_size must be divisible by model.pool_size ** (model.depth - 1)")
    need(d.max_images is None or d.max_images > 0, "dataset.max_images must be null or positive")
    need(1 < n.ratio_min < n.ratio_max, "noise: need 1 < ratio_min < ratio_max")
    need(n.min_new_exposure > 0, "noise.min_new_exposure must be positive")
    need(s.mode in SCALING_MODES, f"scaling.mode must be one of {SCALING_MODES}")
    need(cfg.model.type in MODEL_TYPES, f"model.type must be one of {MODEL_TYPES}")
    need(cfg.model.activation in ACTIVATIONS, f"model.activation must be one of {ACTIVATIONS}")
    need(cfg.model.output_activation in OUTPUT_ACTIVATIONS, f"model.output_activation must be one of {OUTPUT_ACTIVATIONS}")
    need(cfg.model.depth >= 2 and cfg.model.pool_size >= 2 and cfg.model.block_depth >= 1, "model: depth >= 2, pool_size >= 2, block_depth >= 1")
    need(0 <= cfg.model.dropout < 1, "model.dropout must be in [0, 1)")
    need(t.loss in LOSSES, f"train.loss must be one of {LOSSES}")
    need(not (t.loss == "ssim" and s.mode not in BOUNDED_SCALINGS),
         "train.loss ssim needs a bounded scaling (min_max or log_min_max)")
    need(not (cfg.model.output_activation == "sigmoid" and s.mode not in BOUNDED_SCALINGS),
         "model.output_activation sigmoid needs a bounded scaling (min_max or log_min_max)")
    need(t.epochs >= 1 and t.batch_size >= 1 and t.val_samples >= 1, "train: epochs, batch_size, val_samples must be >= 1")
    need(t.samples_per_epoch >= t.batch_size, "train.samples_per_epoch must be >= train.batch_size")
    need(t.preview_every >= 1 and t.loader_workers >= 1, "train: preview_every and loader_workers must be >= 1")
    need(t.learning_rate > 0, "train.learning_rate must be positive")
    need(all(0 < f < 1 for f in t.lr_drops) and t.lr_drops == sorted(set(t.lr_drops)),
         "train.lr_drops must be increasing fractions of train.epochs, each between 0 and 1")
    need(0 < t.lr_drop_factor <= 1, "train.lr_drop_factor must be in (0, 1]")
    need(t.patience is None or t.patience >= 1, "train.patience must be null or >= 1")
    need(cfg.gan.d_lr_ratio > 0, "gan.d_lr_ratio must be positive")
    need(e.split in SPLITS, f"eval.split must be one of {SPLITS}")
    need(bool(e.ratios) and all(r > 1 for r in e.ratios), "eval.ratios must all be > 1")
    need(e.n_crops >= 1 and e.batch_size >= 1, "eval: n_crops and batch_size must be >= 1")
    need(e.mag_bins == sorted(e.mag_bins) and len(e.mag_bins) >= 2, "eval.mag_bins must be increasing")
    groups = cfg.ranking.groups
    need(isinstance(groups, dict) and set(groups) <= set(CROP_GROUPS)
         and all(isinstance(w, (int, float)) and w >= 0 for w in groups.values()) and sum(groups.values()) > 0,
         f"ranking.groups: weights >= 0 with a positive sum for some of {CROP_GROUPS}, e.g. {{sparse: 3, crowded: 1}}")
    unknown = set(cfg.ranking.weights) - set(RANKING_MEASURES)
    need(not unknown, f"ranking.weights: unknown measure(s) {sorted(unknown)}; use {RANKING_MEASURES}")
    need(all(w >= 0 for w in cfg.ranking.weights.values()) and sum(cfg.ranking.weights.values()) > 0,
         "ranking.weights must be non-negative with a positive sum")
    _validate_workflow(cfg, need)
    if errors:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(errors))


def _validate_workflow(cfg: Config, need) -> None:
    w = cfg.workflow
    for key, values in w.models.grid.items():
        need(isinstance(values, list) and len(values) > 0, f"workflow.models.grid.{key} must be a non-empty list")
    for name, items in (("models", w.models.evaluate), ("datasets", w.datasets.evaluate)):
        for item in items:
            dataset, sep, split = str(item).partition(":")
            need(sep == ":" and split in SPLITS and (dataset == "own" or dataset in cfg.datasets),
                 f"workflow.{name}.evaluate: '{item}' must be <dataset>:<split> with a dataset from "
                 f"datasets (or 'own') and a split from {SPLITS}")
    need(bool(w.models.evaluate), "workflow.models.evaluate: the ranking needs one evaluation target")
    t = w.tune
    need(t.top >= 1, "workflow.tune.top must be >= 1")
    need(bool(t.learning_rates) and all(r > 0 for r in t.learning_rates), "workflow.tune.learning_rates must be positive")
    need(1 <= t.min_epochs <= t.max_epochs, "workflow.tune: need 1 <= min_epochs <= max_epochs")
    need(t.patience >= 1 and t.epoch_margin >= 1, "workflow.tune: patience >= 1 and epoch_margin >= 1")
    need(bool(w.datasets.train_on) and all(n in cfg.datasets for n in w.datasets.train_on),
         f"workflow.datasets.train_on: every name must be defined under datasets ({sorted(cfg.datasets)})")
    need(bool(w.datasets.evaluate), "workflow.datasets.evaluate needs at least one target")
