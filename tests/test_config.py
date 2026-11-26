import pytest
import yaml

from astrounets.config import ConfigError, load_config, lr_at, parse_overrides, replace, to_dict, with_dataset
from conftest import DEFAULT, SMOKE, smoke_config


def test_shipped_configs_load():
    default, smoke = load_config(DEFAULT), load_config(SMOKE)
    assert default.dataset.patch_size == 256 and smoke.dataset.patch_size == 64
    assert default.run_dir.name.startswith("unet_attn_z_scale_log_cosh-")


def test_overrides_are_parsed_as_yaml():
    assert parse_overrides(["train.epochs=5", "model.attention=false", "eval.ratios=[2, 4]"]) == {
        "train": {"epochs": 5}, "model": {"attention": False}, "eval": {"ratios": [2, 4]}}
    with pytest.raises(ConfigError):
        parse_overrides(["train.epochs"])


def test_unknown_and_missing_keys_fail(tmp_path):
    data = yaml.safe_load(DEFAULT.read_text())
    data["train"]["epoch"] = 3
    bad = tmp_path / "bad.yaml"; bad.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(bad)
    data = yaml.safe_load(DEFAULT.read_text())
    del data["noise"]["ratio_max"]
    bad.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="missing key"):
        load_config(bad)


def test_types_are_checked():
    with pytest.raises(ConfigError, match="integer"):
        load_config(DEFAULT, ["train.epochs=5.5"])
    with pytest.raises(ConfigError, match="true/false"):
        load_config(DEFAULT, ["model.attention=yes please"])


@pytest.mark.parametrize("override, message", [
    (["train.loss=ssim", "scaling.mode=z_scale"], "ssim needs a bounded scaling"),
    (["train.loss=ssim", "scaling.mode=none"], "ssim needs a bounded scaling"),
    (["model.output_activation=sigmoid"], "sigmoid needs a bounded scaling"),
    (["model.depth=6", "dataset.patch_size=48"], "divisible"),
    (["mast.split={train: 50, val: 20, test: 20}"], "sum to 100"),
    (["train.lr_drops=[0.5, 1.2]"], "lr_drops"),
    (["train.lr_drops=[0.8, 0.5]"], "lr_drops"),
    (["train.patience=0"], "patience"),
    (["workflow.tune.min_epochs=500"], "min_epochs"),
    (["workflow.datasets.train_on=[nowhere]"], "train_on"),
    (["workflow.models.evaluate=[all:dev]"], "evaluate"),
    (["workflow.datasets.evaluate=[nowhere:test]"], "evaluate"),
    (["workflow.models.grid={model.type: []}"], "non-empty"),
    (["noise.ratio_min=1.0"], "ratio_min"),
    (["dataset.name=Has Space"], "not defined under datasets"),
    (["datasets={Bad Name: {pi_names: []}}"], "names use lowercase"),
    (["dataset.pi_names=[FABER]"], "conflicts with datasets.all.pi_names"),
    (["datasets.faber.patch_sizes=3"], "not a dataset setting"),
    (["ranking.weights={speed: 1}"], "unknown measure"),
    (["ranking.groups={sparse: -1}"], "ranking.groups"),
    (["ranking.groups={dense: 1}"], "ranking.groups"),
])
def test_impossible_setups_are_rejected(override, message):
    with pytest.raises(ConfigError, match=message):
        load_config(DEFAULT, override)


def test_valid_bounded_ssim_setup_loads():
    load_config(DEFAULT, ["train.loss=ssim", "scaling.mode=min_max", "model.output_activation=sigmoid"])


def test_run_identity(tmp_path):
    cfg = smoke_config(tmp_path)
    same = replace(cfg, {"train": {"loader_workers": 7, "preview_every": 3}, "eval": {"n_crops": 3}})
    assert same.run_dir == cfg.run_dir, "operational and evaluation settings must not change the run folder"
    for change in ({"train": {"learning_rate": 0.001}}, {"train": {"lr_drops": []}}, {"train": {"patience": 3}},
                   {"dataset": {"name": "mock-faber"}}, {"model": {"depth": 2}},
                   {"noise": {"ratio_max": 50.0}}, {"seed": 7}, {"model": {"initializer": "glorot_uniform"}}):
        assert replace(cfg, change).run_dir != cfg.run_dir, change
    assert cfg.run_dir.parent.name == cfg.dataset.name


def test_gan_settings_only_matter_for_gan(tmp_path):
    cfg = smoke_config(tmp_path)
    assert replace(cfg, {"gan": {"label_smoothing": 0.2}}).run_dir == cfg.run_dir
    gan = replace(cfg, {"model": {"type": "gan"}})
    assert replace(gan, {"gan": {"label_smoothing": 0.2}}).run_dir != gan.run_dir


def test_roundtrip_through_saved_yaml(tmp_path):
    from astrounets.config import from_dict, read_yaml, save_config

    cfg = smoke_config(tmp_path)
    save_config(cfg, tmp_path / "saved.yaml")
    again = from_dict(read_yaml(tmp_path / "saved.yaml"), cfg.root)
    assert to_dict(again) == to_dict(cfg) and again.run_dir == cfg.run_dir


def test_dataset_presets_complete_the_dataset_section(tmp_path):
    cfg = load_config(DEFAULT)
    assert cfg.dataset.name == "all" and cfg.dataset.pi_names == []
    faber = with_dataset(cfg, "faber")
    assert faber.dataset.pi_names == ["FABER"] and faber.dataset.patch_size == cfg.dataset.patch_size
    assert faber.run_dir.parent.name == "faber" and faber.run_hash != cfg.run_hash
    assert "pi_names" not in to_dict(faber)["dataset"], "preset keys are not written back into dataset"
    assert with_dataset(faber, "all").run_dir == cfg.run_dir


def test_overriding_file_replaces_the_stage1_grid_and_set():
    default, smoke = load_config(DEFAULT), load_config(SMOKE)
    assert len(default.workflow.models.grid) == 4 and default.workflow.models.set == {"train.epochs": 20}
    assert smoke.workflow.models.grid == {"model.type": ["unet", "gan"]} and smoke.workflow.models.set == {}


def test_learning_rate_drops_at_fractions_of_the_epochs():
    train = load_config(DEFAULT, ["train.epochs=20", "train.lr_drops=[0.5, 0.75]", "train.lr_drop_factor=0.5"]).train
    rates = [lr_at(train, e) for e in range(20)]
    assert rates[:10] == [0.0002] * 10 and rates[10:15] == [0.0001] * 5 and rates[15:] == [0.00005] * 5
    constant = load_config(DEFAULT, ["train.lr_drops=[]"]).train
    assert {lr_at(constant, e) for e in range(60)} == {0.0002}


def test_ranking_groups_are_replaced_not_merged():
    cfg = load_config(DEFAULT, ["ranking.groups={sparse: 1}"])
    assert cfg.ranking.groups == {"sparse": 1}
    assert load_config(DEFAULT).ranking.groups == {"sparse": 3.0, "crowded": 1.0}
