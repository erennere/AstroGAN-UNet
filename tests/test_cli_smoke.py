import pandas as pd
import pytest

from astrounets.cli import main
from conftest import ROOT


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for command in ("workflow", "status", "stop", "fetch", "dataset", "train", "evaluate", "plots", "run", "info", "smoke"):
        assert command in out


def test_info_prints_run_folder(capsys):
    assert main(["info", "--set", "model.type=gan"]) == 0
    assert "gan_attn_z_scale_log_cosh-" in capsys.readouterr().out


def test_config_errors_are_reported_without_traceback(capsys):
    assert main(["info", "--set", "dataset.name=unknown"]) == 1
    assert "error: dataset.name: 'unknown' is not defined under datasets" in capsys.readouterr().err


def test_pilot_config_is_a_small_real_workflow(tmp_path):
    from astrounets.config import load_config
    from astrounets.workflow import stage_runs

    cfg = load_config(ROOT / "configs" / "pilot.yaml")
    assert cfg.mast.download_limit == 24 and not cfg.mast.mock and cfg.dataset.name == "pilot"
    assert cfg.runs_dir == ROOT / "pilot" / "runs" and cfg.data_dir == ROOT / "data"
    assert len(stage_runs(cfg, "models")[0]) == 2 and cfg.workflow.datasets.train_on == ["pilot"]


@pytest.mark.slow
def test_smoke_runs_the_whole_workflow(tmp_path):
    from astrounets.smoke import smoke

    table = smoke(workdir=tmp_path)
    runs = tmp_path / "runs"
    work = runs / "_workflow"
    assert (work / "models" / "ranking.csv").exists() and (work / "tune" / "choice.csv").exists()
    results = pd.read_csv(work / "datasets" / "results.csv")
    assert len(results) == 4  # best setup x 2 training datasets x 2 test datasets
    assert set(results["train_on"]) == {"mock", "mock-faber"} and set(results["test_on"]) == {"mock:test", "mock-faber:test"}
    assert results["flux_scatter"].notna().all() and "mock-faber" in table
    assert len(list(work.glob("*/done/*/*.json"))) == 2 + 2 + 2
    assert len(list(runs.glob("*/*/eval/*/plots/*.png"))) >= 4 * 4 + 2 * 4
    assert not list(tmp_path.rglob(".*.tmp*")), "no temporary files are left behind"
    # a second start finds everything done and changes nothing
    before = (work / "datasets" / "results.csv").stat().st_mtime
    smoke(fresh=False, workdir=tmp_path)
    assert (work / "datasets" / "results.csv").stat().st_mtime == before
