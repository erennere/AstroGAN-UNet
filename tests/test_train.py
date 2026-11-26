import json

import numpy as np
import pandas as pd
import pytest

from astrounets.config import ConfigError, replace
from astrounets.data.dataset import load_crops
from astrounets.data.fits_io import CropStore
from astrounets.data.noise import usable
from astrounets.data.pairs import draw_weights, fixed_set, training_stream
from astrounets.train import train, train_result


def _rows(cfg, split):
    crops = usable(load_crops(cfg), cfg.noise)
    return crops[crops["split"] == split].reset_index(drop=True)


def test_validation_set_is_identical_every_time(mock_data):
    store = CropStore(mock_data.mast_dir, "SCI")
    a = fixed_set(_rows(mock_data, "val"), store, mock_data, 8, seed=1)
    b = fixed_set(_rows(mock_data, "val"), store, mock_data, 8, seed=1)
    assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["y"], b["y"])


def test_training_stream_is_reproducible_and_scaled_from_noisy(mock_data):
    store = CropStore(mock_data.mast_dir, "SCI")
    rows = _rows(mock_data, "train")
    first = [x for x, _ in zip(training_stream(rows, store, mock_data, 0), range(6))]
    again = [x for x, _ in zip(training_stream(rows, store, mock_data, 0), range(6))]
    later = [x for x, _ in zip(training_stream(rows, store, mock_data, 1), range(6))]
    assert all(np.array_equal(p[0], q[0]) for p, q in zip(first, again))
    assert not all(np.array_equal(p[0], q[0]) for p, q in zip(first, later)), "each epoch draws new samples"
    for x, y in first:  # z-scaled with the noisy image's mean and std: x has mean 0, std 1; y does not
        assert x.shape == (64, 64, 1) and abs(x.mean()) < 1e-4 and x.std() == pytest.approx(1, rel=1e-3)


@pytest.mark.parametrize("model_type", ["unet", "gan"])
def test_train_resume_and_outputs(mock_data, model_type):
    cfg = replace(mock_data, {"model": {"type": model_type}, "train": {"epochs": 2, "samples_per_epoch": 16,
                                                                       "learning_rate": 0.0002, "lr_drops": [0.5]}})
    run = train(cfg)
    history = pd.read_csv(run / "history.csv")
    assert history["epoch"].tolist() == [1, 2] and history["lr"].tolist() == [0.0002, 0.0001]
    best = json.loads((run / "checkpoints" / "best.json").read_text())
    assert best["val_rec_loss"] == pytest.approx(history["val_rec_loss"].min())
    assert (run / "checkpoints" / "best.keras").exists() and (run / "previews" / "epoch_0002.png").exists()
    result = train_result(run)
    assert result["status"] == "finished" and result["epochs"] == 2 and result["best_epoch"] == best["epoch"]
    assert not list(run.rglob(".*.tmp*")), "no temporary files are left behind"

    train(cfg)  # already finished: nothing happens, history unchanged
    assert pd.read_csv(run / "history.csv")["epoch"].tolist() == [1, 2]


def test_resume_continues_from_last_epoch(mock_data, monkeypatch):
    import astrounets.train as tr

    cfg = replace(mock_data, {"train": {"epochs": 3, "samples_per_epoch": 16, "lr_drops": []}})
    original = tr.RunRecorder.on_epoch_end

    def stop_after_two(self, epoch, logs=None):
        original(self, epoch, logs)
        if epoch + 1 == 2:
            self.model.stop_training = True

    monkeypatch.setattr(tr.RunRecorder, "on_epoch_end", stop_after_two)
    run = train(cfg)
    assert pd.read_csv(run / "history.csv")["epoch"].tolist() == [1, 2]
    (run / "train.json").unlink()  # as if the job had been killed after epoch 2
    best_before = json.loads((run / "checkpoints" / "best.json").read_text())
    monkeypatch.setattr(tr.RunRecorder, "on_epoch_end", original)
    train(cfg)
    history = pd.read_csv(run / "history.csv")
    assert history["epoch"].tolist() == [1, 2, 3]
    best_after = json.loads((run / "checkpoints" / "best.json").read_text())
    assert best_after["val_rec_loss"] <= best_before["val_rec_loss"]


def test_run_folder_cannot_be_reused_by_another_setup(mock_data, tmp_path):
    cfg = replace(mock_data, {"train": {"epochs": 1, "samples_per_epoch": 8, "lr_drops": []}})
    run = cfg.run_dir
    run.mkdir(parents=True, exist_ok=True)
    from astrounets.config import save_config
    save_config(replace(cfg, {"seed": 99}), run / "config.yaml")
    with pytest.raises(ConfigError, match="different setup"):
        train(cfg)


def _fake_val_losses(monkeypatch, losses):
    """Make RunRecorder see the given validation losses, one per epoch."""
    import astrounets.train as tr

    original = tr.RunRecorder.on_epoch_end

    def patched(self, epoch, logs=None):
        logs = dict(logs or {})
        logs["val_rec_loss"] = losses[epoch]
        original(self, epoch, logs)

    monkeypatch.setattr(tr.RunRecorder, "on_epoch_end", patched)


def test_patience_stops_training_and_survives_a_resume(mock_data, monkeypatch):
    cfg = replace(mock_data, {"train": {"epochs": 6, "samples_per_epoch": 8, "patience": 2, "lr_drops": []}})
    _fake_val_losses(monkeypatch, [1.0, 0.5, 0.6, 0.7, 0.1, 0.1])
    run = train(cfg)
    result = train_result(run)
    assert result == {"status": "stopped_early", "epochs": 4, "best_epoch": 2, "best_val_rec_loss": 0.5}
    assert pd.read_csv(run / "history.csv")["epoch"].tolist() == [1, 2, 3, 4]
    (run / "train.json").unlink()
    train(cfg)  # a resumed run that is already out of patience trains no further
    assert train_result(run)["status"] == "stopped_early" and pd.read_csv(run / "history.csv")["epoch"].max() == 4


def test_diverged_run_stops_and_keeps_its_last_good_state(mock_data, monkeypatch):
    cfg = replace(mock_data, {"train": {"epochs": 4, "samples_per_epoch": 8, "lr_drops": [], "beta_1": 0.6}})
    _fake_val_losses(monkeypatch, [0.5, float("nan"), 0.1, 0.1])
    run = train(cfg)
    result = train_result(run)
    assert result["status"] == "diverged" and result["epochs"] == 1 and result["best_epoch"] == 1
    assert pd.read_csv(run / "history.csv")["epoch"].tolist() == [1]


def test_crowded_crops_are_capped_in_training_draws():
    rows = pd.DataFrame({"crowded": [True] * 60 + [False] * 40})
    w = draw_weights(rows, 0.25)
    assert w.sum() == pytest.approx(1.0) and w[:60].sum() == pytest.approx(0.25)
    assert len(set(w[:60])) == 1 and len(set(w[60:])) == 1  # equal within each group
    few = pd.DataFrame({"crowded": [True] * 10 + [False] * 90})
    assert draw_weights(few, 0.25)[:10].sum() == pytest.approx(0.10)  # a cap, not a quota
    assert draw_weights(rows, 1.0)[:60].sum() == pytest.approx(0.60)  # 1 = every crop equally
    for only in (pd.DataFrame({"crowded": [False] * 5}), pd.DataFrame({"crowded": [True] * 5}), pd.DataFrame({"x": [1] * 5})):
        assert np.allclose(draw_weights(only, 0.25), 0.2)


def test_draws_on_the_mock_sky_follow_the_cap(mock_data):
    crops = usable(load_crops(mock_data), mock_data.noise)
    rows = crops[crops["split"] == "train"].reset_index(drop=True)
    assert rows["crowded"].mean() > 0.1
    capped = replace(mock_data, {"train": {"max_crowded_share": 0.05}})
    w = draw_weights(rows, capped.train.max_crowded_share)
    rng = np.random.default_rng(0)
    drawn = rows["crowded"].to_numpy()[rng.choice(len(rows), size=20000, p=w)]
    assert drawn.mean() == pytest.approx(0.05, abs=0.01)
