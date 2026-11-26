"""Stage `train`: one run = one config = one folder runs/<dataset>/<run_name>-<hash8>/.

Run folder contents
  config.yaml          exact configuration
  history.csv          one row per epoch (epoch means; validation on a fixed set)
  train.json           written when training ends: status (finished | stopped_early | diverged),
                       epochs trained, best epoch and loss. Its presence means "trained".
  checkpoints/best.keras  generator with the lowest validation reconstruction loss
  checkpoints/best.json   its epoch and loss
  checkpoints/state.*  full training state (generator, discriminator, optimizers, epoch, best loss)
  previews/epoch_XXXX.png
Resuming continues from the last finished epoch with the best loss restored. With train.patience
set, training stops once the validation loss has not improved for that many epochs. A run whose
losses become NaN or infinite stops as "diverged" and keeps its last good state.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from astrounets.config import Config, ConfigError, identity_hash, lr_at, read_yaml, save_config
from astrounets.data.dataset import load_crops
from astrounets.data.fits_io import CropStore
from astrounets.data.noise import usable
from astrounets.data.pairs import fixed_set, training_stream
from astrounets.files import atomic, write_json

log = logging.getLogger(__name__)
VAL_SEED = 1


def _check_run_folder(cfg: Config) -> None:
    saved = cfg.run_dir / "config.yaml"
    if saved.exists():
        from astrounets.config import from_dict

        other = from_dict(read_yaml(saved), cfg.root)
        if identity_hash(other) != cfg.run_hash:
            raise ConfigError(f"{saved} belongs to a different setup; refusing to mix runs")


def train_result(run_dir: Path) -> dict | None:
    """Contents of train.json, or None while the run is not trained."""
    path = Path(run_dir) / "train.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def train(cfg: Config) -> Path:
    """Train (or resume) the run of `cfg`."""
    import tensorflow as tf

    from astrounets.models import build, optimizers
    from astrounets.plots import save_triptychs

    run = cfg.run_dir
    if train_result(run) is not None:
        log.info("run already trained: %s", run)
        return run
    ckpt = run / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    _check_run_folder(cfg)
    save_config(cfg, run / "config.yaml")
    handler = logging.FileHandler(run / "train.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    try:
        keras.utils.set_random_seed(cfg.seed)
        crops = usable(load_crops(cfg), cfg.noise)
        train_rows = crops[crops["split"] == "train"].reset_index(drop=True)
        val_rows = crops[crops["split"] == "val"].reset_index(drop=True)
        if train_rows.empty or val_rows.empty:
            raise RuntimeError("no usable train or val crops; check dataset and noise settings")
        store = CropStore(cfg.mast_dir, cfg.dataset.hdu, cfg.dataset.weight_hdu)
        val = fixed_set(val_rows, store, cfg, cfg.train.val_samples, VAL_SEED)
        log.info("run %s: %d train crops, %d val crops, %d fixed val samples",
                 run, len(train_rows), len(val_rows), len(val["x"]))

        model, generator, discriminator = build(cfg)
        g_opt, d_opt = optimizers(cfg)
        model.compile(g_opt, d_opt)
        g_opt.build(generator.trainable_variables)
        state_parts = {"generator": generator, "g_opt": g_opt}
        if discriminator is not None:
            d_opt.build(discriminator.trainable_variables)
            state_parts.update(discriminator=discriminator, d_opt=d_opt)
        progress = {"epoch": tf.Variable(0, dtype=tf.int64), "best": tf.Variable(math.inf, dtype=tf.float64),
                    "best_epoch": tf.Variable(0, dtype=tf.int64)}
        state = tf.train.Checkpoint(**progress, **state_parts)
        prefix = str(ckpt / "state")
        if cfg.train.resume and Path(prefix + ".index").exists():
            state.read(prefix).assert_existing_objects_matched()
            log.info("resumed at epoch %d (best validation loss %.6g)", int(progress["epoch"].numpy()),
                     float(progress["best"].numpy()))
        start = int(progress["epoch"].numpy())
        recorder = RunRecorder(cfg, run, generator, state, prefix, progress, val, start, save_triptychs)
        if start < cfg.train.epochs and not recorder.out_of_patience(start):
            ps, bs = cfg.dataset.patch_size, cfg.train.batch_size
            spec = tf.TensorSpec((ps, ps, 1), tf.float32)
            train_ds = (tf.data.Dataset.from_generator(lambda: training_stream(train_rows, store, cfg, start),
                                                       output_signature=(spec, spec))
                        .batch(bs, drop_remainder=True).prefetch(2))
            val_ds = tf.data.Dataset.from_tensor_slices((val["x"], val["y"])).batch(bs)
            model.fit(train_ds, epochs=cfg.train.epochs, initial_epoch=start,
                      steps_per_epoch=cfg.train.samples_per_epoch // bs, validation_data=val_ds,
                      callbacks=[LearningRate(cfg, g_opt, d_opt), recorder], shuffle=False, verbose=2)
        store.close()
        result = recorder.result()
        write_json(run / "train.json", result)
        log.info("training %s: %s", result["status"], result)
        return run
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


class LearningRate(keras.callbacks.Callback):
    """Generator rate from train.learning_rate / lr_drops; the discriminator follows at gan.d_lr_ratio."""

    def __init__(self, cfg: Config, g_opt, d_opt):
        super().__init__()
        self.cfg, self.g_opt, self.d_opt = cfg, g_opt, d_opt

    def on_epoch_begin(self, epoch, logs=None):
        rate = lr_at(self.cfg.train, epoch)
        self.g_opt.learning_rate.assign(rate)
        if self.d_opt is not None:
            self.d_opt.learning_rate.assign(rate * self.cfg.gan.d_lr_ratio)


class RunRecorder(keras.callbacks.Callback):
    """Per epoch: divergence check, best generator, full state, history row, previews, early stop."""

    def __init__(self, cfg, run, generator, state, prefix, progress, val, start, save_triptychs):
        super().__init__()
        self.cfg, self.run, self.generator, self.state, self.prefix = cfg, run, generator, state, prefix
        self.progress, self.val, self.save_triptychs = progress, val, save_triptychs
        self.status = "finished"
        history = run / "history.csv"
        if history.exists():  # drop rows written after the last saved state (crash between the two)
            rows = pd.read_csv(history)
            rows[rows["epoch"] <= start].to_csv(history, index=False)
        self.t0 = time.time()

    def out_of_patience(self, done: int) -> bool:
        patience = self.cfg.train.patience
        if patience is not None and done - int(self.progress["best_epoch"].numpy()) >= patience:
            self.status = "stopped_early"
            return True
        return False

    def result(self) -> dict:
        best_path = self.run / "checkpoints" / "best.json"
        best = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
        return {"status": self.status, "epochs": int(self.progress["epoch"].numpy()),
                "best_epoch": best.get("epoch"), "best_val_rec_loss": best.get("val_rec_loss")}

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = {k: float(v) for k, v in dict(logs or {}).items() if k != "learning_rate"}
        done = epoch + 1
        if not all(np.isfinite(v) for v in logs.values()):
            log.error("epoch %d: non-finite losses %s; stopping (the last good state is kept)", done, logs)
            self.status = "diverged"
            self.model.stop_training = True
            return
        val_loss = logs["val_rec_loss"]
        ckpt = self.run / "checkpoints"
        if val_loss < float(self.progress["best"].numpy()):
            self.progress["best"].assign(val_loss)
            self.progress["best_epoch"].assign(done)
            with atomic(ckpt / "best.keras") as tmp:
                self.generator.save(tmp)
            write_json(ckpt / "best.json", {"epoch": done, "val_rec_loss": val_loss})
        self.progress["epoch"].assign(done)
        self.state.write(self.prefix)
        row = {"epoch": done, "lr": lr_at(self.cfg.train, epoch), "seconds": round(time.time() - self.t0, 1), **logs}
        history = self.run / "history.csv"
        pd.DataFrame([row]).to_csv(history, mode="a", header=not history.exists(), index=False)
        if done % self.cfg.train.preview_every == 0 or done == self.cfg.train.epochs:
            self._preview(done)
        if self.out_of_patience(done):
            log.info("no improvement for %d epochs; stopping after epoch %d", self.cfg.train.patience, done)
            self.model.stop_training = True

    def _preview(self, done: int) -> None:
        from astrounets.data.scaling import Scaler

        k = min(self.cfg.train.preview_samples, len(self.val["x"]))
        scaler = Scaler(self.cfg.scaling.mode, self.cfg.scaling.log_clip_max)
        pred = self.generator.predict(self.val["x"][:k], verbose=0)[..., 0]
        rows = []
        for i in range(k):
            rec = scaler.inverse(pred[i], self.val["params"][i])
            rec[self.val["noisy"][i] == 0] = 0.0
            rows.append((self.val["noisy"][i], rec, self.val["clean"][i]))
        self.save_triptychs(self.run / "previews" / f"epoch_{done:04d}.png", rows, self.cfg.plots.dpi)
