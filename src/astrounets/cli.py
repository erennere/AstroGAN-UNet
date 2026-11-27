"""Command line: `astrounets <command> [-c CONFIG] [--set key=value ...]`.

The whole study
  workflow   fetch, datasets, stage 1, tuning, stage 2 - in the background; continues where it stopped
  status     progress of each stage and the end of the workflow log
  stop       stop the background workflow (its runs resume later)
  report     study figures, tables and the final models (the workflow's last step)

Using a trained model
  denoise    reconstruct real FITS images of any size (default: the study's final model)

Single steps (for checks and debugging)
  fetch      query MAST once, assign sky tiles and splits, download
  dataset    select images for one dataset and list its usable crops
  train      train the run defined by the config
  evaluate   evaluate a run on frozen noisy inputs (--on dataset:split, repeatable)
  plots      draw figures for a run (--detailed: also the hexbin family)
  run        train + evaluate + plots
  info       print the resolved config and the run folder
  smoke      the whole workflow on a synthetic sky (a few minutes on a CPU)

Used by the workflow itself: job, select.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml


def _config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", type=Path, default=None, help="config file (default: configs/default.yaml)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override one value, e.g. --set train.epochs=5 (repeatable)")


def _on_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--on", action="append", default=[], metavar="DATASET:SPLIT",
                   help="evaluation target; DATASET may be 'own' (default: own:<eval.split>)")


def parser() -> argparse.ArgumentParser:
    from astrounets.launch import UNTIL
    from astrounets.workflow import SELECTIONS, STAGES

    ap = argparse.ArgumentParser(prog="astrounets", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("workflow", help="run the whole study in the background"); _config_args(p)
    p.add_argument("--local", action="store_true", help="run on this machine even if SLURM is available")
    p.add_argument("--foreground", action="store_true", help="stay attached and print the log here")
    p.add_argument("--dry-run", action="store_true", help="only list the jobs that would run")
    p.add_argument("--until", choices=UNTIL, default=None, help="stop after this stage")
    _config_args(sub.add_parser("status", help="progress of the workflow"))
    _config_args(sub.add_parser("stop", help="stop the background workflow"))
    p = sub.add_parser("fetch", help="query MAST, assign sky tiles and splits, download"); _config_args(p)
    p.add_argument("--refresh", action="store_true", help="query MAST again even if metadata exists")
    _config_args(sub.add_parser("dataset", help="build data/<dataset.name>/crops.parquet"))
    _config_args(sub.add_parser("train", help="train one run"))
    p = sub.add_parser("evaluate", help="evaluate a run"); _config_args(p); _on_args(p)
    p.add_argument("--run", type=Path, default=None, help="run folder (default: the config's run)")
    p = sub.add_parser("plots", help="draw the figures of a run"); _config_args(p)
    p.add_argument("--run", type=Path, default=None)
    p.add_argument("--detailed", action="store_true", help="also the hexbin figures (flux ratio, Kron radius, ...)")
    p = sub.add_parser("report", help="study figures, tables and final models in runs/_workflow/report"); _config_args(p)
    p = sub.add_parser("run", help="train + evaluate + plots"); _config_args(p); _on_args(p)
    p = sub.add_parser("denoise", help="reconstruct real FITS images with a trained model"); _config_args(p)
    p.add_argument("images", nargs="+", metavar="IMAGE", help="FITS files, folders or patterns such as 'data/*_drz.fits'")
    p.add_argument("--model", default=None, help="model folder, or 'best' (default: denoise.model)")
    p.add_argument("--out", type=Path, default=None, help="output folder (default: denoise.out)")
    p.add_argument("--overlap", type=int, default=None, help="tile overlap in pixels (default: denoise.overlap)")
    p.add_argument("--batch-size", type=int, default=None, help="tiles per model call (default: denoise.batch_size)")
    p.add_argument("--tta", action="store_true", default=None, help="average over the 8 rotations and flips")
    p.add_argument("--no-figure", dest="figure", action="store_false", default=None, help="no comparison figure")
    p.add_argument("--zoom", action="append", default=[], metavar="X,Y",
                   help="centre (pixels, from 0) of a full-resolution cutout in the figure (repeatable)")
    p.add_argument("--show", action="store_true", help="also show original and reconstruction in a window")
    _config_args(sub.add_parser("info", help="print the resolved config and run folder"))
    p = sub.add_parser("smoke", help="whole workflow on a synthetic sky")
    p.add_argument("--keep", action="store_true", help="keep the results of an earlier smoke run")
    p = sub.add_parser("job", help=argparse.SUPPRESS); _config_args(p)
    p.add_argument("stage", choices=STAGES); p.add_argument("key")
    p = sub.add_parser("select", help=argparse.SUPPRESS); _config_args(p)
    p.add_argument("stage", choices=SELECTIONS)
    return ap


def _config_path(args) -> Path:
    from astrounets.config import default_config_path

    return args.config or default_config_path()


def _load(args):
    from astrounets.config import load_config

    return load_config(_config_path(args), args.overrides)


def _targets(cfg, items):
    return items or [f"own:{cfg.eval.split}"]


def _centre(text: str) -> tuple[int, int]:
    from astrounets.config import ConfigError

    try:
        x, y = (int(float(v)) for v in text.split(","))
    except ValueError:
        raise ConfigError(f"--zoom expects X,Y in pixels, got {text!r}") from None
    return x, y


def main(argv: list[str] | None = None) -> int:
    from astrounets.config import ConfigError

    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    logging.getLogger("absl").setLevel(logging.WARNING)
    try:
        return _dispatch(args)
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


def _dispatch(args) -> int:
    cmd = args.command

    if cmd == "workflow":
        from astrounets.launch import start
        return start(_config_path(args), args.overrides, args.local, args.foreground, args.dry_run, args.until)
    if cmd == "status":
        from astrounets.launch import status
        print(status(_config_path(args), args.overrides))
    elif cmd == "stop":
        from astrounets.launch import stop
        return stop(_config_path(args), args.overrides)
    elif cmd == "fetch":
        from astrounets.data.mast import fetch
        print(fetch(_load(args), refresh=args.refresh))
    elif cmd == "dataset":
        from astrounets.data.dataset import build_dataset
        print(build_dataset(_load(args)))
    elif cmd == "train":
        from astrounets.train import train
        print(train(_load(args)))
    elif cmd == "evaluate":
        from astrounets.evaluate.run import evaluate
        from astrounets.workflow import targets
        cfg = _load(args)
        for dataset, split in targets(cfg, _targets(cfg, args.on)):
            print(evaluate(cfg, args.run, dataset, split))
    elif cmd == "plots":
        from astrounets.plots import make_plots
        cfg = _load(args)
        for path in make_plots(cfg, args.run or cfg.run_dir, args.detailed):
            print(path)
    elif cmd == "report":
        from astrounets.report import make_report
        print(make_report(_load(args)))
    elif cmd == "denoise":
        from astrounets.config import _deep_merge, load_config, parse_overrides
        from astrounets.infer import denoise_files
        chosen = {"model": None if args.model is None else
                  (args.model if args.model.strip().lower() == "best" else str(Path(args.model).resolve())),
                  "out": None if args.out is None else str(args.out.resolve()), "overlap": args.overlap,
                  "batch_size": args.batch_size, "tta": args.tta, "figure": args.figure}
        overrides = _deep_merge(parse_overrides(args.overrides), {"denoise": {k: v for k, v in chosen.items() if v is not None}})
        cfg = load_config(_config_path(args), overrides)
        for path in denoise_files(cfg, args.images, [_centre(z) for z in args.zoom], args.show):
            print(path)
    elif cmd == "run":
        from astrounets.workflow import run_one
        cfg = _load(args)
        print(run_one(cfg, _targets(cfg, args.on)))
    elif cmd == "info":
        from astrounets.config import to_dict
        cfg = _load(args)
        print(yaml.safe_dump(to_dict(cfg), sort_keys=False))
        print(f"run folder: {cfg.run_dir}")
    elif cmd == "smoke":
        from astrounets.smoke import smoke
        print(smoke(fresh=not args.keep))
    elif cmd == "job":
        from astrounets.workflow import job
        print(job(_load(args), args.stage, args.key))
    elif cmd == "select":
        from astrounets.workflow import select
        print(select(_load(args), args.stage))
    return 0


if __name__ == "__main__":
    sys.exit(main())
