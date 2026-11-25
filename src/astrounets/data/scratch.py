"""Node-local scratch: inside a SLURM job, read images from a copy on the node's own disk.

Training reads thousands of small windows per second from the FITS files. On a cluster those files
sit on a shared file system, so with paths.node_scratch true and a job that has $TMPDIR (Helix sets
it to a fresh node-local folder per job), the images a job needs are copied there first and read
from the copy. Outside SLURM jobs nothing is copied. If the copy would not fit, the job reads from
the shared file system and says so in its log.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from astrounets.config import Config
from astrounets.files import atomic

log = logging.getLogger(__name__)


def scratch_dir() -> Path | None:
    """The job's node-local folder, or None outside a SLURM job."""
    tmp = os.environ.get("TMPDIR")
    if not os.environ.get("SLURM_JOB_ID") or not tmp or not Path(tmp).is_dir():
        return None
    return Path(tmp) / "astrounets"


def image_root(cfg: Config, rel_paths) -> Path:
    """Folder to read the given images (paths relative to data/mast) from."""
    target = scratch_dir() if cfg.paths.node_scratch else None
    if target is None:
        return cfg.mast_dir
    files = sorted(set(rel_paths))
    need = sum((cfg.mast_dir / f).stat().st_size for f in files)
    free = shutil.disk_usage(target.parent).free
    if need > 0.9 * free:
        log.warning("node scratch: %.1f GB needed but only %.1f GB free; reading from %s",
                    need / 1e9, free / 1e9, cfg.mast_dir)
        return cfg.mast_dir
    copied = 0
    for rel in files:
        src, dest = cfg.mast_dir / rel, target / rel
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            continue
        with atomic(dest) as tmp:
            shutil.copyfile(src, tmp)
        copied += 1
    log.info("node scratch: %d images (%.1f GB) in %s, %d copied now", len(files), need / 1e9, target, copied)
    return target
