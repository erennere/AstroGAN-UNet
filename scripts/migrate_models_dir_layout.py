#!/usr/bin/env python3
"""Migrate model directories from old layout to new layout.

Old layout:
    <models_root>/<is_gan>/<use_attention>/<loss>/<data_alias>/<scaling>/DO.../ACT.../OUT.../DACT.../DOUT...

New layout:
    <models_root>/<data_alias>/<is_gan>/<use_attention>/<loss>/<scaling>/DO.../ACT.../OUT.../DACT.../DOUT...

Dry-run by default. Use --execute to apply filesystem changes.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

OLD_LAYOUT_RE = re.compile(
    r"^(?P<is_gan>GAN|UNET)/(?P<use_attention>ATTN|NOATTN)/(?P<loss_function>[^/]+)/"
    r"(?P<data_alias_enriched_hex>[^/]+)/(?P<scaling_tag>[^/]+)/"
    r"DO(?P<dropout_tag>[^/]+)/ACT(?P<activation_tag>[^/]+)/OUT(?P<output_activation_tag>[^/]+)/"
    r"DACT(?P<discriminator_activation_tag>[^/]+)/DOUT(?P<discriminator_output_activation_tag>[^/]+)$"
)


def build_new_relative_path(old_relative: Path) -> Path | None:
    normalized = old_relative.as_posix().strip("/")
    match = OLD_LAYOUT_RE.fullmatch(normalized)
    if match is None:
        return None

    parts = match.groupdict()
    return Path(
        parts["data_alias_enriched_hex"],
        parts["is_gan"],
        parts["use_attention"],
        parts["loss_function"],
        parts["scaling_tag"],
        f"DO{parts['dropout_tag']}",
        f"ACT{parts['activation_tag']}",
        f"OUT{parts['output_activation_tag']}",
        f"DACT{parts['discriminator_activation_tag']}",
        f"DOUT{parts['discriminator_output_activation_tag']}",
    )


def find_old_layout_leaf_dirs(models_root: Path) -> list[Path]:
    found: list[Path] = []
    for path in models_root.rglob("DOUT*"):
        if not path.is_dir():
            continue
        rel = path.relative_to(models_root)
        if build_new_relative_path(rel) is not None:
            found.append(path)
    return sorted(found)


def migrate(models_root: Path, execute: bool) -> tuple[int, int, int]:
    candidates = find_old_layout_leaf_dirs(models_root)
    moved = 0
    skipped = 0

    for old_leaf in candidates:
        old_rel = old_leaf.relative_to(models_root)
        new_rel = build_new_relative_path(old_rel)
        if new_rel is None:
            skipped += 1
            continue

        new_leaf = models_root / new_rel

        if old_leaf == new_leaf:
            skipped += 1
            continue

        if new_leaf.exists():
            print(f"SKIP existing target: {new_leaf}")
            skipped += 1
            continue

        print(f"MOVE {old_leaf} -> {new_leaf}")
        if execute:
            new_leaf.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_leaf), str(new_leaf))
            moved += 1

    if not execute:
        moved = 0

    return len(candidates), moved, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate old models_dir layout to new data-first layout.")
    parser.add_argument(
        "--models-root",
        default="models",
        help="Path to models root directory (default: models)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply changes. Without this flag, runs in dry-run mode.",
    )
    args = parser.parse_args()

    models_root = Path(args.models_root).resolve()
    if not models_root.exists() or not models_root.is_dir():
        print(f"ERROR: models root does not exist or is not a directory: {models_root}")
        return 2

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"[{mode}] scanning {models_root}")
    total, moved, skipped = migrate(models_root, execute=args.execute)
    print(f"Done. candidates={total} moved={moved} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
