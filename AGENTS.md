# AGENTS

Repository-wide instructions for AI coding agents working in AstroGAN-UNet.

## Scope and Priorities

- Treat this repository as config-first. Runtime behavior must come from config values, not hardcoded literals.
- Prefer minimal, localized changes that preserve existing architecture and public interfaces.
- Active pipeline lives under src. Legacy scripts under legacy are historical unless explicitly requested.

## Hard Rules (User-Enforced)

- Everything must be config-driven.
- Hardcoded runtime values are exceptional and require explicit user confirmation.
- Usage of hasattr() is exceptional and requires explicit user confirmation.
- Usage of dict.get() defaults for required runtime contract fields is exceptional and requires explicit user confirmation.
- No legacy compatibility shims are allowed unless the user explicitly confirms they want them.

## Config Contract

- Canonical config contract is config.yaml resolved through starter.py.
- For required config keys:
  - Read using strict indexing, not soft fallbacks.
  - Missing keys should fail fast with clear errors.
- Keep starter.py orchestration-focused:
  - Resolve and propagate shared config values centrally.
  - Do not duplicate config wiring across stage scripts.

References:
- [README.md](README.md)
- [src/README.md](src/README.md)

## Canonical Pipeline Order

Run stages in this order unless user requests otherwise:

1. python -m src.data.mast
2. python -m src.data.create_dataset
3. python -m src.training.new_train
4. python -m src.evaluation.metrics
5. python -m src.evaluation.uncropped_metrics
6. python -m src.evaluation.merge_catalogs
7. python -m src.visualization.prepare_images
8. python -m src.visualization.prepare_plots

Reference: [src/README.md](src/README.md)

## Working Directory and Path Safety

- Stage entrypoints should execute from repository root so path templates resolve correctly.
- Be careful with scripts that change cwd; keep behavior consistent with existing stage conventions.

References:
- [README.md](README.md)
- [src/README.md](src/README.md)

## Testing and Validation

- Use synthetic tests under tests; do not rely on repository data or models for unit/integration validation.
- Preferred test commands:
  - pytest --tb=short -q
  - pytest -m unit --tb=short -q
  - pytest -m integration --tb=short -q
  - pytest -m slow --tb=short -q

Reference: [tests/README.md](tests/README.md)

## Change Guidelines

- For new runtime options, add them to config.yaml and propagate through starter.py finalization/resolution, then consume in stage code.
- Avoid introducing alternate code paths for backward compatibility unless explicitly requested.
- Update or add focused tests for any new config propagation or stage behavior.

## Where to Read More

- Root architecture and workflow: [README.md](README.md)
- Source tree and run order: [src/README.md](src/README.md)
- Stage details:
  - [src/data/README.md](src/data/README.md)
  - [src/training/README.md](src/training/README.md)
  - [src/evaluation/README.md](src/evaluation/README.md)
  - [src/visualization/README.md](src/visualization/README.md)
  - [src/bash/README.md](src/bash/README.md)

## Repo Customizations

- Skill for propagation audits: [.github/skills/config-propagation-audit/SKILL.md](.github/skills/config-propagation-audit/SKILL.md)
- Skill for strict compliance reviews: [.github/skills/strict-config-review/SKILL.md](.github/skills/strict-config-review/SKILL.md)
- Prompt for config-first refactors: [.github/prompts/config-first-refactor.prompt.md](.github/prompts/config-first-refactor.prompt.md)
