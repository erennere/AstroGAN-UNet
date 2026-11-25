---
name: config-propagation-audit
description: "Use when auditing or fixing config propagation across config.yaml, starter.py, and stage entrypoints; use for missing propagated fields, runtime source wiring, or config ownership mismatches."
---

# Config Propagation Audit

Audit and fix config propagation end-to-end with strict config-driven rules.

## Enforced Policy

- Runtime behavior must be config-driven.
- Do not introduce hardcoded runtime values unless the user explicitly confirms the exception.
- Do not introduce hasattr() checks unless the user explicitly confirms the exception.
- Do not introduce dict.get() fallback access for required runtime contract fields unless the user explicitly confirms the exception.
- Do not add legacy compatibility branches unless the user explicitly confirms they are required.

## Inputs

- Target field names (for example: uncropped_single_parallel, uncropped_write_histograms).
- Optional target stages (data, training, evaluation, visualization).

## Workflow

1. Identify ownership
- Locate each target field in config.yaml.
- Confirm owning section and any runtime-source comments.

2. Trace resolution
- In starter.py, verify runtime-source extraction and section finalizers propagate required values.
- Verify derived structures (for example kwargs_source, data_kwargs, model_kwargs) include target fields when downstream code expects them.

3. Trace consumption
- Find all consumers in src.
- Confirm each consumer reads the same canonical field path and does not reintroduce hardcoded defaults.

4. Patch minimally
- Apply the smallest change set that preserves architecture.
- Prefer wiring through central resolution/finalization over duplicating bindings in stage scripts.

5. Validate
- Run focused tests first, then broader tests if needed.
- Report exactly which fields were missing and where they were added.

## Checklist

- Field exists in config.yaml under the correct owner section.
- Runtime-source comment exists when value is resolved from another section.
- starter.py resolves value into final runtime section.
- starter.py finalizer forwards value into any derived dict consumed by stages.
- Stage consumer accesses resolved field without hardcoded fallback.
- Tests cover both direct section access and derived dict access when both are used.

## Output Format

- Findings first, ordered by severity.
- For each finding: source path, missing propagation hop, consumer impact.
- Then list exact edits made.
- Then list tests run and remaining risks.
