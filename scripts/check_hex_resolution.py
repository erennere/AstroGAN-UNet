"""
Diagnostic: show what data_alias_enriched_hex is produced for each dataset
configuration and whether it matches the AUN_DATA_ALIAS_HEX bash default.

Run from the repository root:
    python scripts/check_hex_resolution.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from starter import _encode_data_signature, _decode_alias_payload

# These match the values in build_data_args() in utils.sh.
SHARED = dict(nsigma=2, footprint_radius=10, npixels=8, filter_surveys=True)

SCENARIOS = [
    ("no filter (build_default_data_args)",  dict(filter_by_last_name=False, last_name_filter_value=None)),
    ("FABER   (build_uncropped_data_args)",  dict(filter_by_last_name=True,  last_name_filter_value="FABER")),
    ("MOMCHEVA (custom)",                    dict(filter_by_last_name=True,  last_name_filter_value="MOMCHEVA")),
]

# Hardcoded default in utils.sh → init_aun_runtime_selectors.
BASH_DEFAULT_HEX = "eNpzYWRl-M_AxMDFwAEAClcBXg"

TEMPLATE = "../metrics/cropped_modal/#/all_metrics_*.csv"

print(f"{'Scenario':<40}  {'computed hex':<40}  match bash default  resolved path prefix")
print("-" * 150)

for label, extra in SCENARIOS:
    sig = _encode_data_signature(**SHARED, allowed_survey=None, **extra)
    computed = sig["data_alias_enriched_hex"]
    match = computed == BASH_DEFAULT_HEX
    resolved = TEMPLATE.replace("#", computed)
    print(f"{label:<40}  {computed:<40}  {'YES ✓' if match else 'NO  ✗':<18}  {resolved}")

print()
print(f"AUN_DATA_ALIAS_HEX bash default: {BASH_DEFAULT_HEX}")
try:
    decoded = _decode_alias_payload(BASH_DEFAULT_HEX)
    print(f"  decodes to: {decoded}")
except Exception as exc:
    print(f"  decode failed: {exc}")
print()
print()
print("What Python uses as the '#' replacement:")
print("  - If AUN_DATA_ALIAS_HEX != 'null'  →  bash default hex is passed as CLI arg 3")
print("    and selector_cli['data_alias_enriched_hex'] is non-None, so it wins regardless")
print("    of what load_config() computed from the actual data args.")
print("  - If AUN_DATA_ALIAS_HEX == 'null'  →  Python falls back to load_config()'s")
print("    computed hex, which IS derived from the --filter-by-last-name / --last-name-filter-value flags.")
print()
print("Conclusion: '#' appears literally only if the final hex is '' or None.")
print("Wrong-name directory (e.g. FABER hex for MOMCHEVA run) means bash default won.")
