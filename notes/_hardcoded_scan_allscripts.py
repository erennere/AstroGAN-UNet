import ast
import re
from pathlib import Path

root = Path(r"d:/astro_images/physik_thesis/scripts/astroUnets")
py_targets = [root / "starter.py"]
for folder in ["src/data", "src/training", "src/evaluation", "src/visualization", "legacy"]:
    d = root / folder
    if d.exists():
        for p in d.glob("*.py"):
            if p.name != "__init__.py":
                py_targets.append(p)
sh_targets = list((root / "src/bash").glob("*.sh"))

rows = []

def _line(src, n):
    return src[n-1].strip() if 1 <= n <= len(src) else ""

def _category(value, ctx):
    s = str(value)
    c = ctx.lower()
    if any(t in s for t in ["/", "\\", ".csv", ".parquet", ".fits", ".png", ".keras", ".json", ".yml", ".yaml", ".sh"]):
        return "path-or-filename"
    if any(t in c for t in ["epochs", "batch", "learning_rate", "dropout", "sigma", "npixels", "footprint", "threshold", "stride", "patch", "workers", "maxiters", "frac", "lr", "rad", "size"]):
        return "hyperparameter-or-threshold"
    if any(t in c for t in ["slurm", "sbatch", "partition", "gres", "mem", "time", "cpus", "nodes"]):
        return "scheduler-resource"
    if isinstance(value, str) and value in {"gan", "unet", "sigmoid", "z_scale", "log_min_max", "min_max", "average", "matched", "sequential", "parallel", "array"}:
        return "mode-or-enum"
    return "literal"

class Scan(ast.NodeVisitor):
    def __init__(self, path, src):
        self.path = path
        self.src = src
        self.parents = []

    def visit(self, node):
        self.parents.append(node)
        super().visit(node)
        self.parents.pop()

    def visit_Constant(self, node):
        val = node.value
        if isinstance(val, bool) or val is None:
            return
        parent = self.parents[-2] if len(self.parents) > 1 else None
        if not isinstance(parent, (ast.Assign, ast.AnnAssign, ast.keyword)):
            return
        ln = getattr(node, "lineno", None)
        if not ln:
            return
        ctx = _line(self.src, ln)
        if isinstance(val, str):
            if len(val.strip()) == 0 or len(val) > 120:
                return
            if not (
                any(x in val for x in ["/", "\\", ".", "_", "--", "AUN_", "SLURM_"])
                or val in {"gan", "unet", "sigmoid", "z_scale", "log_min_max", "min_max", "average", "matched", "sequential", "parallel", "array"}
            ):
                return
        if isinstance(val, (int, float)) and val in {0, 1, -1}:
            if not any(k in ctx.lower() for k in ["workers", "batch", "epoch", "sigma", "threshold", "stride", "patch", "dropout", "rate", "time", "mem", "size", "rad"]):
                return
        rows.append(("python", str(self.path.relative_to(root)).replace('\\', '/'), ln, repr(val), ctx, _category(val, ctx)))

for p in py_targets:
    try:
        txt = p.read_text(encoding="utf-8")
        Scan(p, txt.splitlines()).visit(ast.parse(txt))
    except Exception:
        pass

assign_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
for p in sh_targets:
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    rel = str(p.relative_to(root)).replace('\\', '/')
    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#SBATCH"):
            rows.append(("bash", rel, i, line, line, "scheduler-resource"))
            continue
        if line.startswith("#"):
            continue
        m = assign_re.match(raw)
        if not m:
            continue
        rhs = m.group(2).strip()
        if not rhs:
            continue
        if any(ch.isdigit() for ch in rhs) or any(tok in rhs for tok in ["/", ".", "%", "gpu", "cpu", "AUN_", "SLURM", "true", "false"]):
            rows.append(("bash", rel, i, rhs[:120], line, _category(rhs, line)))

seen = set()
uniq = []
for r in rows:
    key = (r[0], r[1], r[2], r[3])
    if key in seen:
        continue
    seen.add(key)
    uniq.append(r)

uniq.sort(key=lambda x: (x[1], x[2], x[0]))

full = root / "notes" / "hardcoded_values_allscripts_full.md"
with full.open("w", encoding="utf-8") as f:
    f.write("# Hardcoded Values Audit (All Scripts, Full)\n\n")
    f.write("| Category | Lang | File | Line | Hardcoded value | Context |\n")
    f.write("| --- | --- | --- | ---: | --- | --- |\n")
    for r in uniq:
        cat, lang, file, line, val, ctx = r[5], r[0], r[1], r[2], r[3], r[4]
        val_s = str(val).replace('|', '\\|')
        ctx_s = str(ctx).replace('|', '\\|')
        f.write(f"| {cat} | {lang} | {file} | {line} | {val_s} | {ctx_s} |\n")

by_file = {}
for r in uniq:
    by_file.setdefault(r[1], []).append(r)
curated = []
for f, items in by_file.items():
    pri = [x for x in items if x[5] != "literal"]
    lit = [x for x in items if x[5] == "literal"]
    curated.extend(pri[:60])
    curated.extend(lit[:15])
curated.sort(key=lambda x: (x[1], x[2], x[0]))

out = root / "notes" / "hardcoded_values_allscripts_priority.md"
with out.open("w", encoding="utf-8") as f:
    f.write("# Hardcoded Values Audit (All Scripts, Priority)\n\n")
    f.write("| Category | Lang | File | Line | Hardcoded value | Context |\n")
    f.write("| --- | --- | --- | ---: | --- | --- |\n")
    for r in curated:
        cat, lang, file, line, val, ctx = r[5], r[0], r[1], r[2], r[3], r[4]
        val_s = str(val).replace('|', '\\|')
        ctx_s = str(ctx).replace('|', '\\|')
        f.write(f"| {cat} | {lang} | {file} | {line} | {val_s} | {ctx_s} |\n")

print(f"all scripts full rows: {len(uniq)}")
print(f"all scripts priority rows: {len(curated)}")
print(full)
print(out)
