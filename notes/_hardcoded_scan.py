import ast
import re
from pathlib import Path

root = Path(r"d:/astro_images/physik_thesis/scripts/astroUnets")
py_targets = [root / "starter.py"]
for folder in ["src/data", "src/training", "src/evaluation", "src/visualization"]:
    for p in (root / folder).glob("*.py"):
        if p.name != "__init__.py":
            py_targets.append(p)

sh_targets = list((root / "src/bash").glob("*.sh"))

results = []

def is_docstring(node, parent):
    if not isinstance(node, ast.Expr):
        return False
    if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
        return False
    if isinstance(parent, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return bool(parent.body) and parent.body[0] is node
    return False

class V(ast.NodeVisitor):
    def __init__(self, path, src):
        self.path = path
        self.src = src
        self.parents = []

    def visit(self, node):
        if self.parents and is_docstring(node, self.parents[-1]):
            return
        self.parents.append(node)
        super().visit(node)
        self.parents.pop()

    def visit_Constant(self, node):
        val = node.value
        lineno = getattr(node, "lineno", None)
        if lineno is None:
            return
        line = self.src[lineno - 1].strip() if lineno - 1 < len(self.src) else ""

        if isinstance(val, str):
            s = val.strip()
            if not s:
                return
            if (
                "/" in s
                or "\\" in s
                or s.endswith((".csv", ".parquet", ".fits", ".png", ".keras", ".json", ".yml", ".yaml", ".sh", ".gpkg"))
                or s.startswith(("--", "AUN_", "SLURM_", "CUDA_", "PYTHON"))
                or "_" in s
                or s.lower() in {"gan", "unet", "sigmoid", "z_scale", "log_min_max", "min_max", "average", "matched", "sequential", "parallel", "array"}
            ):
                results.append(("python", str(self.path.relative_to(root)).replace('\\', '/'), lineno, repr(val), line))
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            if val in {0, 1, -1}:
                parent = self.parents[-2] if len(self.parents) >= 2 else None
                if not isinstance(parent, (ast.Assign, ast.AnnAssign, ast.keyword, ast.Call, ast.Compare, ast.Subscript, ast.BinOp)):
                    return
            results.append(("python", str(self.path.relative_to(root)).replace('\\', '/'), lineno, repr(val), line))

for p in py_targets:
    try:
        text = p.read_text(encoding="utf-8")
        tree = ast.parse(text)
        V(p, text.splitlines()).visit(tree)
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
            results.append(("bash", rel, i, line, line))
            continue
        if line.startswith("#"):
            continue
        m = assign_re.match(raw)
        if m:
            rhs = m.group(2).strip()
            if any(ch.isdigit() for ch in rhs) or any(tok in rhs for tok in ["/", ".", "%", "gpu", "cpu", "AUN_", "SLURM", "true", "false"]):
                results.append(("bash", rel, i, rhs[:120], line))

seen = set()
final = []
for row in results:
    key = (row[0], row[1], row[2], row[3])
    if key in seen:
        continue
    seen.add(key)
    final.append(row)

final.sort(key=lambda r: (r[1], r[2], r[0]))

out = root / "notes" / "hardcoded_values_audit.md"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    f.write("# Hardcoded Values Audit (Scripts)\n\n")
    f.write("This table is generated from static literal scanning across script entrypoints and orchestration scripts.\n\n")
    f.write("| Lang | File | Line | Hardcoded value | Context |\n")
    f.write("| --- | --- | ---: | --- | --- |\n")
    for lang, file, line, val, ctx in final:
        val_s = str(val).replace("|", "\\|").replace("\n", " ")
        ctx_s = str(ctx).replace("|", "\\|").replace("\n", " ")
        f.write(f"| {lang} | {file} | {line} | {val_s} | {ctx_s} |\n")

print(f"Wrote {len(final)} rows to {out}")
