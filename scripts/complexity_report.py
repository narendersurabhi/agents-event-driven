"""Generate a lightweight code complexity report for this repo.

This script is intentionally dependency-light:
- Always reports file sizes, internal dependency fan-in/out, and layering violations.
- Uses Ruff (already in requirements) to report C901 (mccabe) complexity issues.
- Uses Radon (optional) to compute per-function complexity and maintainability index.

Outputs:
  - JSON report (machine readable)
  - Markdown summary (human readable)

Run:
  python -m scripts.complexity_report
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SCAN_DIRS = ("agents", "core", "api", "scripts", "ui", "tests")
DEFAULT_OUT_DIR = REPO_ROOT / "out"

INTERNAL_TOPLEVEL = {"agents", "core", "api", "scripts", "ui", "tests"}


def _iter_py_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    excluded = {
        ".git",
        ".venv",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
        "node_modules",
    }
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            files.append(p)
            continue
        if not p.is_dir():
            continue
        for f in p.rglob("*.py"):
            if any(part in excluded for part in f.parts):
                continue
            files.append(f)
    return sorted(set(files))


def _module_name(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative_import(current_module: str, level: int, module: str | None) -> str | None:
    if level <= 0:
        return module
    parts = current_module.split(".") if current_module else []
    if level > len(parts):
        return None
    base = parts[:-level]
    if module:
        base.extend(module.split("."))
    return ".".join([p for p in base if p])


def _parse_imports(path: Path, module: str) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return set()

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return set()

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_relative_import(module, node.level or 0, node.module)
            if resolved:
                imported.add(resolved)
    return imported


def _internalize_import(name: str, module_map: dict[str, Path]) -> str | None:
    """Map an import name to a known internal module if possible."""
    if not name:
        return None
    top = name.split(".", 1)[0]
    if top not in INTERNAL_TOPLEVEL:
        return None
    # Prefer the most specific module that exists in the map.
    candidate = name
    while candidate:
        if candidate in module_map:
            return candidate
        if "." not in candidate:
            break
        candidate = candidate.rsplit(".", 1)[0]
    return None


@dataclass(frozen=True)
class FileStats:
    path: Path
    module: str
    lines: int


def _count_lines(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except OSError:
        return 0


def _run_ruff_c901(scan_paths: list[str]) -> list[dict[str, Any]]:
    """Run Ruff for C901 only and return the JSON diagnostics list."""
    cmd = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--select",
        "C901",
        "--output-format",
        "json",
        "--exit-zero",
        *scan_paths,
    ]
    try:
        res = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    if not res.stdout.strip():
        return []
    try:
        parsed = json.loads(res.stdout)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        return []
    return []


def _try_radon() -> tuple[bool, Any, Any, Any]:
    try:
        from radon.complexity import cc_visit
        from radon.metrics import mi_rank, mi_visit
    except ImportError:
        return False, None, None, None
    return True, cc_visit, mi_visit, mi_rank


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _top_n(items: list[tuple[Any, float]], n: int) -> list[tuple[Any, float]]:
    return sorted(items, key=lambda x: x[1], reverse=True)[:n]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate repo complexity report (JSON + Markdown)."
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=list(DEFAULT_SCAN_DIRS),
        help="Directories/files to scan (default: agents core api scripts ui tests).",
    )
    parser.add_argument(
        "--out-json",
        default=str(DEFAULT_OUT_DIR / "complexity_report.json"),
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--out-md",
        default=str(DEFAULT_OUT_DIR / "complexity_report.md"),
        help="Where to write the Markdown summary.",
    )
    parser.add_argument(
        "--top", type=int, default=15, help="How many items to show in 'top' lists."
    )
    return parser.parse_args(argv)


def _collect_file_stats(scan_paths: list[str]) -> tuple[list[FileStats], dict[str, Path]]:
    scan_path_objs = [(REPO_ROOT / p).resolve() for p in scan_paths]
    py_files = _iter_py_files(scan_path_objs)

    module_map: dict[str, Path] = {_module_name(p): p for p in py_files}
    file_stats: list[FileStats] = [
        FileStats(path=p, module=_module_name(p), lines=_count_lines(p)) for p in py_files
    ]
    return file_stats, module_map


def _build_dependency_graph(
    file_stats: list[FileStats], module_map: dict[str, Path]
) -> tuple[dict[str, set[str]], list[dict[str, str]]]:
    deps: dict[str, set[str]] = {}
    layering_violations: list[dict[str, str]] = []

    for fs in file_stats:
        imported = _parse_imports(fs.path, fs.module)
        internal_imports: set[str] = set()
        for name in imported:
            internal = _internalize_import(name, module_map)
            if not internal:
                continue
            internal_imports.add(internal)

            if (fs.module.startswith("core.") or fs.module == "core") and internal.startswith(
                "agents."
            ):
                layering_violations.append(
                    {
                        "module": fs.module,
                        "path": str(fs.path.relative_to(REPO_ROOT)),
                        "imports": internal,
                    }
                )

        deps[fs.module] = internal_imports

    return deps, layering_violations


def _compute_fan_in_out(deps: dict[str, set[str]]) -> tuple[dict[str, int], dict[str, int]]:
    fan_out = {m: len(v) for m, v in deps.items()}
    fan_in: dict[str, int] = {m: 0 for m in deps}
    for imports in deps.values():
        for dst in imports:
            fan_in[dst] = fan_in.get(dst, 0) + 1
    return fan_in, fan_out


def _count_c901_by_file(ruff_c901: list[dict[str, Any]]) -> dict[str, int]:
    c901_by_file: dict[str, int] = {}
    for diag in ruff_c901:
        filename = str(diag.get("filename") or "")
        if filename:
            try:
                filename = str(Path(filename).resolve().relative_to(REPO_ROOT))
            except Exception:
                pass
        if filename:
            c901_by_file[filename] = c901_by_file.get(filename, 0) + 1
    return c901_by_file


def _compute_radon_metrics(
    file_stats: list[FileStats], top: int
) -> tuple[bool, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    radon_available, cc_visit, mi_visit, mi_rank = _try_radon()
    radon_cc_top: list[dict[str, Any]] = []
    radon_mi: dict[str, dict[str, Any]] = {}

    if not radon_available:
        return False, radon_cc_top, radon_mi

    for fs in file_stats:
        try:
            text = fs.path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            results = cc_visit(text)
            for r in results:
                radon_cc_top.append(
                    {
                        "path": str(fs.path),
                        "module": fs.module,
                        "name": getattr(r, "name", ""),
                        "type": getattr(r, "type", ""),
                        "complexity": float(getattr(r, "complexity", 0)),
                        "lineno": int(getattr(r, "lineno", 0)),
                    }
                )
            mi = float(mi_visit(text, True))
            radon_mi[str(fs.path)] = {
                "module": fs.module,
                "mi": mi,
                "rank": str(mi_rank(mi)),
            }
        except Exception:
            continue

    radon_cc_top = sorted(radon_cc_top, key=lambda x: x["complexity"], reverse=True)[:top]
    return True, radon_cc_top, radon_mi


def _resolve_out_paths(out_json: str, out_md: str) -> tuple[Path, Path]:
    json_path = Path(out_json)
    if not json_path.is_absolute():
        json_path = REPO_ROOT / json_path
    md_path = Path(out_md)
    if not md_path.is_absolute():
        md_path = REPO_ROOT / md_path
    return json_path, md_path


def _render_markdown(
    *,
    file_stats: list[FileStats],
    out_json: Path,
    top_lines: list[tuple[Any, float]],
    top_fan_in: list[tuple[Any, float]],
    top_fan_out: list[tuple[Any, float]],
    c901_by_file: dict[str, int],
    layering_violations: list[dict[str, str]],
    radon_available: bool,
    radon_cc_top: list[dict[str, Any]],
    top: int,
) -> str:
    md_lines: list[str] = []
    md_lines.append("# Complexity Report")
    md_lines.append("")
    md_lines.append(f"- Files scanned: `{len(file_stats)}`")
    md_lines.append(f"- Report JSON: `{out_json.relative_to(REPO_ROOT)}`")
    md_lines.append("")

    md_lines.append("## Top Largest Files (LOC)")
    for path, loc in top_lines:
        try:
            rel = str(Path(path).relative_to(REPO_ROOT))
        except Exception:
            rel = str(path)
        md_lines.append(f"- `{rel}`: {int(loc)}")
    md_lines.append("")

    md_lines.append("## Top Fan-In (Most Imported Modules)")
    for mod, val in top_fan_in:
        md_lines.append(f"- `{mod}`: {int(val)}")
    md_lines.append("")

    md_lines.append("## Top Fan-Out (Most Importing Modules)")
    for mod, val in top_fan_out:
        md_lines.append(f"- `{mod}`: {int(val)}")
    md_lines.append("")

    md_lines.append("## Ruff C901 (Too Complex) Summary")
    md_lines.append(f"- Files with C901 violations: `{len(c901_by_file)}`")
    md_lines.append(
        "- Tip: adjust threshold via `[tool.ruff.lint.mccabe] max-complexity` in `pyproject.toml`."
    )
    if c901_by_file:
        worst = _top_n([(k, float(v)) for k, v in c901_by_file.items()], top)
        for path, cnt in worst:
            try:
                rel = str(Path(path).relative_to(REPO_ROOT))
            except Exception:
                rel = str(path)
            md_lines.append(f"- `{rel}`: {int(cnt)}")
    md_lines.append("")

    md_lines.append("## Layering Violations")
    md_lines.append("- Rule: `core/*` must not import `agents/*`.")
    if layering_violations:
        for v in layering_violations:
            try:
                rel = str(Path(v["path"]).relative_to(REPO_ROOT))
            except Exception:
                rel = v["path"]
            md_lines.append(f"- `{rel}` imports `{v['imports']}`")
    else:
        md_lines.append("- None detected.")
    md_lines.append("")

    md_lines.append("## Radon (Optional)")
    if radon_available:
        md_lines.append("- Installed: yes")
        md_lines.append("### Top Complex Functions/Methods")
        for r in radon_cc_top:
            md_lines.append(
                f"- `{r['path']}`:{r['lineno']} `{r['name']}` complexity={int(r['complexity'])}"
            )
    else:
        md_lines.append("- Installed: no (add deps via `pip install -r requirements.txt`).")
    md_lines.append("")

    return "\n".join(md_lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    file_stats, module_map = _collect_file_stats(args.paths)
    deps, layering_violations = _build_dependency_graph(file_stats, module_map)
    fan_in, fan_out = _compute_fan_in_out(deps)

    # Ruff complexity (C901).
    ruff_c901 = _run_ruff_c901([p for p in args.paths])
    c901_by_file = _count_c901_by_file(ruff_c901)

    # Optional Radon metrics.
    radon_available, radon_cc_top, radon_mi = _compute_radon_metrics(file_stats, args.top)

    # Assemble report.
    top_lines = _top_n(
        [(str(s.path.relative_to(REPO_ROOT)), float(s.lines)) for s in file_stats],
        args.top,
    )
    top_fan_in = _top_n([(m, float(v)) for m, v in fan_in.items()], args.top)
    top_fan_out = _top_n([(m, float(v)) for m, v in fan_out.items()], args.top)

    report: dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "scanned_paths": args.paths,
        "file_count": len(file_stats),
        "files": [
            {
                "path": str(s.path.relative_to(REPO_ROOT)),
                "module": s.module,
                "lines": s.lines,
            }
            for s in file_stats
        ],
        "fan_in": fan_in,
        "fan_out": fan_out,
        "layering_violations": layering_violations,
        "ruff": {"c901_violations": ruff_c901, "c901_count_by_file": c901_by_file},
        "radon": {
            "available": radon_available,
            "top_complexity": radon_cc_top,
            "maintainability_index": radon_mi,
        },
    }

    out_json, out_md = _resolve_out_paths(args.out_json, args.out_md)

    _write_json(out_json, report)

    markdown = _render_markdown(
        file_stats=file_stats,
        out_json=out_json,
        top_lines=top_lines,
        top_fan_in=top_fan_in,
        top_fan_out=top_fan_out,
        c901_by_file=c901_by_file,
        layering_violations=layering_violations,
        radon_available=radon_available,
        radon_cc_top=radon_cc_top,
        top=args.top,
    )
    _write_text(out_md, markdown)

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")
    if not radon_available:
        print("Note: Radon not installed; report includes Ruff + structural metrics only.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
