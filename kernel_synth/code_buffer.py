"""In-memory buffer of Python source files for a cloned repository."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


# Directories we should always skip when walking a repo.
_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "build",
    "dist",
    ".venv",
    "venv",
    "env",
    ".tox",
    ".eggs",
    "docs",
    "doc",
    "examples",  # Tend to be tutorials; modules live elsewhere.
    "tests",
    "test",
    "benchmarks",
    "benchmark",
    "third_party",
}

_MAX_FILE_BYTES = 256 * 1024  # 256 KiB cap — agent will not benefit from huge files.


@dataclass
class FileEntry:
    """One Python source file in the buffer."""

    path: str            # relative to repo root
    abs_path: Path
    size: int
    n_lines: int
    n_classes: int
    n_nn_modules: int
    n_functions: int = 0   # module-level def / async def
    classes: list["ClassEntry"] = field(default_factory=list)

    def read(self) -> str:
        return self.abs_path.read_text(encoding="utf-8", errors="replace")


@dataclass
class ClassEntry:
    name: str
    start_line: int
    end_line: int
    bases: list[str]
    is_nn_module: bool


@dataclass
class CodeBuffer:
    """All Python files of a repo, with a lightweight AST overview per file."""

    repo_root: Path
    files: list[FileEntry] = field(default_factory=list)

    @property
    def n_files(self) -> int:
        return len(self.files)

    @property
    def n_loc(self) -> int:
        return sum(f.n_lines for f in self.files)

    @property
    def n_nn_modules(self) -> int:
        return sum(f.n_nn_modules for f in self.files)

    @property
    def n_total_classes(self) -> int:
        """Total ``class`` definitions across the buffer (nn.Module or not)."""
        return sum(f.n_classes for f in self.files)

    @property
    def n_functions(self) -> int:
        """Total top-level ``def`` / ``async def`` across the buffer."""
        return sum(f.n_functions for f in self.files)

    def files_with_nn_modules(self) -> list[FileEntry]:
        return [f for f in self.files if f.n_nn_modules > 0]

    def by_path(self, path: str) -> FileEntry | None:
        for f in self.files:
            if f.path == path:
                return f
        return None

    def overview(self, max_files: int = 80) -> str:
        """Compact textual summary the agent gets as a starting view."""
        modular = sorted(
            self.files_with_nn_modules(),
            key=lambda f: f.n_nn_modules,
            reverse=True,
        )
        head = modular[:max_files]
        lines = [
            f"Repo root: {self.repo_root.name}",
            f"Total .py files: {self.n_files}  |  Total LOC: {self.n_loc}",
            f"Files containing nn.Module subclasses: {len(modular)}",
            "",
            "Top files by nn.Module count:",
        ]
        for f in head:
            cls_names = [c.name for c in f.classes if c.is_nn_module]
            joined = ", ".join(cls_names[:8])
            if len(cls_names) > 8:
                joined += f", … (+{len(cls_names) - 8} more)"
            lines.append(f"  {f.path}  ({f.n_nn_modules}) -> {joined}")
        return "\n".join(lines)


def build_buffer(repo_root: Path) -> CodeBuffer:
    """Walk ``repo_root`` and build a :class:`CodeBuffer`."""
    repo_root = repo_root.resolve()
    buffer = CodeBuffer(repo_root=repo_root)

    for py_path in _iter_python_files(repo_root):
        try:
            size = py_path.stat().st_size
        except OSError:
            continue
        if size > _MAX_FILE_BYTES:
            continue
        try:
            text = py_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        classes = _extract_classes(text)
        n_nn = sum(1 for c in classes if c.is_nn_module)
        n_funcs = _count_top_level_functions(text)
        entry = FileEntry(
            path=str(py_path.relative_to(repo_root)),
            abs_path=py_path,
            size=size,
            n_lines=text.count("\n") + 1,
            n_classes=len(classes),
            n_nn_modules=n_nn,
            n_functions=n_funcs,
            classes=classes,
        )
        buffer.files.append(entry)

    buffer.files.sort(key=lambda f: (-f.n_nn_modules, f.path))
    return buffer


def _iter_python_files(root: Path):
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = list(current.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_symlink():
                continue
            if child.is_dir():
                if child.name in _SKIP_DIRS or child.name.startswith("."):
                    continue
                stack.append(child)
            elif child.is_file() and child.suffix == ".py":
                yield child


def _count_top_level_functions(source: str) -> int:
    """Count module-level ``def`` / ``async def`` statements."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    return sum(
        1
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def _extract_classes(source: str) -> list[ClassEntry]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    out: list[ClassEntry] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [_base_name(b) for b in node.bases]
        is_nn = any(
            b.endswith("nn.Module") or b == "Module" or b == "torch.nn.Module"
            for b in bases
        )
        end = getattr(node, "end_lineno", node.lineno)
        out.append(
            ClassEntry(
                name=node.name,
                start_line=node.lineno,
                end_line=end,
                bases=bases,
                is_nn_module=is_nn,
            )
        )
    return out


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_base_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _base_name(node.func)
    return ""


def slice_source(file: FileEntry, start_line: int, end_line: int) -> str:
    """Return the (1-indexed inclusive) line range of ``file``."""
    text = file.read().splitlines()
    start = max(start_line - 1, 0)
    end = min(end_line, len(text))
    return "\n".join(text[start:end])
