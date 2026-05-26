"""Walk a repo and extract code chunks and documentation sections."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".docsentinel",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
}

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


class CodeChunk(BaseModel):
    key: str
    filepath: str
    name: str
    signature: str
    docstring: str | None = None
    lineno: int
    parent: str | None = None


class DocSection(BaseModel):
    key: str
    filepath: str
    heading_path: list[str] = Field(default_factory=list)
    heading_level: int
    body: str
    lineno: int


def _iter_repo_files(
    repo_path: Path,
    suffix: str,
    exclude_dirs: set[str],
) -> list[Path]:
    files: list[Path] = []
    for path in sorted(repo_path.rglob(f"*{suffix}")):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_path)
        if any(part in exclude_dirs for part in rel.parts):
            continue
        files.append(rel)
    return files


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    args = ast.unparse(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix}def {node.name}({args}){returns}"


def _extract_docstring(node: ast.AST) -> str | None:
    doc = ast.get_docstring(node)
    return doc.strip() if doc else None


def _chunk_key(filepath: str, name: str, parent: str | None) -> str:
    if parent:
        return f"{filepath}::{parent}.{name}"
    return f"{filepath}::{name}"


class _FunctionExtractor(ast.NodeVisitor):
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.chunks: list[CodeChunk] = []
        self._class_stack: list[str] = []
        self._function_depth = 0

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._add_chunk(item)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._function_depth == 0 and not self._class_stack:
            self._add_chunk(node)
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self._function_depth == 0 and not self._class_stack:
            self._add_chunk(node)
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def _add_chunk(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent = self._class_stack[-1] if self._class_stack else None
        self.chunks.append(
            CodeChunk(
                key=_chunk_key(self.filepath, node.name, parent),
                filepath=self.filepath,
                name=node.name,
                signature=_function_signature(node),
                docstring=_extract_docstring(node),
                lineno=node.lineno,
                parent=parent,
            )
        )


def _parse_python_file(repo_path: Path, rel_path: Path) -> list[CodeChunk]:
    source = (repo_path / rel_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(rel_path))
    extractor = _FunctionExtractor(str(rel_path))
    extractor.visit(tree)
    return extractor.chunks


def _heading_path_key(filepath: str, heading_path: list[str]) -> str:
    if not heading_path:
        return filepath
    return f"{filepath}::{'::'.join(heading_path)}"


def _parse_markdown_file(repo_path: Path, rel_path: Path) -> list[DocSection]:
    lines = (repo_path / rel_path).read_text(encoding="utf-8").splitlines()
    sections: list[DocSection] = []
    heading_stack: list[tuple[int, str]] = []

    current_heading: tuple[int, list[str], int] | None = None
    current_body: list[str] = []

    def flush_section() -> None:
        nonlocal current_heading, current_body
        if current_heading is None:
            return
        level, path, lineno = current_heading
        sections.append(
            DocSection(
                key=_heading_path_key(str(rel_path), path),
                filepath=str(rel_path),
                heading_path=path,
                heading_level=level,
                body="\n".join(current_body).strip(),
                lineno=lineno,
            )
        )
        current_body = []

    for idx, line in enumerate(lines, start=1):
        match = HEADING_RE.match(line)
        if match:
            flush_section()
            level = len(match.group(1))
            title = match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            path = [name for _, name in heading_stack]
            current_heading = (level, path, idx)
            continue

        if current_heading is not None:
            current_body.append(line)

    flush_section()
    return sections


def parse_repo(
    repo_path: Path | str,
    exclude_dirs: set[str] | None = None,
    *,
    verbose: bool = False,
) -> tuple[list[CodeChunk], list[DocSection]]:
    """Walk *repo_path* and return code chunks and documentation sections."""
    root = Path(repo_path).resolve()
    excluded = exclude_dirs or DEFAULT_EXCLUDE_DIRS
    _ = verbose  # reserved for progress logging

    code_chunks: list[CodeChunk] = []
    for rel_path in _iter_repo_files(root, ".py", excluded):
        try:
            code_chunks.extend(_parse_python_file(root, rel_path))
        except SyntaxError:
            continue

    doc_sections: list[DocSection] = []
    for rel_path in _iter_repo_files(root, ".md", excluded):
        doc_sections.extend(_parse_markdown_file(root, rel_path))

    return code_chunks, doc_sections
