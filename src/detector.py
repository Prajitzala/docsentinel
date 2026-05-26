"""Parse git diffs, filter meaningful changes, and verify documentation staleness."""

from __future__ import annotations

import ast
import asyncio
import re
from pathlib import Path
from typing import Literal

import instructor
import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field
from unidiff import PatchSet

from embedder import LinkGraph, load_doc_sections
from parser import DocSection

log = structlog.get_logger()

PROMPTS_DIR = Path(__file__).parent / "prompts"
STALENESS_PROMPT = (PROMPTS_DIR / "staleness_check.txt").read_text(encoding="utf-8")
STALENESS_MODEL = "gpt-4o"
MAX_LLM_RETRIES = 2

TEST_FILE_RE = re.compile(
    r"(^|/)test_[^/]+\.py$|"
    r"(^|/)[^/]+_test\.py$|"
    r"\.test\.[^/]+$|"
    r"\.spec\.[^/]+$",
    re.IGNORECASE,
)


class CodeChange(BaseModel):
    """A meaningful code change extracted from a git diff."""

    model_config = ConfigDict(frozen=True)

    filepath: str
    function_name: str | None
    change_type: Literal["added", "removed", "modified"]
    old_content: str | None
    new_content: str | None
    is_meaningful: bool


class StalenessResult(BaseModel):
    """LLM verdict on whether a doc section is stale after a code change."""

    doc_section_id: str
    code_chunk_id: str
    is_stale: bool
    reason: str | None
    confidence: float = Field(ge=0.0, le=1.0)


def _is_test_file(filepath: str) -> bool:
    return bool(TEST_FILE_RE.search(filepath.replace("\\", "/")))


def _code_chunk_id(filepath: str, function_name: str | None) -> str | None:
    if not function_name:
        return None
    return f"{filepath}::{function_name}"


def _function_source(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return ast.unparse(node)


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    args = ast.unparse(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix}def {node.name}({args}){returns}"


def _body_without_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    module = ast.Module(body=body, type_ignores=[])
    return ast.unparse(module)


class _FunctionIndex(ast.NodeVisitor):
    """Collect top-level and class methods with line ranges."""

    def __init__(self) -> None:
        self.functions: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._register(item)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not self._class_stack:
            self._register(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if not self._class_stack:
            self._register(node)
        self.generic_visit(node)

    def _register(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent = self._class_stack[-1] if self._class_stack else None
        name = f"{parent}.{node.name}" if parent else node.name
        self.functions.append((name, node))


def _index_functions(source: str, filepath: str) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return {}

    indexer = _FunctionIndex()
    indexer.visit(tree)
    return dict(indexer.functions)


def _functions_covering_lines(
    source: str,
    filepath: str,
    line_numbers: set[int],
) -> set[str]:
    if not line_numbers:
        return set()

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return set()

    matched: set[str] = set()
    indexer = _FunctionIndex()
    indexer.visit(tree)

    for name, node in indexer.functions:
        end_line = getattr(node, "end_lineno", node.lineno)
        if any(node.lineno <= line_no <= end_line for line_no in line_numbers):
            matched.add(name)

    return matched


def _is_import_only_init_change(
    filepath: str,
    old_source: str | None,
    new_source: str | None,
) -> bool:
    if Path(filepath).name != "__init__.py":
        return False

    for source in (old_source, new_source):
        if source is None:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False
        for node in tree.body:
            if not isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)):
                return False
            if isinstance(node, ast.Expr) and not (
                isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                return False
    return True


def _is_whitespace_only(old_content: str | None, new_content: str | None) -> bool:
    old_norm = re.sub(r"\s+", "", old_content or "")
    new_norm = re.sub(r"\s+", "", new_content or "")
    return old_norm == new_norm


def _is_comment_only_change(old_content: str | None, new_content: str | None) -> bool:
    if old_content is None or new_content is None:
        return False

    def strip_comments(source: str) -> str:
        lines = []
        for line in source.splitlines():
            stripped = line.split("#", maxsplit=1)[0]
            lines.append(stripped)
        return re.sub(r"\s+", "", "".join(lines))

    return strip_comments(old_content) == strip_comments(new_content)


def _is_docstring_only_change(
    old_node: ast.FunctionDef | ast.AsyncFunctionDef | None,
    new_node: ast.FunctionDef | ast.AsyncFunctionDef | None,
) -> bool:
    if old_node is None or new_node is None:
        return False
    if _function_signature(old_node) != _function_signature(new_node):
        return False
    return _body_without_docstring(old_node) == _body_without_docstring(new_node)


def _assess_meaningful_change(
    filepath: str,
    function_name: str | None,
    change_type: Literal["added", "removed", "modified"],
    old_content: str | None,
    new_content: str | None,
    old_source: str | None,
    new_source: str | None,
    old_node: ast.FunctionDef | ast.AsyncFunctionDef | None,
    new_node: ast.FunctionDef | ast.AsyncFunctionDef | None,
) -> bool:
    if _is_test_file(filepath):
        return False
    if _is_import_only_init_change(filepath, old_source, new_source):
        return False
    if _is_whitespace_only(old_content, new_content):
        return False
    if _is_comment_only_change(old_content, new_content):
        return False
    if change_type == "modified" and _is_docstring_only_change(old_node, new_node):
        return False
    if change_type == "added" and new_content is None:
        return False
    if change_type == "removed" and old_content is None:
        return False
    return True


def _reconstruct_file_source(patched_file: object) -> tuple[str | None, str | None]:
    """Build pre- and post-change file text from hunk lines."""
    old_lines: list[str] = []
    new_lines: list[str] = []

    for hunk in patched_file:  # type: ignore[attr-defined]
        for line in hunk:
            value = line.value.rstrip("\n")
            if line.is_context:
                old_lines.append(value)
                new_lines.append(value)
            elif line.is_removed:
                old_lines.append(value)
            elif line.is_added:
                new_lines.append(value)

    old_source = "\n".join(old_lines) if old_lines else None
    new_source = "\n".join(new_lines) if new_lines else None
    return old_source, new_source


def _hunk_changed_lines(patched_file: object) -> tuple[set[int], set[int]]:
    old_lines: set[int] = set()
    new_lines: set[int] = set()

    for hunk in patched_file:  # type: ignore[attr-defined]
        for line in hunk:
            if line.is_removed and line.source_line_no is not None:
                old_lines.add(line.source_line_no)
            if line.is_added and line.target_line_no is not None:
                new_lines.add(line.target_line_no)

    return old_lines, new_lines


def _build_change(
    filepath: str,
    function_name: str,
    change_type: Literal["added", "removed", "modified"],
    old_node: ast.FunctionDef | ast.AsyncFunctionDef | None,
    new_node: ast.FunctionDef | ast.AsyncFunctionDef | None,
    old_source: str | None,
    new_source: str | None,
) -> CodeChange:
    old_content = _function_source(old_node) if old_node else None
    new_content = _function_source(new_node) if new_node else None
    is_meaningful = _assess_meaningful_change(
        filepath,
        function_name,
        change_type,
        old_content,
        new_content,
        old_source,
        new_source,
        old_node,
        new_node,
    )
    return CodeChange(
        filepath=filepath,
        function_name=function_name,
        change_type=change_type,
        old_content=old_content,
        new_content=new_content,
        is_meaningful=is_meaningful,
    )


def parse_diff(diff_text: str) -> list[CodeChange]:
    """Parse raw git diff output into structured code changes."""
    if not diff_text.strip():
        return []

    changes: list[CodeChange] = []
    patch_set = PatchSet(diff_text.splitlines(keepends=True))

    for patched_file in patch_set:
        filepath = patched_file.path.replace("\\", "/")
        if _is_test_file(filepath):
            continue

        old_source, new_source = _reconstruct_file_source(patched_file)
        old_lines, new_lines = _hunk_changed_lines(patched_file)

        old_functions = _index_functions(old_source or "", filepath) if old_source else {}
        new_functions = _index_functions(new_source or "", filepath) if new_source else {}

        touched_old = _functions_covering_lines(old_source or "", filepath, old_lines)
        touched_new = _functions_covering_lines(new_source or "", filepath, new_lines)
        touched_names = touched_old | touched_new

        for name in set(old_functions) & set(new_functions):
            old_node = old_functions[name]
            new_node = new_functions[name]
            if _function_source(old_node) != _function_source(new_node):
                touched_names.add(name)

        if not touched_names and patched_file.is_added_file:  # type: ignore[attr-defined]
            touched_names = set(new_functions)
        elif not touched_names and patched_file.is_removed_file:  # type: ignore[attr-defined]
            touched_names = set(old_functions)

        if patched_file.is_added_file:  # type: ignore[attr-defined]
            file_change_type: Literal["added", "removed", "modified"] = "added"
        elif patched_file.is_removed_file:
            file_change_type = "removed"
        else:
            file_change_type = "modified"

        for name in sorted(touched_names):
            old_node = old_functions.get(name)
            new_node = new_functions.get(name)

            if old_node and new_node:
                change_type = "modified"
            elif new_node:
                change_type = "added"
            elif old_node:
                change_type = "removed"
            else:
                change_type = file_change_type

            if old_node is None and new_node is None:
                continue

            changes.append(
                _build_change(
                    filepath,
                    name,
                    change_type,
                    old_node,
                    new_node,
                    old_source,
                    new_source,
                )
            )

    return changes


def find_affected_sections(
    changes: list[CodeChange],
    link_graph: LinkGraph,
    persist_dir: Path,
) -> dict[CodeChange, list[DocSection]]:
    """Find documentation sections linked to each meaningful code change."""
    chunk_to_docs: dict[str, list[str]] = {}
    for doc_id, chunk_ids in link_graph.links.items():
        for chunk_id in chunk_ids:
            chunk_to_docs.setdefault(chunk_id, []).append(doc_id)

    meaningful = [change for change in changes if change.is_meaningful]
    all_doc_ids: set[str] = set()
    change_doc_ids: dict[int, list[str]] = {}

    for idx, change in enumerate(meaningful):
        chunk_id = _code_chunk_id(change.filepath, change.function_name)
        doc_ids: list[str] = []
        if chunk_id:
            doc_ids = chunk_to_docs.get(chunk_id, [])
        change_doc_ids[idx] = doc_ids
        all_doc_ids.update(doc_ids)

    loaded: dict[str, DocSection] = {}
    if all_doc_ids:
        loaded = load_doc_sections(sorted(all_doc_ids), persist_dir)

    result: dict[CodeChange, list[DocSection]] = {}
    for idx, change in enumerate(meaningful):
        sections = [loaded[doc_id] for doc_id in change_doc_ids[idx] if doc_id in loaded]
        if sections:
            result[change] = sections

    return result


def _format_user_prompt(
    change: CodeChange,
    section: DocSection,
    code_chunk_id: str,
) -> str:
    heading = " > ".join(section.heading_path) if section.heading_path else section.filepath
    old_code = change.old_content or "N/A"
    new_code = change.new_content or "N/A"
    return (
        f"doc_section_id: {section.key}\n"
        f"code_chunk_id: {code_chunk_id}\n\n"
        f"## Old code\n```python\n{old_code}\n```\n\n"
        f"## New code\n```python\n{new_code}\n```\n\n"
        f"## Documentation section ({heading})\n{section.body}"
    )


async def verify_staleness(
    change: CodeChange,
    section: DocSection,
    code_chunk_id: str,
    *,
    client: object | None = None,
) -> StalenessResult:
    """Ask the LLM whether *section* is stale given *change*."""
    llm_client = client or instructor.from_openai(AsyncOpenAI())
    user_prompt = _format_user_prompt(change, section, code_chunk_id)

    result = await llm_client.chat.completions.create(
        model=STALENESS_MODEL,
        response_model=StalenessResult,
        messages=[
            {"role": "system", "content": STALENESS_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_retries=MAX_LLM_RETRIES,
    )

    return result.model_copy(
        update={
            "doc_section_id": section.key,
            "code_chunk_id": code_chunk_id,
        }
    )


async def verify_staleness_batch(
    pairs: list[tuple[CodeChange, DocSection, str]],
    *,
    client: object | None = None,
) -> list[StalenessResult]:
    """Run staleness verification for all change/section pairs concurrently."""
    if not pairs:
        return []

    llm_client = client or instructor.from_openai(AsyncOpenAI())
    tasks = [
        verify_staleness(change, section, code_chunk_id, client=llm_client)
        for change, section, code_chunk_id in pairs
    ]
    return list(await asyncio.gather(*tasks))


def run_staleness_checks(
    pairs: list[tuple[CodeChange, DocSection, str]],
    *,
    client: object | None = None,
) -> list[StalenessResult]:
    """Sync entry point that runs concurrent LLM staleness checks."""
    return asyncio.run(verify_staleness_batch(pairs, client=client))


def collect_verification_pairs(
    affected: dict[CodeChange, list[DocSection]],
) -> list[tuple[CodeChange, DocSection, str]]:
    """Flatten affected-section mapping into verification tuples."""
    pairs: list[tuple[CodeChange, DocSection, str]] = []
    for change, sections in affected.items():
        chunk_id = _code_chunk_id(change.filepath, change.function_name)
        if not chunk_id:
            continue
        for section in sections:
            pairs.append((change, section, chunk_id))
    return pairs


def fetch_git_diff(repo_path: Path, base_branch: str) -> str:
    """Return unified diff from *base_branch* to HEAD."""
    import subprocess

    result = subprocess.run(
        ["git", "diff", f"{base_branch}...HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        msg = result.stderr.strip() or "git diff failed"
        raise RuntimeError(msg)
    return result.stdout
