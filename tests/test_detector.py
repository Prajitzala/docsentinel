"""Tests for detector.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from detector import (
    CodeChange,
    StalenessResult,
    collect_verification_pairs,
    find_affected_sections,
    parse_diff,
    verify_staleness,
    verify_staleness_batch,
)
from embedder import LinkGraph, load_link_graph
from parser import DocSection

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_diff() -> str:
    return (FIXTURES / "sample.diff").read_text(encoding="utf-8")


@pytest.fixture
def test_only_diff() -> str:
    return (FIXTURES / "test_only.diff").read_text(encoding="utf-8")


@pytest.fixture
def docstring_only_diff() -> str:
    return (FIXTURES / "docstring_only.diff").read_text(encoding="utf-8")


@pytest.fixture
def sample_link_graph() -> LinkGraph:
    return load_link_graph(FIXTURES / "links.json")


@pytest.fixture
def usage_doc_section() -> DocSection:
    return DocSection(
        key="docs/guide.md::Guide::Usage",
        filepath="docs/guide.md",
        heading_path=["Guide", "Usage"],
        heading_level=2,
        body="Call `greet(name: str)` to say hi. It returns a greeting string.",
        lineno=5,
    )


def test_parse_diff_extracts_meaningful_signature_change(sample_diff: str) -> None:
    changes = parse_diff(sample_diff)

    assert len(changes) == 1
    change = changes[0]
    assert change.filepath == "src/service.py"
    assert change.function_name == "greet"
    assert change.change_type == "modified"
    assert change.is_meaningful is True
    assert change.old_content is not None
    assert change.new_content is not None
    assert "loud: bool" in change.new_content
    assert "loud" not in (change.old_content or "")


def test_parse_diff_skips_test_files(test_only_diff: str) -> None:
    changes = parse_diff(test_only_diff)
    assert changes == []


def test_parse_diff_marks_docstring_only_as_not_meaningful(
    docstring_only_diff: str,
) -> None:
    changes = parse_diff(docstring_only_diff)

    assert len(changes) == 1
    assert changes[0].function_name == "greet"
    assert changes[0].is_meaningful is False


def test_find_affected_sections(sample_link_graph: LinkGraph, usage_doc_section: DocSection) -> None:
    change = CodeChange(
        filepath="src/service.py",
        function_name="greet",
        change_type="modified",
        old_content="def greet(name: str) -> str:\n    return f'Hello, {name}'",
        new_content="def greet(name: str, loud: bool = False) -> str:\n    ...",
        is_meaningful=True,
    )

    with patch("detector.load_doc_sections") as mock_load:
        mock_load.return_value = {usage_doc_section.key: usage_doc_section}
        affected = find_affected_sections(
            [change],
            sample_link_graph,
            Path(".docsentinel/chroma"),
        )

    mock_load.assert_called_once()
    assert change in affected
    assert affected[change] == [usage_doc_section]


def test_find_affected_sections_ignores_non_meaningful(sample_link_graph: LinkGraph) -> None:
    change = CodeChange(
        filepath="src/service.py",
        function_name="greet",
        change_type="modified",
        old_content="def greet(name: str) -> str:\n    ...",
        new_content="def greet(name: str) -> str:\n    ...",
        is_meaningful=False,
    )

    with patch("detector.load_doc_sections") as mock_load:
        affected = find_affected_sections(
            [change],
            sample_link_graph,
            Path(".docsentinel/chroma"),
        )

    mock_load.assert_not_called()
    assert affected == {}


def test_collect_verification_pairs(usage_doc_section: DocSection) -> None:
    change = CodeChange(
        filepath="src/service.py",
        function_name="greet",
        change_type="modified",
        old_content="old",
        new_content="new",
        is_meaningful=True,
    )
    pairs = collect_verification_pairs({change: [usage_doc_section]})

    assert len(pairs) == 1
    assert pairs[0][0] == change
    assert pairs[0][1] == usage_doc_section
    assert pairs[0][2] == "src/service.py::greet"


@pytest.mark.asyncio
async def test_verify_staleness_returns_expected_shape(usage_doc_section: DocSection) -> None:
    change = CodeChange(
        filepath="src/service.py",
        function_name="greet",
        change_type="modified",
        old_content="def greet(name: str) -> str:\n    return f'Hello, {name}'",
        new_content="def greet(name: str, loud: bool = False) -> str:\n    ...",
        is_meaningful=True,
    )
    expected = StalenessResult(
        doc_section_id=usage_doc_section.key,
        code_chunk_id="src/service.py::greet",
        is_stale=True,
        reason="The function signature now includes a loud parameter.",
        confidence=0.92,
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=expected)

    result = await verify_staleness(
        change,
        usage_doc_section,
        "src/service.py::greet",
        client=mock_client,
    )

    assert isinstance(result, StalenessResult)
    assert result.doc_section_id == usage_doc_section.key
    assert result.code_chunk_id == "src/service.py::greet"
    assert result.is_stale is True
    assert result.reason is not None
    assert 0.0 <= result.confidence <= 1.0
    mock_client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_verify_staleness_batch_runs_concurrently(usage_doc_section: DocSection) -> None:
    change = CodeChange(
        filepath="src/service.py",
        function_name="greet",
        change_type="modified",
        old_content="old",
        new_content="new",
        is_meaningful=True,
    )
    expected = StalenessResult(
        doc_section_id=usage_doc_section.key,
        code_chunk_id="src/service.py::greet",
        is_stale=False,
        reason="Still accurate.",
        confidence=0.8,
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=expected)

    results = await verify_staleness_batch(
        [(change, usage_doc_section, "src/service.py::greet")],
        client=mock_client,
    )

    assert len(results) == 1
    assert results[0].is_stale is False
