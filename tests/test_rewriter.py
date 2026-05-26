"""Tests for rewriter.py."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from detector import CodeChange, StalenessResult
from parser import DocSection
from rewriter import (
    VALIDATION_FAILED_PREFIX,
    CorrectionResult,
    _ValidationVerdict,
    generate_correction,
    repair_all,
    route_correction,
    validate_correction,
)

DEFAULT_THRESHOLD = 0.85


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


@pytest.fixture
def greet_change() -> CodeChange:
    return CodeChange(
        filepath="src/service.py",
        function_name="greet",
        change_type="modified",
        old_content="def greet(name: str) -> str:\n    return f'Hello, {name}'",
        new_content="def greet(name: str, loud: bool = False) -> str:\n    ...",
        is_meaningful=True,
    )


@pytest.fixture
def stale_result(usage_doc_section: DocSection) -> StalenessResult:
    return StalenessResult(
        doc_section_id=usage_doc_section.key,
        code_chunk_id="src/service.py::greet",
        is_stale=True,
        reason="The function signature now includes a loud parameter.",
        confidence=0.92,
    )


def _correction(
    section: DocSection,
    *,
    confidence: float,
    changes_summary: str = "Updated signature to include loud parameter.",
) -> CorrectionResult:
    return CorrectionResult(
        doc_section_id=section.key,
        original_content=section.body,
        corrected_content=(
            "Call `greet(name: str, loud: bool = False)` to say hi. "
            "It returns a greeting string."
        ),
        confidence=confidence,
        correction_mode="auto_fix",
        changes_summary=changes_summary,
    )


def test_route_correction_auto_fix_above_threshold(
    usage_doc_section: DocSection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTO_FIX_CONFIDENCE", str(DEFAULT_THRESHOLD))
    result = _correction(usage_doc_section, confidence=0.9)

    routed = route_correction(result)

    assert routed.correction_mode == "auto_fix"
    assert routed.confidence == 0.9


def test_route_correction_human_review_below_threshold(
    usage_doc_section: DocSection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTO_FIX_CONFIDENCE", str(DEFAULT_THRESHOLD))
    result = _correction(usage_doc_section, confidence=0.7)

    routed = route_correction(result)

    assert routed.correction_mode == "human_review"
    assert routed.confidence == 0.7


@pytest.mark.asyncio
async def test_validate_correction_failure_lowers_confidence(
    usage_doc_section: DocSection,
    greet_change: CodeChange,
) -> None:
    corrected = _correction(usage_doc_section, confidence=0.9)
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_ValidationVerdict(
            is_valid=False,
            warning="Removed accurate usage details.",
        )
    )

    result = await validate_correction(
        usage_doc_section,
        corrected,
        greet_change,
        client=mock_client,
    )

    assert result.confidence == pytest.approx(0.7)
    assert result.changes_summary.startswith(VALIDATION_FAILED_PREFIX)
    assert "Removed accurate usage details." in result.changes_summary
    assert result.correction_mode == "human_review"
    mock_client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_validate_correction_success_preserves_confidence(
    usage_doc_section: DocSection,
    greet_change: CodeChange,
) -> None:
    corrected = _correction(usage_doc_section, confidence=0.9)
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_ValidationVerdict(is_valid=True, warning=None)
    )

    result = await validate_correction(
        usage_doc_section,
        corrected,
        greet_change,
        client=mock_client,
    )

    assert result.confidence == 0.9
    assert result.correction_mode == "auto_fix"
    assert VALIDATION_FAILED_PREFIX not in result.changes_summary


@pytest.mark.asyncio
async def test_generate_correction_returns_expected_shape(
    usage_doc_section: DocSection,
    greet_change: CodeChange,
    stale_result: StalenessResult,
) -> None:
    expected = _correction(usage_doc_section, confidence=0.88)
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=expected)

    result = await generate_correction(
        usage_doc_section,
        greet_change,
        stale_result,
        client=mock_client,
    )

    assert isinstance(result, CorrectionResult)
    assert result.doc_section_id == usage_doc_section.key
    assert result.original_content == usage_doc_section.body
    assert "loud" in result.corrected_content
    mock_client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_repair_all_runs_concurrently(
    usage_doc_section: DocSection,
    greet_change: CodeChange,
    stale_result: StalenessResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTO_FIX_CONFIDENCE", str(DEFAULT_THRESHOLD))

    second_section = DocSection(
        key="docs/guide.md::Guide::Advanced",
        filepath="docs/guide.md",
        heading_path=["Guide", "Advanced"],
        heading_level=2,
        body="Use `greet` with loud=True for uppercase output.",
        lineno=10,
    )
    second_stale = StalenessResult(
        doc_section_id=second_section.key,
        code_chunk_id="src/service.py::greet",
        is_stale=True,
        reason="loud parameter behavior undocumented.",
        confidence=0.9,
    )

    generated = _correction(usage_doc_section, confidence=0.9)
    generated_second = _correction(second_section, confidence=0.91)
    valid_verdict = _ValidationVerdict(is_valid=True, warning=None)
    generate_responses = [generated, generated_second]

    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()
    generate_index = 0

    async def tracked_create(*args: object, **kwargs: object) -> object:
        nonlocal in_flight, max_in_flight, generate_index
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)

        await asyncio.sleep(0.05)

        async with lock:
            in_flight -= 1

        response_model = kwargs.get("response_model")
        if response_model is _ValidationVerdict:
            return valid_verdict
        if response_model is CorrectionResult:
            result = generate_responses[generate_index]
            generate_index += 1
            return result
        raise AssertionError(f"Unexpected response_model: {response_model}")

    mock_client = MagicMock()
    mock_client.chat.completions.create = tracked_create

    results = await repair_all(
        [stale_result, second_stale],
        [usage_doc_section, second_section],
        [greet_change],
        client=mock_client,
    )

    assert len(results) == 2
    assert all(result.correction_mode == "auto_fix" for result in results)
    assert max_in_flight > 1
    assert generate_index == 2


@pytest.mark.asyncio
async def test_repair_all_skips_non_stale(
    usage_doc_section: DocSection,
    greet_change: CodeChange,
    stale_result: StalenessResult,
) -> None:
    fresh = StalenessResult(
        doc_section_id=usage_doc_section.key,
        code_chunk_id="src/service.py::greet",
        is_stale=False,
        reason="Still accurate.",
        confidence=0.95,
    )
    generated = _correction(usage_doc_section, confidence=0.9)
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[
            generated,
            _ValidationVerdict(is_valid=True, warning=None),
        ]
    )

    results = await repair_all(
        [fresh, stale_result],
        [usage_doc_section],
        [greet_change],
        client=mock_client,
    )

    assert len(results) == 1
    assert mock_client.chat.completions.create.await_count == 2
