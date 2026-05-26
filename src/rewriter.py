"""LLM-powered documentation repair with validation and confidence routing."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Literal

import instructor
import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from detector import CodeChange, StalenessResult
from parser import DocSection

log = structlog.get_logger()

PROMPTS_DIR = Path(__file__).parent / "prompts"
CORRECTION_GENERATE_PROMPT = (PROMPTS_DIR / "correction_generate.txt").read_text(
    encoding="utf-8"
)
CORRECTION_VALIDATE_PROMPT = (PROMPTS_DIR / "correction_validate.txt").read_text(
    encoding="utf-8"
)
CORRECTION_MODEL = "gpt-4o"
MAX_LLM_RETRIES = 2
DEFAULT_AUTO_FIX_CONFIDENCE = 0.85
CONFIDENCE_VALIDATION_PENALTY = 0.2
VALIDATION_FAILED_PREFIX = "[Validation failed]"


class CorrectionResult(BaseModel):
    """Result of generating and validating a documentation correction."""

    model_config = ConfigDict(frozen=True)

    doc_section_id: str
    original_content: str
    corrected_content: str
    confidence: float = Field(ge=0.0, le=1.0)
    correction_mode: Literal["auto_fix", "human_review"]
    changes_summary: str


class _ValidationVerdict(BaseModel):
    """LLM response for the correction validation pass."""

    is_valid: bool
    warning: str | None = None


def _auto_fix_threshold() -> float:
    raw = os.environ.get("AUTO_FIX_CONFIDENCE", str(DEFAULT_AUTO_FIX_CONFIDENCE))
    try:
        return float(raw)
    except ValueError:
        log.warning(
            "invalid_auto_fix_confidence",
            value=raw,
            fallback=DEFAULT_AUTO_FIX_CONFIDENCE,
        )
        return DEFAULT_AUTO_FIX_CONFIDENCE


def _format_generate_prompt(
    section: DocSection,
    change: CodeChange,
    staleness: StalenessResult,
) -> str:
    heading = " > ".join(section.heading_path) if section.heading_path else section.filepath
    old_code = change.old_content or "N/A"
    new_code = change.new_content or "N/A"
    return (
        f"doc_section_id: {section.key}\n\n"
        f"## Staleness reason\n{staleness.reason or 'Not provided'}\n\n"
        f"## Staleness confidence\n{staleness.confidence:.2f}\n\n"
        f"## Old code\n```python\n{old_code}\n```\n\n"
        f"## New code\n```python\n{new_code}\n```\n\n"
        f"## Documentation section ({heading})\n{section.body}"
    )


def _format_validate_prompt(
    original: DocSection,
    corrected: CorrectionResult,
    change: CodeChange,
) -> str:
    heading = (
        " > ".join(original.heading_path)
        if original.heading_path
        else original.filepath
    )
    old_code = change.old_content or "N/A"
    new_code = change.new_content or "N/A"
    return (
        f"## Original documentation section ({heading})\n{original.body}\n\n"
        f"## Proposed corrected section\n{corrected.corrected_content}\n\n"
        f"## Claimed changes summary\n{corrected.changes_summary}\n\n"
        f"## Old code\n```python\n{old_code}\n```\n\n"
        f"## New code\n```python\n{new_code}\n```"
    )


async def generate_correction(
    section: DocSection,
    change: CodeChange,
    staleness: StalenessResult,
    *,
    client: object | None = None,
) -> CorrectionResult:
    """Generate a corrected documentation section for a stale section."""
    llm_client = client or instructor.from_openai(AsyncOpenAI())
    user_prompt = _format_generate_prompt(section, change, staleness)

    result = await llm_client.chat.completions.create(
        model=CORRECTION_MODEL,
        response_model=CorrectionResult,
        messages=[
            {"role": "system", "content": CORRECTION_GENERATE_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_retries=MAX_LLM_RETRIES,
    )

    return result.model_copy(
        update={
            "doc_section_id": section.key,
            "original_content": section.body,
        }
    )


async def validate_correction(
    original: DocSection,
    corrected: CorrectionResult,
    change: CodeChange,
    *,
    client: object | None = None,
) -> CorrectionResult:
    """Validate a proposed correction and adjust confidence on failure."""
    llm_client = client or instructor.from_openai(AsyncOpenAI())
    user_prompt = _format_validate_prompt(original, corrected, change)

    verdict = await llm_client.chat.completions.create(
        model=CORRECTION_MODEL,
        response_model=_ValidationVerdict,
        messages=[
            {"role": "system", "content": CORRECTION_VALIDATE_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_retries=MAX_LLM_RETRIES,
    )

    if verdict.is_valid:
        return corrected

    warning_text = verdict.warning or "Correction failed validation."
    summary = corrected.changes_summary
    if not summary.startswith(VALIDATION_FAILED_PREFIX):
        summary = f"{VALIDATION_FAILED_PREFIX} {warning_text} {summary}"

    return corrected.model_copy(
        update={
            "confidence": max(0.0, corrected.confidence - CONFIDENCE_VALIDATION_PENALTY),
            "changes_summary": summary,
            "correction_mode": "human_review",
        }
    )


def route_correction(result: CorrectionResult) -> CorrectionResult:
    """Route a correction to auto-fix or human review based on confidence."""
    if result.changes_summary.startswith(VALIDATION_FAILED_PREFIX):
        return result.model_copy(update={"correction_mode": "human_review"})

    threshold = _auto_fix_threshold()
    mode: Literal["auto_fix", "human_review"] = (
        "auto_fix" if result.confidence >= threshold else "human_review"
    )
    return result.model_copy(update={"correction_mode": mode})


def _lookup_sections(sections: list[DocSection]) -> dict[str, DocSection]:
    return {section.key: section for section in sections}


def _lookup_changes(changes: list[CodeChange]) -> dict[str, CodeChange]:
    lookup: dict[str, CodeChange] = {}
    for change in changes:
        if change.function_name is None:
            continue
        chunk_id = f"{change.filepath}::{change.function_name}"
        lookup[chunk_id] = change
    return lookup


async def _repair_one(
    staleness: StalenessResult,
    section: DocSection,
    change: CodeChange,
    *,
    client: object,
) -> CorrectionResult:
    """Run generate, validate, and route for a single stale section."""
    generated = await generate_correction(
        section, change, staleness, client=client
    )
    validated = await validate_correction(
        section, generated, change, client=client
    )
    return route_correction(validated)


async def repair_all(
    staleness_results: list[StalenessResult],
    sections: list[DocSection],
    changes: list[CodeChange],
    *,
    client: object | None = None,
) -> list[CorrectionResult]:
    """Repair all stale sections concurrently."""
    llm_client = client or instructor.from_openai(AsyncOpenAI())
    sections_by_id = _lookup_sections(sections)
    changes_by_id = _lookup_changes(changes)

    tasks: list[asyncio.Task[CorrectionResult]] = []
    for staleness in staleness_results:
        if not staleness.is_stale:
            continue

        section = sections_by_id.get(staleness.doc_section_id)
        change = changes_by_id.get(staleness.code_chunk_id)
        if section is None or change is None:
            log.warning(
                "repair_skipped_missing_context",
                doc_section_id=staleness.doc_section_id,
                code_chunk_id=staleness.code_chunk_id,
                has_section=section is not None,
                has_change=change is not None,
            )
            continue

        tasks.append(
            asyncio.create_task(
                _repair_one(staleness, section, change, client=llm_client)
            )
        )

    if not tasks:
        return []

    return list(await asyncio.gather(*tasks))
