"""Tests for github_client.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from github_client import (
    GitHubClient,
    _build_fix_pr_body,
    _build_review_comment_body,
    _build_summary_comment_body,
    write_github_outputs,
)
from parser import DocSection
from rewriter import CorrectionResult


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
def auto_fix_correction(usage_doc_section: DocSection) -> CorrectionResult:
    return CorrectionResult(
        doc_section_id=usage_doc_section.key,
        original_content=usage_doc_section.body,
        corrected_content=(
            "Call `greet(name: str, loud: bool = False)` to say hi. "
            "It returns a greeting string."
        ),
        confidence=0.92,
        correction_mode="auto_fix",
        changes_summary="Updated signature to include loud parameter.",
    )


@pytest.fixture
def human_review_correction(usage_doc_section: DocSection) -> CorrectionResult:
    return CorrectionResult(
        doc_section_id=usage_doc_section.key,
        original_content=usage_doc_section.body,
        corrected_content="Needs manual review for tone and examples.",
        confidence=0.55,
        correction_mode="human_review",
        changes_summary="Uncertain whether loud parameter affects this section.",
    )


@pytest.fixture
def mock_github_repo() -> MagicMock:
    repo = MagicMock()
    repo.full_name = "owner/repo"

    base_ref = MagicMock()
    base_ref.object.sha = "abc123"
    repo.get_git_ref.return_value = base_ref

    base_commit = MagicMock()
    base_tree = MagicMock()
    base_commit.tree.sha = "tree123"
    repo.get_git_commit.return_value = base_commit
    repo.get_git_tree.return_value = base_tree

    new_tree = MagicMock()
    new_commit = MagicMock()
    new_commit.sha = "commit456"
    repo.create_git_tree.return_value = new_tree
    repo.create_git_commit.return_value = new_commit

    file_contents = MagicMock()
    file_contents.decoded_content = (
        b"# Guide\n\n## Usage\nCall `greet(name: str)` to say hi. "
        b"It returns a greeting string.\n"
    )
    repo.get_contents.return_value = file_contents

    pull = MagicMock()
    pull.html_url = "https://github.com/owner/repo/pull/99"
    repo.create_pull.return_value = pull

    issue_pull = MagicMock()
    repo.get_pull.return_value = issue_pull

    return repo


@pytest.fixture
def github_client(mock_github_repo: MagicMock) -> GitHubClient:
    with patch("github_client.Github") as github_cls:
        github_instance = MagicMock()
        github_instance.get_repo.return_value = mock_github_repo
        github_cls.return_value = github_instance
        client = GitHubClient(token="test-token", repo_name="owner/repo")
    client._repo = mock_github_repo
    return client


@pytest.mark.asyncio
async def test_fetch_pr_diff_returns_unified_text(github_client: GitHubClient) -> None:
    diff_text = "diff --git a/src/service.py b/src/service.py\n"

    mock_response = MagicMock()
    mock_response.text = diff_text
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("github_client.httpx.AsyncClient", return_value=mock_client):
        result = await github_client.fetch_pr_diff(42)

    assert result == diff_text
    mock_client.get.assert_awaited_once()
    call_args = mock_client.get.call_args
    assert call_args.args[0] == "https://api.github.com/repos/owner/repo/pulls/42"
    assert call_args.kwargs["headers"]["Authorization"] == "Bearer test-token"
    assert call_args.kwargs["headers"]["Accept"] == "application/vnd.github.v3.diff"


def test_build_fix_pr_body_includes_diff_blocks(
    auto_fix_correction: CorrectionResult,
    usage_doc_section: DocSection,
) -> None:
    body = _build_fix_pr_body(
        [auto_fix_correction],
        {usage_doc_section.key: usage_doc_section},
    )

    assert "docs/guide.md — Guide > Usage" in body
    assert "```diff" in body
    assert "- Call `greet(name: str)`" in body
    assert "+ Call `greet(name: str, loud: bool = False)`" in body
    assert auto_fix_correction.changes_summary in body


def test_build_review_comment_body_table_format(
    human_review_correction: CorrectionResult,
    usage_doc_section: DocSection,
) -> None:
    body = _build_review_comment_body(
        [human_review_correction],
        {usage_doc_section.key: usage_doc_section},
    )

    assert "| Section | Reason | Confidence |" in body
    assert "docs/guide.md#usage" in body
    assert human_review_correction.changes_summary in body
    assert "1 section(s) need manual review" in body


def test_build_summary_comment_with_fix_pr() -> None:
    body = _build_summary_comment_body(
        verified_count=5,
        auto_fixed_count=2,
        flagged_count=1,
        fix_pr_url="https://github.com/owner/repo/pull/99",
    )

    assert body == "✓ 5 sections verified · 2 auto-fixed (PR #99) · 1 flagged"


def test_build_summary_comment_without_fix_pr() -> None:
    body = _build_summary_comment_body(
        verified_count=3,
        auto_fixed_count=0,
        flagged_count=0,
        fix_pr_url=None,
    )

    assert body == "✓ 3 sections verified · 0 auto-fixed · 0 flagged"


@pytest.mark.asyncio
async def test_create_fix_pr_creates_branch_commit_and_pr(
    github_client: GitHubClient,
    mock_github_repo: MagicMock,
    auto_fix_correction: CorrectionResult,
    usage_doc_section: DocSection,
) -> None:
    url = await github_client.create_fix_pr(
        "main",
        [auto_fix_correction],
        [usage_doc_section],
    )

    assert url == "https://github.com/owner/repo/pull/99"
    mock_github_repo.get_contents.assert_called_once_with("docs/guide.md", ref="main")
    mock_github_repo.create_git_ref.assert_called_once()
    branch_ref = mock_github_repo.create_git_ref.call_args.kwargs["ref"]
    assert branch_ref.startswith("refs/heads/docsentinel/fix-")
    mock_github_repo.create_git_commit.assert_called_once()
    assert (
        mock_github_repo.create_git_commit.call_args.args[0]
        == "docs: auto-fix stale sections [DocSentinel]"
    )
    mock_github_repo.create_pull.assert_called_once()
    pull_kwargs = mock_github_repo.create_pull.call_args.kwargs
    assert pull_kwargs["base"] == "main"
    assert pull_kwargs["head"].startswith("docsentinel/fix-")


@pytest.mark.asyncio
async def test_post_review_comment_creates_single_comment(
    github_client: GitHubClient,
    mock_github_repo: MagicMock,
    human_review_correction: CorrectionResult,
    usage_doc_section: DocSection,
) -> None:
    await github_client.post_review_comment(
        42,
        [human_review_correction],
        [usage_doc_section],
    )

    issue_pull = mock_github_repo.get_pull.return_value
    issue_pull.create_issue_comment.assert_called_once()
    body = issue_pull.create_issue_comment.call_args.args[0]
    assert "| Section | Reason | Confidence |" in body


@pytest.mark.asyncio
async def test_post_summary_comment(
    github_client: GitHubClient,
    mock_github_repo: MagicMock,
) -> None:
    await github_client.post_summary_comment(
        42,
        auto_fixed=2,
        flagged=1,
        fix_pr_url="https://github.com/owner/repo/pull/99",
        verified_count=5,
    )

    issue_pull = mock_github_repo.get_pull.return_value
    issue_pull.create_issue_comment.assert_called_once()
    body = issue_pull.create_issue_comment.call_args.args[0]
    assert body == "✓ 5 sections verified · 2 auto-fixed (PR #99) · 1 flagged"


def test_write_github_outputs(tmp_path: Path) -> None:
    output_file = tmp_path / "github_output"
    write_github_outputs(
        stale_sections_count=3,
        auto_fixed_count=2,
        flagged_count=1,
        output_path=str(output_file),
    )

    content = output_file.read_text(encoding="utf-8")
    assert "stale_sections_count=3\n" in content
    assert "auto_fixed_count=2\n" in content
    assert "flagged_count=1\n" in content


def test_write_github_outputs_skips_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    write_github_outputs(
        stale_sections_count=1,
        auto_fixed_count=0,
        flagged_count=1,
    )
