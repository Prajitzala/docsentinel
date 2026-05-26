"""GitHub PR operations — diff fetching, fix PRs, and review comments."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import anyio
import httpx
import structlog
from github import Github
from github.InputGitTreeElement import InputGitTreeElement

from parser import DocSection
from rewriter import CorrectionResult

if TYPE_CHECKING:
    from github.Repository import Repository

log = structlog.get_logger()

COMMIT_MESSAGE = "docs: auto-fix stale sections [DocSentinel]"
PR_TITLE = "docs: auto-fix stale sections [DocSentinel]"
BRANCH_PREFIX = "docsentinel/fix"
GITHUB_DIFF_ACCEPT = "application/vnd.github.v3.diff"


def _heading_label(section: DocSection) -> str:
    if section.heading_path:
        return " > ".join(section.heading_path)
    return section.filepath


def _heading_anchor(section: DocSection) -> str:
    if not section.heading_path:
        return ""
    title = section.heading_path[-1].lower()
    slug = re.sub(r"[^\w\s-]", "", title)
    return slug.replace(" ", "-")


def _section_link(section: DocSection) -> str:
    label = _heading_label(section)
    anchor = _heading_anchor(section)
    if anchor:
        return f"[{label}]({section.filepath}#{anchor})"
    return f"[{label}]({section.filepath})"


def _diff_block(before: str, after: str) -> str:
    lines: list[str] = []
    for line in before.splitlines():
        lines.append(f"- {line}")
    for line in after.splitlines():
        lines.append(f"+ {line}")
    body = "\n".join(lines) if lines else "- (empty)\n+ (empty)"
    return f"```diff\n{body}\n```"


def _build_fix_pr_body(
    corrections: list[CorrectionResult],
    sections_by_id: dict[str, DocSection],
) -> str:
    parts = [
        "## DocSentinel auto-fixes",
        "",
        "The following documentation sections were updated to match recent code changes.",
        "",
    ]
    for correction in corrections:
        section = sections_by_id[correction.doc_section_id]
        heading = _heading_label(section)
        parts.append(f"### {section.filepath} — {heading}")
        parts.append("")
        parts.append(_diff_block(correction.original_content, correction.corrected_content))
        parts.append("")
        parts.append(f"**Summary:** {correction.changes_summary}")
        parts.append("")
    return "\n".join(parts).strip()


def _build_review_comment_body(
    flagged: list[CorrectionResult],
    sections_by_id: dict[str, DocSection],
) -> str:
    lines = [
        "## DocSentinel — sections needing manual review",
        "",
        "| Section | Reason | Confidence |",
        "| --- | --- | ---: |",
    ]
    for correction in flagged:
        section = sections_by_id.get(correction.doc_section_id)
        if section is None:
            link = correction.doc_section_id
        else:
            link = _section_link(section)
        reason = correction.changes_summary.replace("|", "\\|")
        lines.append(
            f"| {link} | {reason} | {correction.confidence:.2f} |"
        )
    lines.append("")
    lines.append(
        f"DocSentinel — {len(flagged)} section(s) need manual review"
    )
    return "\n".join(lines)


def _build_summary_comment_body(
    verified_count: int,
    auto_fixed_count: int,
    flagged_count: int,
    fix_pr_url: str | None,
) -> str:
    parts = [f"✓ {verified_count} sections verified"]
    if auto_fixed_count:
        if fix_pr_url:
            pr_ref = fix_pr_url.rstrip("/").split("/")[-1]
            if pr_ref.isdigit():
                parts.append(f"{auto_fixed_count} auto-fixed (PR #{pr_ref})")
            else:
                parts.append(f"{auto_fixed_count} auto-fixed ({fix_pr_url})")
        else:
            parts.append(f"{auto_fixed_count} auto-fixed")
    else:
        parts.append("0 auto-fixed")
    parts.append(f"{flagged_count} flagged")
    return " · ".join(parts)


def _apply_corrections(
    content: str,
    corrections: list[CorrectionResult],
) -> str:
    updated = content
    for correction in corrections:
        if correction.original_content not in updated:
            msg = (
                f"Could not locate section content in {correction.doc_section_id!r} "
                "for auto-fix."
            )
            raise ValueError(msg)
        updated = updated.replace(
            correction.original_content,
            correction.corrected_content,
            1,
        )
    return updated


def write_github_outputs(
    *,
    stale_sections_count: int,
    auto_fixed_count: int,
    flagged_count: int,
    output_path: str | None = None,
) -> None:
    """Append GitHub Action outputs to *GITHUB_OUTPUT* when set."""
    path = output_path or os.environ.get("GITHUB_OUTPUT")
    if not path:
        return

    payload = {
        "stale_sections_count": stale_sections_count,
        "auto_fixed_count": auto_fixed_count,
        "flagged_count": flagged_count,
    }
    lines = [f"{key}={value}\n" for key, value in payload.items()]
    with open(path, "a", encoding="utf-8") as handle:
        handle.writelines(lines)


class GitHubClient:
    """Async wrapper around PyGithub for DocSentinel PR workflows."""

    def __init__(self, token: str, repo_name: str) -> None:
        self._token = token
        self._repo_name = repo_name
        self._github = Github(token)
        self._repo: Repository = self._github.get_repo(repo_name)

    async def fetch_pr_diff(self, pr_number: int) -> str:
        """Return raw unified diff text for a pull request."""
        url = f"https://api.github.com/repos/{self._repo_name}/pulls/{pr_number}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": GITHUB_DIFF_ACCEPT,
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=60.0)
            response.raise_for_status()
            return response.text

    async def create_fix_pr(
        self,
        base_branch: str,
        corrections: list[CorrectionResult],
        sections: list[DocSection],
    ) -> str:
        """Create a branch with doc fixes and open a PR. Returns the new PR URL."""
        if not corrections:
            msg = "create_fix_pr requires at least one correction."
            raise ValueError(msg)

        sections_by_id = {section.key: section for section in sections}
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        branch_name = f"{BRANCH_PREFIX}-{timestamp}"

        file_updates = await anyio.to_thread.run_sync(
            self._prepare_file_updates,
            base_branch,
            corrections,
            sections_by_id,
        )
        await anyio.to_thread.run_sync(
            self._create_branch_and_commit,
            base_branch,
            branch_name,
            file_updates,
        )
        pr_url = await anyio.to_thread.run_sync(
            self._open_pull_request,
            base_branch,
            branch_name,
            corrections,
            sections_by_id,
        )
        log.info("fix_pr_created", url=pr_url, branch=branch_name, files=len(file_updates))
        return pr_url

    async def post_review_comment(
        self,
        pr_number: int,
        flagged: list[CorrectionResult],
        sections: list[DocSection],
    ) -> None:
        """Post a single PR comment listing sections that need human review."""
        if not flagged:
            return

        sections_by_id = {section.key: section for section in sections}
        body = _build_review_comment_body(flagged, sections_by_id)
        await anyio.to_thread.run_sync(
            self._create_issue_comment,
            pr_number,
            body,
        )

    async def post_summary_comment(
        self,
        pr_number: int,
        auto_fixed: int,
        flagged: int,
        fix_pr_url: str | None,
        *,
        verified_count: int,
    ) -> None:
        """Post the final run summary comment on the triggering PR."""
        body = _build_summary_comment_body(verified_count, auto_fixed, flagged, fix_pr_url)
        await anyio.to_thread.run_sync(
            self._create_issue_comment,
            pr_number,
            body,
        )

    def _prepare_file_updates(
        self,
        base_branch: str,
        corrections: list[CorrectionResult],
        sections_by_id: dict[str, DocSection],
    ) -> dict[str, str]:
        by_file: dict[str, list[CorrectionResult]] = {}
        for correction in corrections:
            section = sections_by_id.get(correction.doc_section_id)
            if section is None:
                msg = f"Missing DocSection for {correction.doc_section_id!r}"
                raise ValueError(msg)
            by_file.setdefault(section.filepath, []).append(correction)

        updates: dict[str, str] = {}
        for filepath, file_corrections in by_file.items():
            contents = self._repo.get_contents(filepath, ref=base_branch)
            if isinstance(contents, list):
                msg = f"Expected file at {filepath!r}, got a directory listing."
                raise ValueError(msg)
            decoded = contents.decoded_content.decode("utf-8")
            updates[filepath] = _apply_corrections(decoded, file_corrections)
        return updates

    def _create_branch_and_commit(
        self,
        base_branch: str,
        branch_name: str,
        file_updates: dict[str, str],
    ) -> None:
        base_ref = self._repo.get_git_ref(f"heads/{base_branch}")
        base_commit = self._repo.get_git_commit(base_ref.object.sha)
        base_tree = self._repo.get_git_tree(base_commit.tree.sha)

        tree_elements = [
            InputGitTreeElement(
                path=filepath,
                mode="100644",
                type="blob",
                content=content,
            )
            for filepath, content in sorted(file_updates.items())
        ]
        new_tree = self._repo.create_git_tree(tree_elements, base_tree)
        new_commit = self._repo.create_git_commit(
            COMMIT_MESSAGE,
            new_tree,
            [base_commit],
        )
        self._repo.create_git_ref(
            ref=f"refs/heads/{branch_name}",
            sha=new_commit.sha,
        )

    def _open_pull_request(
        self,
        base_branch: str,
        branch_name: str,
        corrections: list[CorrectionResult],
        sections_by_id: dict[str, DocSection],
    ) -> str:
        body = _build_fix_pr_body(corrections, sections_by_id)
        pr = self._repo.create_pull(
            title=PR_TITLE,
            body=body,
            head=branch_name,
            base=base_branch,
        )
        return pr.html_url

    def _create_issue_comment(self, pr_number: int, body: str) -> None:
        pr = self._repo.get_pull(pr_number)
        pr.create_issue_comment(body)
