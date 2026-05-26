"""DocSentinel CLI — index a repo's code and documentation."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from detector import (
    StalenessResult,
    collect_verification_pairs,
    fetch_git_diff,
    find_affected_sections,
    parse_diff,
    verify_staleness_batch,
)
from embedder import index_repo, load_doc_sections, load_link_graph
from github_client import GitHubClient, write_github_outputs
from parser import parse_repo
from rewriter import CorrectionResult, repair_all

app = typer.Typer(
    name="docsentinel",
    help="Detect stale documentation and auto-generate fix PRs.",
    add_completion=False,
)
console = Console()


@app.callback()
def main() -> None:
    """Detect stale documentation and auto-generate fix PRs."""


@app.command()
def index(
    repo_path: Path = typer.Option(
        Path("."),
        "--repo-path",
        help="Root of the repository to index.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        Path(".docsentinel"),
        "--output-dir",
        help="Directory for ChromaDB data and links.json.",
    ),
) -> None:
    """Parse, embed, and link code chunks to documentation sections."""
    root = repo_path.resolve()
    console.print(Panel(f"[bold]Indexing[/bold]  {root}", border_style="blue"))

    with console.status("[bold green]Parsing repository…"):
        code_chunks, doc_sections = parse_repo(root)

    console.print(f"  Parsed [cyan]{len(code_chunks)}[/cyan] code chunks")
    console.print(f"  Parsed [cyan]{len(doc_sections)}[/cyan] doc sections")

    with console.status("[bold green]Embedding and building link graph…"):
        code_count, doc_count, link_count = index_repo(
            code_chunks,
            doc_sections,
            output_dir,
        )

    table = Table(title="Index Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Count", justify="right")
    table.add_row("Code chunks indexed", str(code_count))
    table.add_row("Doc sections indexed", str(doc_count))
    table.add_row("Doc sections with links", str(link_count))
    console.print(table)
    console.print(f"\n[dim]Artifacts written to[/dim] [bold]{output_dir.resolve()}[/bold]")


@app.command()
def detect(
    repo_path: Path = typer.Option(
        Path("."),
        "--repo-path",
        help="Root of the repository to scan.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    base_branch: str = typer.Option(
        "main",
        "--base-branch",
        help="Base branch to diff against (uses git diff base...HEAD).",
    ),
    output_dir: Path = typer.Option(
        Path(".docsentinel"),
        "--output-dir",
        help="Directory containing ChromaDB data and links.json.",
    ),
) -> None:
    """Detect stale documentation from changes since *base_branch*."""
    root = repo_path.resolve()
    console.print(
        Panel(
            f"[bold]Detecting stale docs[/bold]  {root}\n"
            f"[dim]base:[/dim] {base_branch}...HEAD",
            border_style="blue",
        )
    )

    with console.status("[bold green]Fetching git diff…"):
        diff_text = fetch_git_diff(root, base_branch)

    with console.status("[bold green]Parsing diff…"):
        changes = parse_diff(diff_text)

    meaningful = [change for change in changes if change.is_meaningful]
    console.print(
        f"  Found [cyan]{len(changes)}[/cyan] changed functions "
        f"([cyan]{len(meaningful)}[/cyan] meaningful)"
    )

    links_path = output_dir / "links.json"
    chroma_dir = output_dir / "chroma"
    link_graph = load_link_graph(links_path)

    with console.status("[bold green]Finding affected doc sections…"):
        affected = find_affected_sections(meaningful, link_graph, chroma_dir)

    pairs = collect_verification_pairs(affected)
    console.print(f"  Checking [cyan]{len(pairs)}[/cyan] change/section pairs")

    with console.status("[bold green]Verifying staleness with LLM…"):
        results = asyncio.run(verify_staleness_batch(pairs))

    table = Table(
        title="Staleness Report",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Filepath", style="cyan", no_wrap=True)
    table.add_column("Section", style="dim")
    table.add_column("Stale", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Reason")

    for (change, section, _chunk_id), result in zip(pairs, results, strict=False):
        heading = (
            " > ".join(section.heading_path)
            if section.heading_path
            else section.filepath
        )
        stale_label = "[red]yes[/red]" if result.is_stale else "[green]no[/green]"
        table.add_row(
            change.filepath,
            heading,
            stale_label,
            f"{result.confidence:.2f}",
            result.reason or "",
        )

    console.print(table)
    stale_count = sum(1 for result in results if result.is_stale)
    console.print(
        f"\n[bold]{stale_count}[/bold] stale section(s) out of {len(results)} checked"
    )


@app.command()
def repair(
    repo_path: Path = typer.Option(
        Path("."),
        "--repo-path",
        help="Root of the repository to scan.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    staleness_report: Path = typer.Option(
        ...,
        "--staleness-report",
        help="JSON file containing StalenessResult records.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    base_branch: str = typer.Option(
        "main",
        "--base-branch",
        help="Base branch to diff against for loading code changes.",
    ),
    output_dir: Path = typer.Option(
        Path(".docsentinel"),
        "--output-dir",
        help="Directory containing ChromaDB data for doc sections.",
    ),
    corrections_output: Path = typer.Option(
        Path("corrections.json"),
        "--corrections-output",
        help="Path to write correction results JSON.",
    ),
) -> None:
    """Repair stale documentation sections from a staleness report."""
    root = repo_path.resolve()
    console.print(
        Panel(
            f"[bold]Repairing stale docs[/bold]  {root}\n"
            f"[dim]report:[/dim] {staleness_report}",
            border_style="blue",
        )
    )

    raw = staleness_report.read_text(encoding="utf-8")
    staleness_results = [StalenessResult.model_validate(item) for item in json.loads(raw)]

    stale = [result for result in staleness_results if result.is_stale]
    console.print(
        f"  Loaded [cyan]{len(staleness_results)}[/cyan] staleness result(s) "
        f"([cyan]{len(stale)}[/cyan] stale)"
    )

    with console.status("[bold green]Fetching git diff…"):
        diff_text = fetch_git_diff(root, base_branch)

    with console.status("[bold green]Parsing diff…"):
        changes = parse_diff(diff_text)

    doc_ids = sorted({result.doc_section_id for result in stale})
    chroma_dir = output_dir / "chroma"
    with console.status("[bold green]Loading doc sections…"):
        sections_by_id = load_doc_sections(doc_ids, chroma_dir)
    sections = list(sections_by_id.values())

    console.print(f"  Repairing [cyan]{len(stale)}[/cyan] stale section(s)")

    with console.status("[bold green]Generating and validating corrections…"):
        corrections = asyncio.run(
            repair_all(staleness_results, sections, changes)
        )

    table = Table(
        title="Correction Report",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("Mode", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Changes Summary")

    for correction in corrections:
        mode_style = (
            "[green]auto_fix[/green]"
            if correction.correction_mode == "auto_fix"
            else "[yellow]human_review[/yellow]"
        )
        table.add_row(
            correction.doc_section_id,
            mode_style,
            f"{correction.confidence:.2f}",
            correction.changes_summary,
        )

    console.print(table)

    payload = [correction.model_dump() for correction in corrections]
    corrections_output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    console.print(
        f"\n[bold]{len(corrections)}[/bold] correction(s) written to "
        f"[bold]{corrections_output.resolve()}[/bold]"
    )


@app.command()
def run(
    repo_path: Path = typer.Option(
        Path("."),
        "--repo-path",
        help="Root of the repository to scan.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    base_branch: str = typer.Option(
        "main",
        "--base-branch",
        help="Base branch for fix PRs and link graph context.",
    ),
    pr_number: int = typer.Option(
        ...,
        "--pr-number",
        help="Pull request number that triggered this run.",
    ),
    output_dir: Path = typer.Option(
        Path(".docsentinel"),
        "--output-dir",
        help="Directory containing ChromaDB data and links.json.",
    ),
    github_token: str | None = typer.Option(
        None,
        "--github-token",
        help="GitHub token (defaults to GITHUB_TOKEN env var).",
    ),
    github_repo: str | None = typer.Option(
        None,
        "--github-repo",
        help="GitHub repo owner/name (defaults to GITHUB_REPOSITORY env var).",
    ),
) -> None:
    """Run the full DocSentinel pipeline for a pull request."""
    root = repo_path.resolve()
    token = github_token or os.environ.get("GITHUB_TOKEN")
    repo_name = github_repo or os.environ.get("GITHUB_REPOSITORY")
    if not token:
        raise typer.BadParameter(
            "GitHub token required: set GITHUB_TOKEN or pass --github-token."
        )
    if not repo_name:
        raise typer.BadParameter(
            "GitHub repo required: set GITHUB_REPOSITORY or pass --github-repo."
        )

    console.print(
        Panel(
            f"[bold]DocSentinel run[/bold]  {root}\n"
            f"[dim]PR:[/dim] #{pr_number}  "
            f"[dim]repo:[/dim] {repo_name}",
            border_style="blue",
        )
    )

    client = GitHubClient(token=token, repo_name=repo_name)

    with console.status("[bold green]Fetching PR diff…"):
        diff_text = asyncio.run(client.fetch_pr_diff(pr_number))

    with console.status("[bold green]Parsing diff…"):
        changes = parse_diff(diff_text)

    meaningful = [change for change in changes if change.is_meaningful]
    console.print(
        f"  Found [cyan]{len(changes)}[/cyan] changed functions "
        f"([cyan]{len(meaningful)}[/cyan] meaningful)"
    )

    links_path = output_dir / "links.json"
    chroma_dir = output_dir / "chroma"
    link_graph = load_link_graph(links_path)

    with console.status("[bold green]Finding affected doc sections…"):
        affected = find_affected_sections(meaningful, link_graph, chroma_dir)

    pairs = collect_verification_pairs(affected)
    all_sections = [
        section
        for sections in affected.values()
        for section in sections
    ]
    console.print(f"  Checking [cyan]{len(pairs)}[/cyan] change/section pairs")

    with console.status("[bold green]Verifying staleness with LLM…"):
        staleness_results = asyncio.run(verify_staleness_batch(pairs))

    stale_results = [result for result in staleness_results if result.is_stale]
    console.print(
        f"  Found [cyan]{len(stale_results)}[/cyan] stale section(s) "
        f"out of {len(staleness_results)} checked"
    )

    corrections: list[CorrectionResult] = []
    if stale_results:
        with console.status("[bold green]Generating and validating corrections…"):
            corrections = asyncio.run(
                repair_all(stale_results, all_sections, meaningful)
            )

    auto_fix = [item for item in corrections if item.correction_mode == "auto_fix"]
    flagged = [item for item in corrections if item.correction_mode == "human_review"]

    fix_pr_url: str | None = None
    if auto_fix:
        with console.status("[bold green]Creating fix PR…"):
            fix_pr_url = asyncio.run(
                client.create_fix_pr(base_branch, auto_fix, all_sections)
            )
        console.print(f"  Fix PR: [link={fix_pr_url}]{fix_pr_url}[/link]")

    if flagged:
        with console.status("[bold green]Posting review comment…"):
            asyncio.run(client.post_review_comment(pr_number, flagged, all_sections))

    with console.status("[bold green]Posting summary comment…"):
        asyncio.run(
            client.post_summary_comment(
                pr_number,
                len(auto_fix),
                len(flagged),
                fix_pr_url,
                verified_count=len(pairs),
            )
        )

    summary_lines = [
        f"[green]✓[/green] {len(pairs)} sections verified",
        f"[cyan]{len(auto_fix)}[/cyan] auto-fixed",
        f"[yellow]{len(flagged)}[/yellow] flagged for review",
    ]
    if fix_pr_url:
        summary_lines.append(f"Fix PR: {fix_pr_url}")
    console.print(
        Panel(
            "\n".join(summary_lines),
            title="DocSentinel Summary",
            border_style="green",
        )
    )

    write_github_outputs(
        stale_sections_count=len(stale_results),
        auto_fixed_count=len(auto_fix),
        flagged_count=len(flagged),
    )


if __name__ == "__main__":
    app()
