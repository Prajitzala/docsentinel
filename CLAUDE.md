# DocSentinel

GitHub Action that detects stale documentation when code changes, then auto-generates
a corrected PR or flags sections for human review. No ML training — embeddings + LLM only.

## Commands

```bash
# Install
pip install -r requirements.txt

# Index a repo (Phase 1)
python src/main.py index --repo-path ./

# Detect staleness from a git diff (Phase 2)
python src/main.py detect --repo-path ./ --base-branch main

# Repair stale sections (Phase 3)
python src/main.py repair --repo-path ./ --staleness-report ./staleness.json

# Full run (Phase 4+)
python src/main.py run --repo-path ./ --base-branch main --github-pr 42

# Tests
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing

# Lint
ruff check src/ tests/
mypy src/
```

## Architecture

```
git diff (PR)
     │
     ▼
detector.py ──── filters meaningful changes ────► CodeChange[]
     │
     ▼
embedder.py ──── queries link graph ──────────► affected DocSection[]
     │
     ▼
detector.py ──── LLM staleness check ────────► StalenessResult[]
     │
     ├── confidence >= AUTO_FIX_CONFIDENCE
     │        ▼
     │   rewriter.py ── generates correction ── github_client.py ── opens PR
     │
     └── confidence < AUTO_FIX_CONFIDENCE
              ▼
         github_client.py ── posts PR comment flagging section for review
```

## File Responsibilities

| File | Does | Does NOT |
|------|------|----------|
| `parser.py` | AST parsing, markdown splitting, CodeChunk/DocSection extraction | Any I/O, embedding, LLM calls |
| `embedder.py` | ChromaDB reads/writes, OpenAI embedding calls, link graph | Parsing, LLM generation |
| `detector.py` | Git diff parsing, meaningful-change filtering, staleness LLM check | Writing to ChromaDB, PR operations |
| `rewriter.py` | LLM doc correction, confidence scoring, correction mode routing | GitHub API, ChromaDB |
| `github_client.py` | PR creation, PR comments, diff fetching via PyGithub | Any LLM calls, file parsing |
| `main.py` | CLI wiring via Typer, orchestration | Business logic |

## Key Data Models

```python
CodeChunk(id, filepath, function_name, signature, docstring, body_hash, line_start, line_end)
DocSection(id, filepath, heading_path, content, heading_level, code_references)
LinkGraph(links: dict[doc_section_id, list[code_chunk_id]], similarity_threshold)
CodeChange(filepath, function_name, change_type, old_content, new_content, is_meaningful)
StalenessResult(doc_section_id, code_chunk_id, is_stale, reason, confidence)
CorrectionResult(doc_section_id, original_content, corrected_content, confidence, correction_mode, changes_summary)
```

All models are Pydantic v2. Never redefine these — import from their source module.

## Environment Variables

```bash
OPENAI_API_KEY=           # required always
GITHUB_TOKEN=             # required for PR operations only
CHROMA_DB_PATH=./.chroma  # where ChromaDB persists
SIMILARITY_THRESHOLD=0.75 # link graph cosine sim cutoff
AUTO_FIX_CONFIDENCE=0.85  # above = auto PR, below = human review flag
LOG_LEVEL=INFO
```

Copy `.env.example` to `.env` for local dev. Never commit `.env`.

## Tech Stack

- Python 3.11, Pydantic v2, ChromaDB (PersistentClient only), openai SDK
- instructor (structured LLM outputs — never parse JSON manually)
- PyGithub, Typer, Rich, structlog, anyio, pytest

## Coding Rules

- `pathlib.Path` for all paths — never `os.path`
- `structlog` for all logging — never `print()`
- `instructor` for all LLM structured output — never manual JSON parsing
- All async in `embedder.py`, `rewriter.py`, `github_client.py`
- All sync in `parser.py`, `detector.py`
- Prompts live in `src/prompts/*.txt` — never inline in code
- Mock all OpenAI + GitHub calls in tests — never hit real APIs

## LLM Call Pattern

```python
import instructor
from openai import AsyncOpenAI

client = instructor.from_openai(AsyncOpenAI())
result = await client.chat.completions.create(
    model="gpt-4o",
    response_model=StalenessResult,  # always a Pydantic model
    messages=[...],
    max_retries=2,
)
```

## ChromaDB Pattern

```python
client = chromadb.PersistentClient(path=str(settings.chroma_db_path))
collection = client.get_or_create_collection("code_chunks")
collection.add(ids=[...], embeddings=[...], metadatas=[...], documents=[...])
results = collection.query(query_embeddings=[...], n_results=10)
```

## Git Diff — Skip These (not meaningful)
- Test files (`test_*.py`, `*.test.*`, `*.spec.*`)
- Comment-only or whitespace-only changes
- Import-only changes in `__init__.py`
- Docstring changes without signature changes

## GitHub Action Outputs
- `stale_sections_count` — total stale sections found
- `auto_fixed_count` — sections auto-corrected via PR
- `flagged_count` — sections flagged for human review
- Exit 0 always unless system error (stale docs = expected, not an error)

## Current Build Phase

**v0.1.0 — shipped**

- [x] `parser.py` — CodeChunk extraction via ast, DocSection via markdown headings
- [x] `embedder.py` — ChromaDB collections + link graph builder
- [x] `main.py` — `index`, `detect`, `repair`, `run` commands
- [x] `tests/fixtures/` — sample .py and .md files
- [x] `detector.py` — git diff parsing + LLM staleness check
- [x] `rewriter.py` — doc repair + confidence routing
- [x] `github_client.py` — PR creation + comment posting
- [x] `action.yml` + `Dockerfile` — GitHub Actions packaging
- [x] CI workflow — index step before Docker `run` on each PR

## Design Decisions & Rationale

**ChromaDB over Pinecone/Qdrant** — file-based, zero infrastructure, persists to disk,
works inside a GitHub Actions runner without a sidecar service.

**instructor over raw JSON parsing** — structured outputs with automatic retry on
validation failure. No brittle JSON parsing.

**confidence-based routing** — above `AUTO_FIX_CONFIDENCE` threshold gets an auto PR;
below gets a PR comment flagging for human review. This prevents bad corrections from
merging silently while still automating the easy cases.

**meaningful-change filtering** — test files, whitespace, comments are filtered before
any LLM calls. Reduces cost and false positives significantly.

**two-pass LLM for rewriter** — first pass generates correction, second pass validates
it preserved accurate sections and didn't introduce new errors.

## Portfolio Notes (for README + blog post)

Key metrics to track and report:
- True positive rate (correctly identified stale docs)
- False positive rate (flagged accurate docs as stale)
- Auto-fix acceptance rate (did the auto-generated PR actually get merged?)
- Cost per repo scan (OpenAI API tokens)

Test on a forked OSS repo (FastAPI or Pydantic) by deliberately making code changes
that should invalidate docs. Screenshot the PR comment and auto-fix PR for portfolio.