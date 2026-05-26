"""Embed code and docs in ChromaDB and build a doc-to-code link graph."""

from __future__ import annotations

import json
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field

from parser import CodeChunk, DocSection

SIMILARITY_THRESHOLD = 0.75
EMBEDDING_MODEL = "text-embedding-3-small"
CODE_COLLECTION = "code_chunks"
DOC_COLLECTION = "doc_sections"


def get_similarity_threshold() -> float:
    """Return link-graph cosine similarity cutoff from env or default."""
    raw = os.environ.get("SIMILARITY_THRESHOLD", str(SIMILARITY_THRESHOLD))
    try:
        return float(raw)
    except ValueError:
        return SIMILARITY_THRESHOLD


class LinkGraph(BaseModel):
    """Maps documentation sections to semantically linked code chunks."""

    links: dict[str, list[str]] = Field(default_factory=dict)
    similarity_threshold: float = SIMILARITY_THRESHOLD


def _code_embed_text(chunk: CodeChunk) -> str:
    parts = [chunk.signature]
    if chunk.docstring:
        parts.append(chunk.docstring)
    return "\n".join(parts)


def _doc_embed_text(section: DocSection) -> str:
    heading = " > ".join(section.heading_path)
    return f"{heading}\n\n{section.body}".strip()


def _get_client(persist_dir: Path) -> chromadb.PersistentClient:
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def _embedding_function() -> embedding_functions.OpenAIEmbeddingFunction:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        msg = "OPENAI_API_KEY environment variable is required for indexing."
        raise RuntimeError(msg)
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBEDDING_MODEL,
    )


def _reset_collection(
    client: chromadb.PersistentClient,
    name: str,
    embed_fn: embedding_functions.OpenAIEmbeddingFunction,
) -> chromadb.Collection:
    try:
        client.delete_collection(name)
    except ValueError:
        pass
    return client.create_collection(name=name, embedding_function=embed_fn)


def index_embeddings(
    code_chunks: list[CodeChunk],
    doc_sections: list[DocSection],
    persist_dir: Path,
) -> tuple[int, int]:
    """Store chunks in ChromaDB. Returns (code_count, doc_count)."""
    embed_fn = _embedding_function()
    client = _get_client(persist_dir)

    code_col = _reset_collection(client, CODE_COLLECTION, embed_fn)
    doc_col = _reset_collection(client, DOC_COLLECTION, embed_fn)

    if code_chunks:
        code_col.add(
            ids=[chunk.key for chunk in code_chunks],
            documents=[_code_embed_text(chunk) for chunk in code_chunks],
            metadatas=[
                {
                    "filepath": chunk.filepath,
                    "name": chunk.name,
                    "lineno": chunk.lineno,
                    "parent": chunk.parent or "",
                }
                for chunk in code_chunks
            ],
        )

    if doc_sections:
        doc_col.add(
            ids=[section.key for section in doc_sections],
            documents=[_doc_embed_text(section) for section in doc_sections],
            metadatas=[
                {
                    "filepath": section.filepath,
                    "heading_level": section.heading_level,
                    "lineno": section.lineno,
                    "heading_path": "::".join(section.heading_path),
                }
                for section in doc_sections
            ],
        )

    return len(code_chunks), len(doc_sections)


def build_link_graph(
    doc_sections: list[DocSection],
    persist_dir: Path,
    threshold: float | None = None,
) -> dict[str, list[dict[str, str | float]]]:
    """Map each doc section to code chunks above the similarity threshold."""
    cutoff = threshold if threshold is not None else get_similarity_threshold()
    if not doc_sections:
        return {}

    embed_fn = _embedding_function()
    client = _get_client(persist_dir)
    try:
        code_col = client.get_collection(
            name=CODE_COLLECTION,
            embedding_function=embed_fn,
        )
    except ValueError:
        return {}

    if code_col.count() == 0:
        return {}

    links: dict[str, list[dict[str, str | float]]] = {}

    for section in doc_sections:
        query_text = _doc_embed_text(section)
        results = code_col.query(
            query_texts=[query_text],
            n_results=min(code_col.count(), 50),
            include=["distances", "metadatas"],
        )

        section_links: list[dict[str, str | float]] = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, distance in zip(ids, distances, strict=False):
            similarity = 1.0 - distance
            if similarity >= cutoff:
                section_links.append(
                    {
                        "code_key": chunk_id,
                        "similarity": round(similarity, 4),
                    }
                )

        section_links.sort(key=lambda item: float(item["similarity"]), reverse=True)
        if section_links:
            links[section.key] = section_links

    return links


def save_links(links: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(links, indent=2), encoding="utf-8")


def load_link_graph(
    links_path: Path,
    threshold: float | None = None,
) -> LinkGraph:
    """Load a persisted link graph from *links_path*."""
    cutoff = threshold if threshold is not None else get_similarity_threshold()
    if not links_path.is_file():
        return LinkGraph(similarity_threshold=cutoff)

    raw = json.loads(links_path.read_text(encoding="utf-8"))
    links: dict[str, list[str]] = {}
    for doc_key, code_links in raw.items():
        links[doc_key] = [item["code_key"] for item in code_links]

    return LinkGraph(links=links, similarity_threshold=cutoff)


def load_doc_sections(
    section_ids: list[str],
    persist_dir: Path,
) -> dict[str, DocSection]:
    """Load documentation sections from ChromaDB by id."""
    if not section_ids:
        return {}

    embed_fn = _embedding_function()
    client = _get_client(persist_dir)
    try:
        doc_col = client.get_collection(
            name=DOC_COLLECTION,
            embedding_function=embed_fn,
        )
    except ValueError:
        return {}

    results = doc_col.get(ids=section_ids, include=["metadatas", "documents"])
    sections: dict[str, DocSection] = {}

    ids = results.get("ids") or []
    metadatas = results.get("metadatas") or []
    documents = results.get("documents") or []

    for section_id, metadata, document in zip(ids, metadatas, documents, strict=False):
        if metadata is None or document is None:
            continue
        heading_path_raw = metadata.get("heading_path", "")
        heading_path = heading_path_raw.split("::") if heading_path_raw else []
        sections[section_id] = DocSection(
            key=section_id,
            filepath=str(metadata.get("filepath", "")),
            heading_path=heading_path,
            heading_level=int(metadata.get("heading_level", 1)),
            body=document,
            lineno=int(metadata.get("lineno", 1)),
        )

    return sections


def index_repo(
    code_chunks: list[CodeChunk],
    doc_sections: list[DocSection],
    output_dir: Path,
) -> tuple[int, int, int]:
    """Embed, link, and persist artifacts. Returns (code_count, doc_count, link_count)."""
    chroma_dir = output_dir / "chroma"
    code_count, doc_count = index_embeddings(code_chunks, doc_sections, chroma_dir)
    links = build_link_graph(doc_sections, chroma_dir)
    save_links(links, output_dir / "links.json")
    return code_count, doc_count, len(links)
