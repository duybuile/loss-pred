"""
Vector store — ChromaDB-backed document retrieval, ready to use.

The collection is initialised during FastAPI startup via app.main. Subsequent
calls to retrieve() use the cached in-process client — no re-loading on each
request.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from app import cfg
from app.embedding import (
    SafeHuggingFaceEmbeddingFunction,
    embedding_functions,
    get_embedding_fn,
)

load_dotenv()

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
_DOCS_DIR = _REPO_ROOT / cfg.get("data.documents_path")
_CHROMA_DIR = _REPO_ROOT / cfg.get("vectorstore.persist_directory")

_embedding_fn: object | None = None
_collection: chromadb.Collection | None = None
_SafeHuggingFaceEmbeddingFunction = SafeHuggingFaceEmbeddingFunction


def _get_collection_name() -> str:
    base_name = cfg.get("vectorstore.collection_name")
    provider = cfg.get("vectorstore.embedding_provider").lower()
    return f"{base_name}_{provider}"


def _get_collection_marker_path() -> Path:
    return _CHROMA_DIR / f"{_get_collection_name()}.ready.json"


def _write_collection_marker(document_count: int) -> None:
    marker_path = _get_collection_marker_path()
    marker_path.write_text(json.dumps({"document_count": document_count}), encoding="utf-8")


def _read_collection_marker() -> dict | None:
    marker_path = _get_collection_marker_path()
    if not marker_path.exists():
        return None
    return json.loads(marker_path.read_text(encoding="utf-8"))


def _get_embedding_fn() -> object:
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = get_embedding_fn()
    return _embedding_fn


def init() -> None:
    _get_collection()


def _load_documents() -> tuple[list[str], list[str], list[dict[str, str]]]:
    doc_paths = sorted(_DOCS_DIR.glob("*.txt"))
    if not doc_paths:
        logger.warning("No documents found in %s. Run generate_data.py first.", _DOCS_DIR)
        raise FileNotFoundError(
            f"No documents found in {_DOCS_DIR}. "
            "Run generate_data.py first."
        )

    ids, documents, metadatas = [], [], []
    for path in doc_paths:
        record_id = path.stem
        text = path.read_text()
        ids.append(record_id)
        documents.append(text)
        metadatas.append({"record_id": record_id})

    return ids, documents, metadatas


def _populate_collection(collection: chromadb.Collection) -> int:
    ids, documents, metadatas = _load_documents()
    batch_size = cfg.get("vectorstore.batch_size")
    total_batches = math.ceil(len(ids) / batch_size)

    logger.info(
        "Populating ChromaDB collection %s with %d documents across %d batches",
        _get_collection_name(),
        len(ids),
        total_batches,
    )

    for batch_index, start in enumerate(range(0, len(ids), batch_size), start=1):
        batch_start = time.perf_counter()
        end = start + batch_size
        logger.info(
            "Populating ChromaDB collection %s batch %d/%d with %d documents",
            _get_collection_name(),
            batch_index,
            total_batches,
            len(ids[start:end]),
        )
        logger.info(
            "Calling Chroma add for collection %s batch %d/%d",
            _get_collection_name(),
            batch_index,
            total_batches,
        )
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        logger.info(
            "Finished ChromaDB collection %s batch %d/%d in %.2fs",
            _get_collection_name(),
            batch_index,
            total_batches,
            time.perf_counter() - batch_start,
        )
    return len(ids)


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is not None:
        logger.debug("Using cached ChromaDB collection")
        return _collection

    logger.debug("ChromaDB collection is not cached. Fetching it from the collection...")
    client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
    collection_name = _get_collection_name()
    marker = _read_collection_marker()

    existing = {c.name for c in client.list_collections()}
    if collection_name in existing:
        if marker is None:
            logger.warning(
                "Existing ChromaDB collection %s has no ready marker. Deleting and rebuilding.",
                collection_name,
            )
            client.delete_collection(name=collection_name)
        else:
            _collection = client.get_collection(
                name=collection_name,
                embedding_function=_get_embedding_fn(),
            )
            logger.debug("Using existing ChromaDB collection")
            logger.info(
                "Loaded ChromaDB collection %s with %d documents",
                collection_name,
                marker.get("document_count", -1),
            )
            return _collection

    _load_documents()
    try:
        collection = client.create_collection(
            name=collection_name,
            embedding_function=_get_embedding_fn(),
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        if "already exists" not in str(exc):
            raise
        logger.warning(
            "Chroma reported collection %s already exists during create. "
            "Deleting and recreating it.",
            collection_name,
        )
        client.delete_collection(name=collection_name)
        collection = client.create_collection(
            name=collection_name,
            embedding_function=_get_embedding_fn(),
            metadata={"hnsw:space": "cosine"},
        )
    logger.debug("Created new ChromaDB collection")
    document_count = _populate_collection(collection)
    _collection = collection
    _write_collection_marker(document_count)
    logger.info(
        "Loaded ChromaDB collection %s with %d documents",
        collection_name,
        document_count,
    )
    return _collection


def retrieve(query: str, n_results: int = 3) -> list[dict]:
    collection = _get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append(
            {
                "record_id": meta["record_id"],
                "document": doc,
                "distance": round(dist, 4),
            }
        )
    return output
