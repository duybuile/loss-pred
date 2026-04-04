"""
Vector store facade for historical-record retrieval.

The selected backend is initialised during FastAPI startup via app.main.
Subsequent calls to retrieve() use the cached backend-specific client or
vector store instance — no re-loading on each request.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_core.documents import Document

from app import cfg
from app.embedding import (
    SafeHuggingFaceEmbeddingFunction,
    embedding_functions,
    get_embedding_fn,
    get_langchain_embeddings,
)
from app.vectorstore_mgt import VectorStoreMgtChroma

load_dotenv()

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
_DOCS_DIR = _REPO_ROOT / cfg.get("data.documents_path")
_CHROMA_DIR = _REPO_ROOT / cfg.get("vectorstore.persist_directory")

_embedding_fn: object | None = None
_collection: chromadb.Collection | None = None
_backend_adapter: "_VectorStoreBackend | None" = None
_SafeHuggingFaceEmbeddingFunction = SafeHuggingFaceEmbeddingFunction


def _get_collection_name() -> str:
    base_name = cfg.get("vectorstore.collection_name")
    provider = cfg.get("vectorstore.embedding_provider").lower()
    return f"{base_name}_{provider}"


def _get_collection_marker_path() -> Path:
    backend = cfg.get("vectorstore.backend").lower()
    return _CHROMA_DIR / f"{_get_collection_name()}_{backend}.ready.json"


def _get_backend_name() -> str:
    return cfg.get("vectorstore.backend").lower()


def _get_langchain_vector_store_dir() -> Path:
    return _CHROMA_DIR / "langchain_chroma" / _get_collection_name()


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
    _get_backend_adapter().init()


class _VectorStoreBackend:
    def init(self) -> None:
        raise NotImplementedError

    def retrieve(self, query: str, n_results: int) -> list[dict]:
        raise NotImplementedError


class _NativeChromaBackend(_VectorStoreBackend):
    def init(self) -> None:
        _get_collection()

    def retrieve(self, query: str, n_results: int) -> list[dict]:
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


class _LangChainChromaBackend(_VectorStoreBackend):
    def __init__(self) -> None:
        self._manager: VectorStoreMgtChroma | None = None
        self._vector_store = None

    def init(self) -> None:
        if self._vector_store is not None:
            logger.debug("Using cached LangChain Chroma vector store")
            return

        marker = _read_collection_marker()
        if marker is not None:
            vector_store = self._get_manager().get_latest_vector_store()
            if vector_store is not None:
                self._vector_store = vector_store
                logger.info(
                    "Loaded LangChain Chroma collection %s with %d documents",
                    _get_collection_name(),
                    marker.get("document_count", -1),
                )
                return

        vector_store_dir = _get_langchain_vector_store_dir()
        if vector_store_dir.exists():
            logger.warning(
                "Existing LangChain Chroma store %s has no ready marker. Deleting and rebuilding.",
                _get_collection_name(),
            )
            shutil.rmtree(vector_store_dir, ignore_errors=True)

        documents = self._load_langchain_documents()
        logger.info(
            "Building LangChain Chroma collection %s with %d documents",
            _get_collection_name(),
            len(documents),
        )
        manager = self._get_manager()
        manager.batch_update(documents, deduplicate=False)
        self._vector_store = manager.get_latest_vector_store()
        if self._vector_store is None:
            raise RuntimeError(
                f"Failed to initialise LangChain Chroma collection {_get_collection_name()}."
            )

        _write_collection_marker(len(documents))
        logger.info(
            "Loaded LangChain Chroma collection %s with %d documents",
            _get_collection_name(),
            len(documents),
        )

    def retrieve(self, query: str, n_results: int) -> list[dict]:
        self.init()
        assert self._vector_store is not None
        results = self._vector_store.similarity_search_with_score(query, k=n_results)
        output = []
        for doc, score in results:
            record_id = doc.metadata.get("record_id") or doc.metadata.get("doc_id", "unknown")
            output.append(
                {
                    "record_id": record_id,
                    "document": doc.page_content,
                    "distance": round(float(score), 4),
                }
            )
        return output

    def _get_manager(self) -> VectorStoreMgtChroma:
        if self._manager is None:
            self._manager = VectorStoreMgtChroma(
                vector_store_dir=str(_get_langchain_vector_store_dir()),
                embedding_function=_get_embedding_provider_name(),
                embedding_model_name=cfg.get(
                    f"embedding.model.{cfg.get('vectorstore.embedding_provider').lower()}"
                ),
                collection_name=_get_collection_name(),
                embedding=get_langchain_embeddings(),
            )
        return self._manager

    @staticmethod
    def _load_langchain_documents() -> list[Document]:
        ids, documents, _ = _load_documents()
        return [
            Document(
                page_content=document,
                metadata={"record_id": record_id, "doc_id": record_id},
            )
            for record_id, document in zip(ids, documents)
        ]


def _get_embedding_provider_name() -> str:
    return cfg.get("vectorstore.embedding_provider").lower()


def _get_backend_adapter() -> _VectorStoreBackend:
    global _backend_adapter
    if _backend_adapter is not None:
        return _backend_adapter

    backend = _get_backend_name()
    if backend == "native_chroma":
        _backend_adapter = _NativeChromaBackend()
    elif backend == "langchain_chroma":
        _backend_adapter = _LangChainChromaBackend()
    else:
        raise ValueError(f"Unsupported vectorstore backend: {backend}")
    return _backend_adapter


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
    embedding_fn = _get_embedding_fn()

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
        batch_documents = documents[start:end]
        batch_ids = ids[start:end]
        batch_metadatas = metadatas[start:end]
        logger.info(
            "Computing embeddings for collection %s batch %d/%d",
            _get_collection_name(),
            batch_index,
            total_batches,
        )
        embeddings = embedding_fn(batch_documents)
        serializable_embeddings = [
            embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            for embedding in embeddings
        ]
        embedding_dim = len(embeddings[0]) if embeddings else 0
        logger.info(
            "Computed embeddings for collection %s batch %d/%d: %d vectors x %d dims",
            _get_collection_name(),
            batch_index,
            total_batches,
            len(embeddings),
            embedding_dim,
        )
        if serializable_embeddings:
            logger.debug(
                "Prepared embeddings for collection %s batch %d/%d: type=%s dim=%d",
                _get_collection_name(),
                batch_index,
                total_batches,
                type(serializable_embeddings[0]).__name__,
                len(serializable_embeddings[0]),
            )
        logger.info(
            "Calling Chroma add for collection %s batch %d/%d",
            _get_collection_name(),
            batch_index,
            total_batches,
        )
        try:
            collection.add(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas,
                embeddings=serializable_embeddings,
            )
        except Exception as e:
            logger.error("Failed to add batch %d/%d to ChromaDB collection %s: %s", batch_index, total_batches, _get_collection_name(), e)
            raise
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
    return _get_backend_adapter().retrieve(query, n_results)
