import logging
import os
from typing import List, Tuple, Any

from codetiming import Timer
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.vectorstore_mgt.vector_store_mgt import VectorStoreMgt

logger = logging.getLogger(__name__)


class VectorStoreMgtChroma(VectorStoreMgt):
    def __init__(self, vector_store_dir, embedding_function, collection_name=None, **kwargs):
        super().__init__(vector_store_dir, embedding_function, **kwargs)
        # if collection_name is provided, use it, otherwise use the default collection name
        self.collection_name = collection_name or "default_collection"

    def _load_existing_vectors(self, version_dir):
        """Loads the existing vector store from the latest version."""
        return Chroma(
            collection_name=self.collection_name,
            persist_directory=version_dir,
            embedding_function=self.embedding_function
        )

    @Timer(name="batch_update", text="{name}: {:.2f} seconds", logger=logging.debug)
    def batch_update(self, new_documents: list[Document], reindex_threshold: int = 1000, deduplicate: bool = True):
        """
        Perform batch update of documents with versioning and deduplication.

        Parameters:
        - new_documents (list of str): The new documents to be added.
        - reindex_threshold (int): The threshold to trigger reindexing based on the number of total documents.
        - deduplicate (bool): Whether to deduplicate documents based on hash comparison.
        """
        # Get the latest version of the vector vault
        latest_version, latest_dir = self._get_latest_version()

        # If a vector vault already exists, load it
        if latest_dir:
            logger.info(f"Loading existing vector store. Version: {latest_version}")
            vector_store = self._load_existing_vectors(latest_dir)
        else:
            logger.info("No vector store found. Creating a new one.")
            vector_store = Chroma(
                collection_name=self.collection_name,
                persist_directory=self._create_new_version(),
                embedding_function=self.embedding_function,
            )

        # Deduplicate by checking document hashes
        if deduplicate:
            logger.info("Deduplicating new chunks based on hash comparison.")
            unique_docs = self._deduplicate(new_documents, vector_store)
        else:
            logger.info("Not deduplicating new chunks. Adding all chunks.")
            unique_docs = new_documents

        # Check if there are any new chunks
        if not unique_docs:
            logger.info("No new chunks to add. All chunks are duplicates.")
            return
        if len(new_documents) != len(unique_docs):
            logger.info(f"Deduplicated {len(new_documents) - len(unique_docs)} duplicate chunks.")

        # Prepare texts and metadatas for the vector store
        for doc in unique_docs:
            doc.metadata = doc.metadata or {}
            doc.metadata["content_hash"] = self._hash_document(doc.page_content)

        texts = [doc.page_content for doc in unique_docs]
        metadatas = [doc.metadata or {} for doc in unique_docs]

        logger.info(f"Adding {len(texts)} new chunks to the vector store with metadata.")
        vector_store.add_texts(texts=texts, metadatas=metadatas)

        # Check if reindexing is needed based on total document count
        all_docs = self.list_documents(vector_store)
        doc_ids = [meta.get("doc_id") for meta in all_docs["metadatas"]]
        total_docs = len(set(doc_ids))

        logger.info(f"Total documents in the vector store: {total_docs}")
        if total_docs > reindex_threshold:
            logger.info("Reindexing required. Rebuilding the vector store.")
            self._reindex()

    def _deduplicate(self, new_documents: List[Document], vector_store) -> List[Document]:
        """
        Deduplicate new documents based on hash comparison.
        :param new_documents: a list of new document chunks. Each should contain a page_content and metadata
        Meta-data should be a dictionary, containing keys: "content_hash" and "doc_id"
        :param vector_store:
        :return:
        """
        unique_docs = []
        for doc in new_documents:
            doc_hash = self._hash_document(doc.page_content)
            if not self._document_exists(vector_store, doc_hash):
                # Ensure metadata is always a dict (even if missing)
                if doc.metadata is None:
                    doc.metadata = {}
                unique_docs.append(doc)
        return unique_docs

    def get_latest_vector_store(self):
        """Returns the vector store."""
        latest_version, latest_dir = self._get_latest_version()
        if latest_dir:
            return self._load_existing_vectors(latest_dir)
        else:
            return None

    def _reindex(self):
        """
        Rebuilds the entire vector store by re-indexing all documents.
        @TODO: Not really sure about the all_docs.extend(vector_store._collection.get("documents"))
        """
        logger.info("Starting reindexing process...")

        # Load all existing versions and combine all documents
        all_docs = []
        for version in os.listdir(self.vector_store_dir):
            version_dir = os.path.join(self.vector_store_dir, version)
            if os.path.isdir(version_dir):
                vector_store = self._load_existing_vectors(version_dir)
                all_docs.extend(vector_store._collection.get("documents"))  # Assuming `documents` holds the texts

        # Rebuild from scratch
        reindex_dir = self._create_new_version()
        vector_store = Chroma(persist_directory=reindex_dir, embedding_function=self.embedding_function)
        vector_store.add_texts(all_docs)  # ChromaDB automatically persists the collection after adding texts

        logger.info(f"Reindexing complete. New version stored in {reindex_dir}.")

    @staticmethod
    def _document_exists(vector_store, doc_hash: str) -> bool:
        """Checks if a document already exists in the vector store based on hash comparison."""
        existing = vector_store.get(where={"content_hash": doc_hash})
        return bool(existing["ids"])

    def _search_top_k(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        pass

    # def get_filtered_retriever(self, metadata_filter: dict):
    #     store = self.get_latest_vector_store()
    #     return store.as_retriever(search_kwargs={"filter": metadata_filter})

    @staticmethod
    def list_documents(vector_store: Chroma) -> dict[str, Any]:
        """
        Lists all documents in the vector store.
        :return: documents
        """
        return vector_store.get(include=["metadatas", "documents"])

    def delete_collection(self, collection_name: str):
        vector_store = self.get_latest_vector_store()
        vector_store._client.delete_collection(collection_name)

    def fetch_table_family(
        self, vectorstore, table_id: str, include_parent: bool = True, row_limit: int = 1000
    ) -> List[Document]:
        coll = getattr(vectorstore, "_collection", None)
        if coll is None:
            # fallback: slow, embedding-based (avoid if possible)
            logger.warning("Collection not found. Using embedding-based search.")
            out = []
            if include_parent:
                out += vectorstore.similarity_search("table", k=1, filter={"table_id": table_id, "type": "table_parent"})
            rows = vectorstore.similarity_search("table rows", k=row_limit, filter={"table_id": table_id, "type": "table_row"})
            rows.sort(key=lambda d: d.metadata.get("row_index", 0))
            return out + rows[:row_limit]

        docs: List[Document] = []
        if include_parent:
            p = self._chroma_get(
                coll,
                {"table_id": table_id, "type": "table_parent"},
                ["documents", "metadatas"], 1
            )
            if p and p.get("documents"):
                docs.append(Document(page_content=p["documents"][0], metadata=p["metadatas"][0]))

        r = self._chroma_get(
            coll, where_dict={"table_id": table_id, "type": "table_row"},
            include=["documents", "metadatas"], limit=row_limit
        )
        pairs = list(zip(r.get("documents", []), r.get("metadatas", [])))
        pairs.sort(key=lambda x: x[1].get("row_index", 0))
        docs.extend(Document(page_content=doc, metadata=meta) for doc, meta in pairs[:row_limit])
        return docs

    @staticmethod
    def _chroma_get(coll, where_dict: dict, include: list, limit: int | None = None):
        """
        Chroma >=0.5 expects one top-level logical operator.
        Build: {"$and": [{"k":{"$eq":v}}, ...]} but try the simple form first for older versions.
        """
        # Try simple form first (older Chroma)
        try:
            return coll.get(where=where_dict, include=include, limit=limit)
        except Exception as e:
            msg = str(e).lower()
            if "exactly one operator" not in msg:
                raise

        # Newer Chroma path
        and_clauses = []
        for k, v in where_dict.items():
            if isinstance(v, dict):  # already has operator
                and_clauses.append({k: v})
            else:
                and_clauses.append({k: {"$eq": v}})
        where_and = {"$and": and_clauses}
        return coll.get(where=where_and, include=include, limit=limit)

    def inspect_tables_for_doc(self, vector_store, doc_id: int, sample: int = 3):
        """
        Inspect tables for a specific document.
        :param vector_store:
        :param doc_id: doc_id from the database (metadata)
        :param sample: only a sample of the tables to show
        :return:
        """
        coll = getattr(vector_store, "_collection", None)
        inc = ["documents", "metadatas"]

        # Count parents & rows
        parents = self._chroma_get(coll, {"doc_id": doc_id, "type": "table_parent"}, include=inc)
        rows = self._chroma_get(coll, {"doc_id": doc_id, "type": "table_row"}, include=inc)

        p_meta = parents.get("metadatas", []) or []
        r_meta = rows.get("metadatas", []) or []
        print(f"[inspect] parents={len(p_meta)} rows={len(r_meta)}")

        # Group rows by table_id and count
        from collections import Counter, defaultdict
        row_by_tid = defaultdict(list)
        for m in r_meta:
            row_by_tid[m.get("table_id")].append(m)
        print("[inspect] top tables by row count:")
        for tid, cnt in Counter({tid: len(v) for tid, v in row_by_tid.items()}).most_common(5):
            print(f"  - {tid}: {cnt} rows")

        # Show columns & a couple of rows from the big table
        if row_by_tid:
            big_tid = max(row_by_tid, key=lambda t: len(row_by_tid[t]))
            print(f"[inspect] biggest table_id={big_tid} n_rows={len(row_by_tid[big_tid])}")
            # fetch its parent (for columns) and a few rows
            p = self._chroma_get(coll, {"table_id": big_tid, "type": "table_parent"}, include=inc, limit=1)
            if p.get("documents"):
                print("[inspect] parent columns:", p["metadatas"][0].get("columns"))
                print("[inspect] parent n_rows:", p["metadatas"][0].get("n_rows"))
            r = self._chroma_get(coll, {"table_id": big_tid, "type": "table_row"}, include=inc, limit=sample)
            for i, (doc, meta) in enumerate(zip(r["documents"], r["metadatas"])):
                print(f"[inspect] row[{i}] row_index={meta.get('row_index')}")
                print(doc.splitlines()[-1])  # last line is the row CSV in your MD
