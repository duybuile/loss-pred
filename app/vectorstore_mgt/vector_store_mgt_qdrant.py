from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client import models
import logging
from langchain_core.documents import Document
from typing import List
# from src.chat_agent import cfg

from src.chat_agent.vector_store_mgt.vector_store_mgt import VectorStoreMgt

logger = logging.getLogger(__name__)


class VectorStoreMgtQdrant(VectorStoreMgt):
    """
    Qdrant-based implementation of VectorStoreMgt.
    Handles storage, deduplication, and reindexing of document vectors with metadata in Qdrant.

    Args:
        vector_store_dir (str): Path used for consistency with the abstract base class. Not used directly by Qdrant.
        embedding_function: The embedding model instance used to convert documents into vectors.
        ollama_url (str, optional): URL for Ollama embedding service if used.
        embedding_model_name (str): Name of the embedding model.
        qdrant_host (str): Hostname of the Qdrant server.
        qdrant_port (int): Port of the Qdrant server.
        collection_name (str): Name of the Qdrant collection to operate on.
    """

    def __init__(
            self,
            vector_store_dir: str,
            embedding_function,
            ollama_url: str = None,
            embedding_model_name: str = "text-embedding-ada-002",
            qdrant_host: str = "localhost",
            qdrant_port: int = 6333,
            collection_name: str = "default_collection"
    ):
        super().__init__(
            vector_store_dir=vector_store_dir,
            embedding_function=embedding_function,
            ollama_url=ollama_url,
            embedding_model_name=embedding_model_name
        )
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name

    def _load_existing_vectors(self, version_dir):
        """Loads the existing vector store from the latest version."""
        return Qdrant(client=self.qdrant_client, collection_name=self.collection_name,
                      embeddings=self.embedding_function)

    def get_latest_vector_store(self):
        """
        Return the most recently active vector store (in Qdrant this is static).

        :return: Qdrant vector store instance.
        """
        return self._load_existing_vectors(self.collection_name)

    def batch_update(self, new_documents: List[Document], reindex_threshold: int = 1000, deduplicate: bool = True):
        """
        Batch update the Qdrant store with new documents.

        :param new_documents: List of new LangChain Document objects.
        :param reindex_threshold: Max number of documents before triggering reindex.
        :param deduplicate: Whether to skip already existing documents by hash.
        """

        vector_store = self.get_latest_vector_store()

        new_texts = []
        metadatas = []

        for doc in new_documents:
            doc_hash = self._hash_document(doc.page_content)
            if not self._document_exists(vector_store, doc_hash):
                new_texts.append(doc.page_content)
                metadata = doc.metadata.copy() if doc.metadata else {}
                metadata["hash"] = doc_hash
                metadatas.append(metadata)

        if not new_texts:
            logger.info("No new documents to add. All documents are duplicates.")
            return

        logger.info(f"Adding {len(new_texts)} new documents to Qdrant.")

        if not self.qdrant_client.collection_exists("default_collection"):
            self.qdrant_client.create_collection(
                collection_name="default_collection",
                vectors_config=models.VectorParams(
                    size=self.embedding_function.client.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE
                )
            )

        vector_store.add_texts(texts=new_texts, metadatas=metadatas)

        total_docs = len(vector_store.similarity_search("", k=reindex_threshold))
        if total_docs > reindex_threshold:
            logger.info("Reindexing required. Rebuilding the vector store.")
            self._reindex()

    def _reindex(self):
        """
        Rebuild the vector store in a new Qdrant collection from existing documents.
        Useful when the document count exceeds a threshold or embeddings need to be refreshed.
        """
        logger.info("Starting reindexing process...")

        all_docs = self.qdrant_client.scroll(collection_name=self.collection_name, with_payload=True)[0]

        texts = []
        metadatas = []
        for point in all_docs:
            doc = point.payload.get("text") or point.payload.get("document")
            if doc:
                texts.append(doc)
                metadatas.append(point.payload)

        reindex_name = f"{self.collection_name}_reindex"
        self.qdrant_client.create_collection(
            collection_name=reindex_name,
            vectors_config=models.VectorParams(
                size=self.embedding_function.vector_size,
                distance=models.Distance.COSINE
            )
        )

        reindex_store = Qdrant(
            client=self.qdrant_client,
            collection_name=reindex_name,
            embeddings=self.embedding_function
        )
        reindex_store.add_texts(texts=texts, metadatas=metadatas)

        logger.info(f"Reindexing complete. New version stored in '{reindex_name}'.")

    # @staticmethod
    # def _document_exists(vector_store, doc_hash: str) -> bool:
    #     """
    #     Check whether a document with a given hash already exists in the store.

    #     :param vector_store: Qdrant vector store instance.
    #     :param doc_hash: SHA-256 hash string of the document content.
    #     :return: True if a similar document exists; False otherwise.
    #     """
    #     try:
    #         results = vector_store.similarity_search(doc_hash, k=1)
    #         return len(results) > 0
    #     except Exception:
    #         return False

    @staticmethod
    def _document_exists(vector_store: "Qdrant", doc_hash: str) -> bool:
        """
        Fast check — does a chunk whose hash == ``doc_hash`` already
        exist in this Qdrant collection?

        Uses a metadata filter (`client.count`) so no embedding or vector search
        is performed.

        Parameters
        ----------
        vector_store : langchain_community.vectorstores.Qdrant
            The wrapper pointing at the target collection.
        doc_hash : str
            The SHA-256 hash of the chunk’s content.

        Returns
        -------
        bool
            True  – at least one stored point has the same hash  
            False – no match or the check failed
        """
        try:
            # LangChain wrapper exposes the underlying client & the collection name
            qclient = getattr(vector_store, "client", None)
            collection = getattr(vector_store, "collection_name", None)
            if qclient is None or collection is None:
                logger.warning("Vector store missing Qdrant client or collection name.")
                return False

            result = qclient.count(
                collection_name=collection,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.hash",  # <- hard-coded
                            match=models.MatchValue(value=doc_hash),
                        )
                    ]
                ),
                exact=True,  # payload-index lookup, fastest path
            )
            exists = result.count > 0
            # logger.debug(                       # ← see the duplicate status here
            #     "Dup-check %s… — %s",
            #     doc_hash[:8],
            #     "duplicate" if exists else "new"
            # )
            return exists

        except Exception as err:
            logger.error(f"Hash lookup failed for {doc_hash[:8]}…: {err}", exc_info=True)
            return False

    def fetch_table_family(
            self, vectorstore, table_id: str, include_parent: bool = True, row_limit: int = 1000
    ) -> List[Document]:
        """
        Fetches the family of a table from the vector store.
        Note: this is not tested
        :param vectorstore:
        :param table_id:
        :param include_parent:
        :param row_limit:
        :return:
        """
        client = getattr(vectorstore, "client", None) or getattr(vectorstore, "_client", None)
        collection = getattr(vectorstore, "collection_name", None)

        try:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue

            def _scroll(key_prefix: str, typ: str, limit: int = 1000):
                # Try root payload first (table_id), then nested metadata.table_id
                flt = Filter(must=[
                    FieldCondition(key=f"{key_prefix}table_id", match=MatchValue(value=table_id)),
                    FieldCondition(key=f"{key_prefix}type", match=MatchValue(value=typ)),
                ])
                points, _ = client.scroll(collection_name=collection, scroll_filter=flt, limit=limit, with_payload=True)
                out = []
                for p in points:
                    payload = p.payload or {}
                    # langchain-qdrant usually stores 'page_content' and 'metadata' in payload
                    text = payload.get("page_content")
                    meta = payload.get("metadata") or {}
                    if text:
                        out.append(Document(page_content=text, metadata=meta))
                return out

            key_prefix = ""  # many setups flatten keys to root
            parents = _scroll(key_prefix, "table_parent", 1) if include_parent else []
            rows = _scroll(key_prefix, "table_row", row_limit)
            if not rows and not parents:
                # try nested 'metadata.' keys
                key_prefix = "metadata."
                parents = _scroll(key_prefix, "table_parent", 1) if include_parent else []
                rows = _scroll(key_prefix, "table_row", row_limit)

            rows.sort(key=lambda d: d.metadata.get("row_index", 0))
            return parents + rows[:row_limit]
        except Exception as ex:
            # Fallback: embedding-based filter (slower)
            logger.error(f"Failed to fetch table family: {ex}")
            out = []
            if include_parent:
                out += vectorstore.similarity_search("table", k=1,
                                                     filter={"table_id": table_id, "type": "table_parent"})
            rows = vectorstore.similarity_search("table rows", k=row_limit,
                                                 filter={"table_id": table_id, "type": "table_row"})
            rows.sort(key=lambda d: d.metadata.get("row_index", 0))
            out += rows[:row_limit]
            return out
