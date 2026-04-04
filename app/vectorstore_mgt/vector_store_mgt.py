import hashlib
import logging
import os
import re
import shutil
from abc import ABC, abstractmethod
from typing import Any, List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class VectorStoreMgt(ABC):
    """
    Class for managing vector stores.
    >>> documents = ["This is a document", "This is another document"]
    >>> vsm = VectorStoreMgt(
    ...     vector_store_dir='data', embedding_function='ollama',
    ...     ollama_url='https://api.zep.ai', embedding_model_name='llama-7b')
    >>> vsm.batch_update(documents)
    """

    def __init__(
        self,
        vector_store_dir: str,
        embedding_function: str,
        ollama_url: str = None,
        embedding_model_name: str = None,
        embedding: Any | None = None,
    ):
        """
        Initialize the Vector Store Manager.

        Parameters:
        - base_dir (str): Base directory where all versions of vector vaults will be stored.
        - embedding_function: The function to embed documents (e.g., using an embedding model from LangChain).
        - ollama_url: The base URL for the offline embedding model with Ollama.
        - embedding_model_name: The name of the embedding model.
        """
        self.vector_store_dir = vector_store_dir
        if embedding is not None:
            self.embedding_function = embedding
        else:
            self.embedding_function = self._get_embedding_function(
                embedding_function=embedding_function,
                base_url=ollama_url,
                embedding_model_name=embedding_model_name,
            )
        if not os.path.exists(vector_store_dir):
            os.makedirs(vector_store_dir)

    @staticmethod
    def _get_embedding_function(embedding_function: str, base_url: str = None, embedding_model_name: str = None):
        """Returns the embedding function."""
        if embedding_function.lower() == 'openai':
            embedding = OpenAIEmbeddings(
                model=embedding_model_name,
                openai_api_key=os.getenv("OPEN_AI_KEY_CHAT_FASANARA"),
            )
        elif embedding_function.lower() == 'huggingface':
            embedding = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={"device": "cpu"}
            )
        elif embedding_function.lower() == 'llamacpp':
            embedding = LlamaCppEmbeddings()
        elif embedding_function.lower() == 'ollama':
            embedding = OllamaEmbeddings(
                base_url=base_url,
                model=embedding_model_name,
            )
        else:
            raise ValueError(f"Unsupported embedding function: {embedding_function}")
        return embedding

    def _get_latest_version(self):
        """Returns the latest version of the vector store based on directory structure."""
        versions = [d for d in os.listdir(self.vector_store_dir) if
                    os.path.isdir(os.path.join(self.vector_store_dir, d))]
        if not versions:
            return None, None
        latest_version = sorted(versions, key=lambda v: int(v.split("_v")[1]))[-1]
        return latest_version, os.path.join(self.vector_store_dir, latest_version)

    def _create_new_version(self):
        """Creates a new version folder for batch update."""
        latest_version, _ = self._get_latest_version()
        if latest_version is None:
            new_version_num = 1
        else:
            new_version_num = int(latest_version.split("_")[1]) + 1

        new_version_dir = os.path.join(self.vector_store_dir, f"vault_v{new_version_num}")
        os.makedirs(new_version_dir)
        return new_version_dir

    def _hash_document(self, document: str) -> str:
        """Hashes a document using SHA-256 to create a unique identifier."""
        # Process document before hashing
        document = self._normalise_doc_before_hashing(document)
        return hashlib.sha256(document.encode('utf-8')).hexdigest()

    @staticmethod
    def _normalise_doc_before_hashing(document: str) -> str:
        # Lowercase the text
        document = document.lower()
        # Replace multiple spaces/tabs/newlines with a single space
        document = re.sub(r'\s+', ' ', document)
        # Remove leading/trailing whitespace
        document = document.strip()
        # Optionally: remove punctuation (comment out if you want to keep it)
        document = re.sub(r'[^\w\s]', '', document)
        return document

    @staticmethod
    def move_uploaded_files_to_archive(upload_dir, archive_dir):
        """Implement this method to move uploaded files to an archive directory using shutil"""
        # Ensure destination folder exists
        os.makedirs(archive_dir, exist_ok=True)
        # Loop through all files in the source directory
        for filename in os.listdir(upload_dir):
            src_file = os.path.join(upload_dir, filename)
            dst_file = os.path.join(archive_dir, filename)

            # Check if it's a file before moving
            if os.path.isfile(src_file):
                shutil.move(src_file, dst_file)
                print(f"Moved: {src_file} -> {dst_file}")

    @staticmethod
    def _make_table_uid(d: Document) -> str:
        """Stable ID for upsert/dedup in the vector store."""
        t = d.metadata.get("type")
        doc_id = d.metadata["doc_id"]
        table_id = d.metadata["table_id"]
        if t == "table_parent":
            return f"{doc_id}::tbl::{table_id}::parent"
        elif t == "table_row":
            idx = d.metadata.get("row_index", 0)
            return f"{doc_id}::tbl::{table_id}::row::{idx}"
        # fallback, shouldn't happen for table docs
        return f"{doc_id}::tbl::{table_id}::misc"

    def ingest_tables_only(
            self,
            chunked_docs: List[Document],
            doc_id: str,  # your application-level doc_id
    ) -> List[Document]:
        """
        Load/parse, then push ONLY table docs (rows + parents) into the existing vector store.
        Returns the list of table docs ingested.
        """
        TABLE_TYPES = {"table_row", "table_parent"}

        # Keep only table docs
        table_docs = [d for d in chunked_docs if d.metadata.get("type") in TABLE_TYPES]
        if not table_docs:
            logger.info("[tables-only] No tables found.")
            return []

        # Stamp your external doc_id and stable per-row/parent IDs for upsert
        with_ids = []
        ids = []
        for d in table_docs:
            d = Document(page_content=d.page_content, metadata={**d.metadata, "doc_id": doc_id})
            with_ids.append(d)
            ids.append(self._make_table_uid(d))

        # Prefer to upsert by IDs to avoid duplicates on re-ingest runs.
        # If your vector_store_manager.batch_update accepts 'ids', pass them.
        # Otherwise, ensure batch_update uses d.metadata["uid"] or similar; we set it here too.
        for d, uid in zip(with_ids, ids):
            d.metadata["uid"] = uid  # in case your manager reads metadata to get the ID

        logger.info(f"[tables-only] Ingesting {len(with_ids)} table docs (rows + parents)")
        try:
            # If your manager supports explicit IDs:
            # vector_store_manager.add_texts(
            #     texts=[d.page_content for d in with_ids],
            #     metadatas=[d.metadata for d in with_ids],
            #     ids=ids
            # )

            # Otherwise, keep your existing call:
            self.batch_update(with_ids)
        except Exception as e:
            logger.exception(f"[tables-only] Failed to ingest tables: {e}")
            raise

        return with_ids

    @abstractmethod
    def get_latest_vector_store(self):
        """Returns the vector store."""
        raise NotImplementedError

    @abstractmethod
    def batch_update(self, new_documents: list[Document], reindex_threshold: int = 1000, deduplicate: bool = True):
        """
        Perform batch update of documents with versioning and deduplication.

        Parameters:
        - new_documents (list of str): The new documents to be added.
        - reindex_threshold (int): The threshold to trigger reindexing based on the number of total documents.
        - deduplicate (bool): Whether to deduplicate documents based on hash comparison.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_table_family(
            self,
            vectorstore,  # the *live* vectorstore instance used by this query
            table_id: str,
            include_parent: bool = True,
            row_limit: int = 1000,
    ) -> List[Document]:
        """Return [parent? + ordered rows] for a table_id using the store's fastest metadata path."""
        raise NotImplementedError

    @abstractmethod
    def inspect_tables_for_doc(self, vector_store, doc_id: int, sample: int = 3):
        raise NotImplementedError

    def _load_existing_vectors(self, version_dir):
        raise NotImplementedError

    def _reindex(self):
        raise NotImplementedError
