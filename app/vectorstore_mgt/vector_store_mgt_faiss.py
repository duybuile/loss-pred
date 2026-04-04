import logging
import os
import shutil
from datetime import datetime
from typing import List, Tuple, Optional

from codetiming import Timer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.chat_agent.vector_store_mgt.vector_store_mgt import VectorStoreMgt

logger = logging.getLogger(__name__)


class VectorStoreMgtFAISS(VectorStoreMgt):
    HASH_METADATA_KEY = "content_hash"
    DEFAULT_INDEX_NAME = "index"  # Default base name for FAISS files (index.faiss, index.pkl)

    def __init__(self, vector_store_dir: str, embedding_function: str,
                 embedding_model_name: Optional[str] = None, ollama_url: Optional[str] = None,
                 index_name: str = DEFAULT_INDEX_NAME):
        super().__init__(
            vector_store_dir=vector_store_dir,
            embedding_function=embedding_function,
            embedding_model_name=embedding_model_name,
            ollama_url=ollama_url
        )
        self.index_name = index_name
        self._loaded_store: Optional[FAISS] = None
        self._loaded_store_path: Optional[str] = None  # Keep track of the loaded version path
        logger.info(f"Initialized VectorStoreMgtFAISS. Base dir: {vector_store_dir}, Index name: {index_name}")

    def _get_latest_version_dir(self) -> Optional[str]:
        """Finds the path to the latest version directory based on timestamp."""
        try:
            version_dirs = [
                d for d in os.listdir(self.vector_store_dir)
                if os.path.isdir(os.path.join(self.vector_store_dir, d))
            ]
            # Filter for directories that look like our timestamp format (YYYYMMDD_HHMMSS)
            valid_versions = []
            for d in version_dirs:
                try:
                    datetime.strptime(d, "%Y%m%d_%H%M%S")
                    valid_versions.append(d)
                except ValueError:
                    logger.debug(f"Ignoring directory '{d}' as it doesn't match version format.")
                    continue  # Ignore directories not matching the format

            if not valid_versions:
                return None

            # Sort directories chronologically
            latest_version_dir = sorted(valid_versions)[-1]
            return os.path.join(self.vector_store_dir, latest_version_dir)
        except FileNotFoundError:
            logger.warning(f"Vector store base directory not found: {self.vector_store_dir}")
            return None
        except Exception as e:
            logger.error(f"Error finding latest version directory: {e}")
            return None

    def _load_vector_store_from_dir(self, dir_path: str) -> Optional[FAISS]:
        """Loads a FAISS index from the specified directory."""
        if not os.path.exists(dir_path):
            logger.error(f"Directory does not exist: {dir_path}")
            return None

        # Check if index files exist (adjust based on index_name)
        faiss_path = os.path.join(dir_path, f"{self.index_name}.faiss")
        pkl_path = os.path.join(dir_path, f"{self.index_name}.pkl")

        if not os.path.exists(faiss_path) or not os.path.exists(pkl_path):
            logger.warning(
                f"FAISS index files ('{self.index_name}.faiss', '{self.index_name}.pkl') not found in {dir_path}")
            return None

        try:
            # FAISS.load_local requires allow_dangerous_deserialization from langchain 0.1.14 onwards
            allow_dangerous = True  # Set to True if you trust the source of the index files
            logger.info(f"Loading FAISS index from: {dir_path} with index_name='{self.index_name}'")

            # Check LangChain version or FAISS constructor signature if needed for compatibility
            try:
                # Newer Langchain versions (> ~0.1.14) might require this flag
                loaded_store = FAISS.load_local(
                    folder_path=dir_path,
                    embeddings=self.embedding_function,
                    index_name=self.index_name,
                    allow_dangerous_deserialization=allow_dangerous
                )
            except TypeError:
                # Fallback for older versions that don't have the flag
                logger.warning(
                    "Attempting to load FAISS index without 'allow_dangerous_deserialization'. Ensure LangChain "
                    "version compatibility.")
                loaded_store = FAISS.load_local(
                    folder_path=dir_path,
                    embeddings=self.embedding_function,
                    index_name=self.index_name
                )

            logger.info(f"Successfully loaded FAISS index from {dir_path}")
            return loaded_store
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {dir_path}: {e}", exc_info=True)
            # Potentially handle specific exceptions like PickleError, EOFError etc.
            return None

    def get_latest_vector_store(self) -> Optional[FAISS]:
        """
        Finds the latest version directory, loads the FAISS index from it,
        and returns the FAISS vector store instance. Caches the loaded store.
        """
        latest_dir = self._get_latest_version_dir()
        if not latest_dir:
            logger.info("No existing vector store versions found.")
            self._loaded_store = None
            self._loaded_store_path = None
            return None

        # Check if the latest version is already loaded
        if self._loaded_store and self._loaded_store_path == latest_dir:
            logger.info(f"Returning cached latest vector store from: {latest_dir}")
            return self._loaded_store

        logger.info(f"Found latest version directory: {latest_dir}")
        loaded_store = self._load_vector_store_from_dir(latest_dir)

        # Update cache
        self._loaded_store = loaded_store
        self._loaded_store_path = latest_dir if loaded_store else None

        return self._loaded_store

    @staticmethod
    def _get_existing_documents(vector_store: Optional[FAISS]) -> List[Document]:
        """Extracts all documents from the loaded FAISS index's docstore."""
        if not vector_store or not hasattr(vector_store, 'docstore') or not hasattr(vector_store.docstore, '_dict'):
            return []
        # The docstore._dict maps internal IDs to Documents
        return list(vector_store.docstore._dict.values())

    @Timer(name="Time taken for deduplicating documents", text="{name}: {:.2f} seconds", logger=logging.debug)
    def _deduplicate_documents(self, existing_docs: List[Document], new_docs: List[Document]) -> List[Document]:
        """
        Filters new_docs based on content hash, returning only unique documents.
        Adds the content hash to the metadata of the unique new documents.
        """
        if not new_docs:
            return []  # No new documents to process

        # Extract hashes from existing documents (if available in metadata)
        existing_hashes = set()
        for doc in existing_docs:
            if doc.metadata and self.HASH_METADATA_KEY in doc.metadata:
                existing_hashes.add(doc.metadata[self.HASH_METADATA_KEY])
            else:
                # Fallback: If hash is missing, calculate and use it for comparison,
                # but log a warning as ideally it should be persisted.
                content_hash = self._hash_document(doc.page_content)
                existing_hashes.add(content_hash)
                logger.warning(
                    f"Existing document (source: {doc.metadata.get('source', 'N/A')}) missing '{self.HASH_METADATA_KEY}'. Calculated on the fly.")

        unique_new_docs = []
        processed_new_hashes = set()  # Keep track of hashes we've already added from the *new* batch

        for doc in new_docs:
            content_hash = self._hash_document(doc.page_content)

            # Add hash to the document's metadata *before* checking uniqueness.
            # This ensures it gets saved if the document is added.
            if doc.metadata is None:  # Ensure metadata dict exists
                doc.metadata = {}
            doc.metadata[self.HASH_METADATA_KEY] = content_hash

            # Check if hash is neither in existing docs nor already added from this new batch
            if content_hash not in existing_hashes and content_hash not in processed_new_hashes:
                unique_new_docs.append(doc)
                processed_new_hashes.add(content_hash)  # Mark this hash as added

        logger.info(
            f"Deduplication (hash-based): Found {len(unique_new_docs)} unique documents out of {len(new_docs)} new "
            f"documents.")
        return unique_new_docs

    def _create_new_version_dir(self) -> str:
        """Creates a new timestamped directory for the next version."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_dir_path = os.path.join(self.vector_store_dir, timestamp)
        os.makedirs(new_dir_path)
        logger.info(f"Created new version directory: {new_dir_path}")
        return new_dir_path

    @Timer(name="Time taken for batch_update", text="{name}: {:.2f} seconds", logger=logging.debug)
    def batch_update(self, new_documents: List[Document], reindex_threshold: int = 1000, deduplicate: bool = True
                     ) -> Tuple[Optional[FAISS], str]:
        """
        Performs a batch update with versioning and deduplication.

        1. Loads the latest existing vector store (if any).
        2. Extracts existing documents.
        3. Deduplicates the `new_documents` against existing ones based on `page_content`.
        4. If no unique new documents are found, returns the current store and its path.
        5. If unique documents exist, creates a new version directory.
        6. Creates a *new* FAISS index containing *all* documents (old + unique new).
        7. Saves the new index to the new version directory.
        8. Updates the internal cache (`_loaded_store`) to point to the new version.

        Args:
        :param new_documents:
        :param reindex_threshold:
        :param deduplicate:

        Returns:
        - Tuple[Optional[FAISS], str]: The newly created FAISS instance and its directory path.
                                       Returns (None, old_path) if no update was needed.
        """
        if not new_documents:
            logger.warning("batch_update called with an empty list of new documents.")
            latest_store = self.get_latest_vector_store()
            latest_path = self._loaded_store_path or ""
            return latest_store, latest_path

        logger.info("Starting batch update (incremental with versioning)...")

        # --- Find latest version and deduplicate ---
        latest_version_dir = self._get_latest_version_dir()
        current_store = None
        existing_documents = []

        if latest_version_dir:
            logger.info(f"Loading latest version from: {latest_version_dir}")
            current_store = self._load_vector_store_from_dir(latest_version_dir)
            if current_store:
                existing_documents = self._get_existing_documents(current_store)
                logger.info(f"Loaded {len(existing_documents)} existing documents.")
            else:
                logger.warning(
                    f"Failed to load index from latest directory {latest_version_dir}. Consider rebuilding if this "
                    f"persists.")
                # Optional: Could add logic here to fall back to a full rebuild if loading fails.
                # For now, we proceed assuming it's either the first run or loading worked.

        if deduplicate:
            unique_new_docs = self._deduplicate_documents(existing_documents, new_documents)
        else:
            unique_new_docs = new_documents

        # --- Check if update is needed ---
        if not unique_new_docs:
            logger.info("No unique new documents found. Batch update aborted.")
            # Return the latest loaded store (or None if none exists/loaded) and its path
            return current_store, latest_version_dir or ""

        # --- Perform Update (Create new version dir first) ---
        new_version_dir = self._create_new_version_dir()
        # new_vector_store = None

        try:
            if current_store and latest_version_dir:
                # --- Incremental Update Case ---
                logger.info(f"Performing incremental update. Adding {len(unique_new_docs)} documents.")

                # 1. Copy index files from latest version to new version directory
                latest_faiss_path = os.path.join(latest_version_dir, f"{self.index_name}.faiss")
                latest_pkl_path = os.path.join(latest_version_dir, f"{self.index_name}.pkl")
                new_faiss_path = os.path.join(new_version_dir, f"{self.index_name}.faiss")
                new_pkl_path = os.path.join(new_version_dir, f"{self.index_name}.pkl")

                if os.path.exists(latest_faiss_path) and os.path.exists(latest_pkl_path):
                    logger.info(f"Copying index files from {latest_version_dir} to {new_version_dir}")
                    shutil.copy2(latest_faiss_path, new_faiss_path)  # copy2 preserves metadata
                    shutil.copy2(latest_pkl_path, new_pkl_path)
                else:
                    # This shouldn't happen if current_store was loaded successfully, but handle defensively
                    raise FileNotFoundError(
                        f"Source index files not found in {latest_version_dir} despite loading store.")

                # 2. Load the *copied* index from the *new* directory
                # Use the same embedding function instance used for initialization
                logger.info(f"Loading copied index from new directory: {new_version_dir}")
                store_to_update = self._load_vector_store_from_dir(new_version_dir)
                if not store_to_update:
                    raise RuntimeError(f"Failed to load the copied index from {new_version_dir}")

                # 3. Add new documents (embedding happens here)
                logger.info(f"Adding {len(unique_new_docs)} documents to the index...")
                # FAISS.add_documents handles embedding internally
                added_ids = store_to_update.add_documents(unique_new_docs)
                logger.info(f"Successfully added {len(added_ids)} new document vectors.")

                # 4. Save the modified index back to the new directory
                logger.info(f"Saving updated index to: {new_version_dir}")
                store_to_update.save_local(folder_path=new_version_dir, index_name=self.index_name)
                new_vector_store = store_to_update

            else:
                # --- First Run Case ---
                logger.info(
                    f"First run or no loadable previous version. Creating index from scratch with {len(unique_new_docs)} documents.")
                # Embed and create index only with the new documents
                new_vector_store = FAISS.from_documents(
                    documents=unique_new_docs,  # Only unique new docs needed
                    embedding=self.embedding_function
                )
                logger.info("Successfully created new FAISS index in memory.")
                # Save the new index to the new directory
                logger.info(f"Saving new index to: {new_version_dir}")
                new_vector_store.save_local(folder_path=new_version_dir, index_name=self.index_name)

            # --- Update Cache ---
            logger.info(f"Update successful. New version path: {new_version_dir}")
            self._loaded_store = new_vector_store
            self._loaded_store_path = new_version_dir
            return new_vector_store, new_version_dir

        except Exception as e:
            logger.error(f"Error during batch update process: {e}", exc_info=True)
            # Cleanup: Remove the potentially incomplete new version directory
            if os.path.exists(new_version_dir):
                try:
                    shutil.rmtree(new_version_dir)
                    logger.info(f"Cleaned up failed version directory: {new_version_dir}")
                except Exception as cleanup_e:
                    logger.error(f"Error cleaning up directory {new_version_dir}: {cleanup_e}")
            # Attempt to revert cache to previous state if possible
            self._loaded_store = current_store
            self._loaded_store_path = latest_version_dir
            # Re-raise or handle the error appropriately
            raise RuntimeError(f"Batch update failed: {e}") from e

    def fetch_table_family(
            self, vectorstore, table_id: str, include_parent: bool = True, row_limit: int = 1000
    ) -> List[Document]:
        # LangChain FAISS keeps a docstore mapping id -> Document
        store = getattr(vectorstore, "docstore", None)
        if store is None:
            return []

        # Try standard access
        docs_iter = []
        if hasattr(store, "_dict"):  # InMemoryDocstore
            docs_iter = store._dict.values()
        elif hasattr(store, "search"):  # Docstore API
            # no native metadata query; must scan ids
            ids = list(getattr(vectorstore, "index_to_docstore_id", {}).values())
            docs_iter = (store.search(i) for i in ids)

        parents, rows = [], []
        for d in docs_iter:
            if not isinstance(d, Document):
                continue
            m = d.metadata or {}
            if m.get("table_id") != table_id:
                continue
            t = m.get("type")
            if t == "table_parent" and include_parent and not parents:
                parents.append(d)
            elif t == "table_row":
                rows.append(d)

        rows.sort(key=lambda d: d.metadata.get("row_index", 0))
        return parents + rows[:row_limit]
