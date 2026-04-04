from langchain_core.documents import Document

from src.chat_agent.vector_store_mgt.vector_store_mgt import VectorStoreMgt


class VectorStoreMgtLlamaIndex(VectorStoreMgt):
    def get_latest_vector_store(self):
        """Returns the vector store."""
        pass

    def batch_update(self, new_documents: list[Document], reindex_threshold: int = 1000, deduplicate: bool = True):
        """Perform batch update of documents with versioning and deduplication.
        """
        pass

    def _load_existing_vectors(self, version_dir):
        """Loads the existing vector store from the latest version."""
        pass

    def _reindex(self):
        """Rebuilds the entire vector store by re-indexing all documents."""
        pass
