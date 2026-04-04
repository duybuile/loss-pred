"""
This script contains different functions for collection management.
"""
from langchain.vectorstores import Chroma


def rename_chroma_collection(persist_dir, embedding_function, old_name=None, new_name=None):
    """
    Rename a Chroma collection.
    :param old_name: The old name of the collection.
    :param new_name: The new name of the collection.
    :param persist_dir: The directory where the collection is stored.
    :param embedding_function: The embedding function used to create the vectors.
    :return: None
    """
    # Load old collection
    if old_name is None:
        old_collection = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_function,
        )
    else:
        old_collection = Chroma(
            collection_name=old_name,
            persist_directory=persist_dir,
            embedding_function=embedding_function,
        )

    # Fetch all data
    docs = old_collection.get(include=["metadatas", "documents", "embeddings"])

    # Create new collection
    if new_name is None:
        new_collection = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_function,
        )
    else:
        new_collection = Chroma(
            collection_name=new_name,
            persist_directory=persist_dir,
            embedding_function=embedding_function,
        )

    # Add all docs to new collection
    new_collection.add_documents(
        ids=docs["ids"],
        documents=docs["documents"],
        metadatas=docs["metadatas"],
        embeddings=docs["embeddings"],
    )

    # Optional: remove old collection data (e.g., delete directory manually if needed)
    print(f"Copied {len(docs['ids'])} records from {old_name} to {new_name}")


if __name__ == '__main__':
    rename_chroma_collection(
        persist_dir="data/vector_store_chroma",
        embedding_function="openai",
        old_name=None,
        new_name="prod_openai",
    )
