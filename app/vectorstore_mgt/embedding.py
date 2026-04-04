import os

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, LlamaCppEmbeddings, OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter



def read_docs_files_local(path) -> list:
    """
    Read all documents from a local folder
    :param path:
    :return:
    """
    documents = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
            elif file.endswith('.docx') or file.endswith('.doc'):
                doc_path = os.path.join(root, file)
                loader = Docx2txtLoader(doc_path)
                documents.extend(loader.load())
            elif file.endswith('.txt'):
                text_path = os.path.join(root, file)
                loader = TextLoader(text_path, encoding="utf-8")
                documents.extend(loader.load())
            elif file.endswith('.md'):
                text_path = os.path.join(root, file)
                loader = UnstructuredMarkdownLoader(text_path)
                documents.extend(loader.load())
            else:
                continue
    return documents


def split_documents_into_chunks(docs: list, chunk_size: int = 1000, chunk_overlap: int = 10) -> list:
    """Split the documents into chunks of 1000 characters with an overlap of 10 characters.
    :param docs: A list of documents.
    :param chunk_size: The size of the chunks.
    :param chunk_overlap: The overlap between the chunks.
    :return: A list of chunked documents.
    """
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunk_docs = text_splitter.split_documents(docs)
    return chunk_docs


def create_embeddings(chunked_docs: list, persist_directory: str):
    """Create the embeddings for the chunked documents.
    :param persist_directory:
    :param chunked_docs: A list of chunked documents.
    :return: A vectorstore with the embeddings.
    """
    if cfg["chat_agent"]["embedding"].lower() == 'openai':
        embedding = OpenAIEmbeddings()
    elif cfg["chat_agent"]["embedding"].lower() == 'huggingface':
        embedding = HuggingFaceEmbeddings()
    elif cfg["chat_agent"]["embedding"].lower() == 'llamacpp':
        embedding = LlamaCppEmbeddings()
    elif cfg["chat_agent"]["embedding"].lower() == 'ollama':
        embedding = OllamaEmbeddings(base_url=cfg["chat_agent"]["url"], model=cfg["chat_agent"]["chat_model_name"])
    else:
        embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(chunked_docs, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb


def load_vector_db(persist_directory: str):
    """Load the vectorstore.
    :param persist_directory: The directory where the vector store is stored.
    :return: The vectorstore.
    """
    vectordb = Chroma(persist_directory=persist_directory)
    return vectordb


def prompt_gpt(query: str, qa: ConversationalRetrievalChain, chat_history: list):
    """Prompt the GPT model.
    :param query: The query.
    :param qa: The LLM model.
    :param chat_history: The chat history.
    :return: The response."""
    return qa({"question": query, "chat_history": chat_history})


def naive_split_text(text: str, max_chunk_size: int):
    """Naive text splitter chunks document into chunks of max_chunk_size,
    using paragraphs and sentences as guides."""
    chunks = []

    # remove extraneous whitespace
    text = " ".join(text.split())
    # split into paragraphs
    paragraphs = text.split("\n\n")

    # clean up paragraphs
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 0]

    for paragraph in paragraphs:
        if 0 > len(paragraph) <= max_chunk_size:
            chunks.append(paragraph)
        else:
            sentences = paragraph.split(". ")
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) > max_chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    current_chunk += ". " + sentence

            chunks.append(current_chunk)

    return chunks


def read_chunk_from_file(file: str, chunk_size: int):
    try:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file} not found")

    chunks = naive_split_text(text, chunk_size)

    print(
        f"Splitting text into {len(chunks)} chunks of max size {chunk_size} characters."
    )

    return chunks
