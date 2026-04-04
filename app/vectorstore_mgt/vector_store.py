import os

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatAnthropic, ChatOpenAI, ChatOllama
from langchain.vectorstores.chroma import Chroma


def load_langchain_model(vector_store, chat_model_name):
    chatbot = ConversationalRetrievalChain.from_llm(
        ChatAnthropic(
            model=chat_model_name
        ),
        retriever=vector_store,
        verbose=False
    )
    return chatbot


def load_llm_model(vectordb: Chroma, chat_model_name: str, model_selection: str):
    """Load the LLM model.
    :param model_selection: either openai or anthropic
    :param vectordb: The vectorstore with the embeddings.
    :param chat_model_name: The chat model.
    :return: The LLM model.
    """
    if model_selection == 'anthropic':
        return ConversationalRetrievalChain.from_llm(
            ChatAnthropic(model=chat_model_name, anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')),
            vectordb.as_retriever(search_kwargs={'k': 6}),
            return_source_documents=False,
            verbose=False
        )
    elif model_selection == 'openai':
        return ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0.9, model_name=chat_model_name),
            vectordb.as_retriever(search_kwargs={'k': 6}),
            return_source_documents=False,
            verbose=False
        )
    elif model_selection == 'ollama':
        return ConversationalRetrievalChain.from_llm(
            ChatOllama(model_name=chat_model_name),
            vectordb.as_retriever(search_kwargs={'k': 6}),
            return_source_documents=False,
            verbose=False
        )
    else:
        return None


def prompt_query(query: str, qa: ConversationalRetrievalChain, chat_history: list):
    """Prompt
    :param query: The query.
    :param qa: The LLM model.
    :param chat_history: The chat history.
    :return: The response."""
    return qa({"question": query, "chat_history": chat_history})
