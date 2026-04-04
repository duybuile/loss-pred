from __future__ import annotations

import logging
import os
import time

import numpy as np
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from app import cfg

load_dotenv()

logger = logging.getLogger(__name__)


def get_required_env_value(env_var_name: str) -> str:
    value = os.environ.get(env_var_name)
    if not value:
        raise EnvironmentError(f"{env_var_name} is not set.")
    return value


def get_model_name(provider: str) -> str:
    return cfg.get(f"embedding.model.{provider}")


class SafeHuggingFaceEmbeddingFunction(embedding_functions.HuggingFaceEmbeddingFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._api_url = (
            "https://router.huggingface.co/hf-inference/models/"
            f"{self.model_name}/pipeline/feature-extraction"
        )
        timeout_seconds = cfg.get("vectorstore.huggingface_timeout_seconds")
        self._session.timeout = timeout_seconds

    def __call__(self, input):  # type: ignore[override]
        request_start = time.perf_counter()
        logger.debug(
            "Requesting Hugging Face embeddings for %d documents from %s",
            len(input),
            self.model_name,
        )
        response = self._session.post(
            self._api_url,
            json={"inputs": input, "options": {"wait_for_model": True}},
        ).json()

        if isinstance(response, dict):
            detail = response.get("error", "Unknown HuggingFace API error.")
            estimated_time = response.get("estimated_time")
            if estimated_time is not None:
                detail = f"{detail} Estimated time: {estimated_time}s."
            raise RuntimeError(
                f"HuggingFace embedding request failed for model "
                f"{self.model_name!r}: {detail}"
            )

        logger.debug(
            "Received Hugging Face embeddings for %d documents from %s in %.2fs",
            len(input),
            self.model_name,
            time.perf_counter() - request_start,
        )
        return [np.array(embedding, dtype=np.float32) for embedding in response]


def get_embedding_fn() -> object:
    provider = cfg.get("vectorstore.embedding_provider").lower()
    api_base = cfg.get("vectorstore.embedding_api_base")

    if provider == "onnx":
        return embedding_functions.ONNXMiniLM_L6_V2()

    if provider == "openai":
        api_key_env_var = cfg.get("vectorstore.openai_api_key_env_var")
        kwargs = {
            "api_key": get_required_env_value(api_key_env_var),
            "model_name": get_model_name("openai"),
        }
        if api_base:
            kwargs["api_base"] = api_base
        return embedding_functions.OpenAIEmbeddingFunction(**kwargs)

    if provider == "huggingface":
        api_key_env_var = cfg.get("vectorstore.huggingface_api_key_env_var")
        return SafeHuggingFaceEmbeddingFunction(
            api_key=get_required_env_value(api_key_env_var),
            model_name=get_model_name("huggingface"),
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")


__all__ = [
    "SafeHuggingFaceEmbeddingFunction",
    "embedding_functions",
    "get_embedding_fn",
    "get_model_name",
    "get_required_env_value",
]
