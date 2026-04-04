"""
LLM client — provider-aware factory and synthesis adapter.

Supports Anthropic and OpenAI. Provider is configured in conf/agent.toml.
"""
from __future__ import annotations

import json
import os
import re

import anthropic
import openai
from dotenv import load_dotenv

from utils.llms.llm_client import LLMClient

load_dotenv()


def get_client(provider: str):
    """Return a configured client for the given provider."""
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return anthropic.Anthropic(api_key=api_key)
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        return openai.OpenAI(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Must be 'anthropic' or 'openai'.")


class LLMAdapter:
    """
    Thin adapter for the LLM synthesis call.

    Handles Anthropic and OpenAI API differences so agent.py stays provider-agnostic.
    The synthesis call never uses tool_use — it is a plain completion call that
    returns structured JSON.
    """

    def __init__(self, provider: str, model: str, max_tokens: int):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        api_key = (
            os.environ.get("ANTHROPIC_API_KEY")
            if provider == "anthropic"
            else os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            missing = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
            raise EnvironmentError(f"{missing} is not set.")
        self.llm_client = LLMClient(api=provider, model=model, api_key=api_key)

    def synthesize(self, evidence: dict, system_prompt: str) -> dict:
        """
        Call the LLM with structured evidence and return parsed JSON output.

        Args:
            evidence:      Structured evidence dict from the controller.
            system_prompt: Instructions for the synthesis role.

        Returns:
            Parsed dict with keys: recommendation, summary, key_factors,
            similar_records_summary, review_guidance.
        """
        user_content = json.dumps(evidence, indent=2)

        try:
            raw_text = self.llm_client.call(
                log_usage=True,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
        except (ValueError, AttributeError):
            raise
        except Exception as exc:
            raise RuntimeError(f"LLM API call failed: {exc}") from exc

        return _parse_json_response(raw_text)


def _parse_json_response(text: str) -> dict:
    """Extract and parse a JSON object from LLM output, tolerating markdown fences."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        candidate = match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError(
                f"LLM response contained no JSON object. Raw response: {text!r}"
            )
        candidate = text[start:end]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM response could not be parsed as JSON. Raw response: {text!r}"
        ) from exc
