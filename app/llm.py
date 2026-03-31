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
        self._client = get_client(provider)

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

        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            raw_text = response.content[0].text
        else:  # openai
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            raw_text = response.choices[0].message.content

        return _parse_json_response(raw_text)


def _parse_json_response(text: str) -> dict:
    """Extract and parse a JSON object from LLM output, tolerating markdown fences."""
    # Strip markdown code fences if present
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        # Find the outermost JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
    return json.loads(text)
