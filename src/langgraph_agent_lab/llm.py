"""LLM factory helper.

Provides a simple interface to create LLM clients for use in nodes.
Students should use this helper so the lab works with any supported provider.

Usage in nodes:
    from .llm import get_llm
    llm = get_llm()
    response = llm.invoke("Hello")
"""

from __future__ import annotations

import os


def get_llm(model: str | None = None, temperature: float = 0.0):
    """Create an LLM client from environment configuration.

    Checks for API keys in this order:
    1. GEMINI_API_KEY → ChatGoogleGenerativeAI
    2. OPENAI_API_KEY → ChatOpenAI
    3. ANTHROPIC_API_KEY → ChatAnthropic
    
    Fallback: if LAB_MOCK_LLM=true, return a deterministic mock for testing.

    Override model with the `model` parameter or LLM_MODEL env var.
    """
    # Try to load .env if dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Mock LLM for offline testing / CI without API keys
    if os.getenv("LAB_MOCK_LLM", "false").lower() == "true":
        return _build_mock_llm()

    if os.getenv("GEMINI_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-google-genai") from exc
        return ChatGoogleGenerativeAI(
            model=model or os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )

    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        return ChatOpenAI(
            model=model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-anthropic") from exc
        return ChatAnthropic(
            model=model or os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
            temperature=temperature,
        )

    raise RuntimeError(
        "No LLM API key found. Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env\n"
        "See .env.example for configuration.\n"
        "For offline testing without an API key, set LAB_MOCK_LLM=true"
    )


def _build_mock_llm():
    """Build a deterministic mock LLM for offline testing."""
    from pydantic import BaseModel

    class MockMessage:
        def __init__(self, content: str):
            self.content = content

    class MockStructuredLLM:
        """Mock that returns structured output based on keyword heuristics."""

        def __init__(self, schema):
            self._schema = schema

        def invoke(self, messages: list) -> object:
            # Extract text from user messages only (skip system prompts to avoid keyword contamination)
            text = ""
            for msg in messages:
                role = ""
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = str(msg.get("content", ""))
                elif hasattr(msg, "type"):
                    role = getattr(msg, "type", "human")
                    content = str(getattr(msg, "content", ""))
                else:
                    role = "user"
                    content = str(msg)
                if role not in ("system",):
                    text += " " + content
            text = text.lower()

            # Determine route based on keywords
            if any(w in text for w in ["refund", "delete", "cancel", "send email", "account deletion", "remove", "destructive"]):
                route, risk = "risky", "high"
            elif any(w in text for w in ["lookup", "order", "status", "track", "search", "find", "retrieve"]):
                route, risk = "tool", "low"
            elif any(w in text for w in ["fix it", "can you fix", "it's broken", "vague", "help me", "unclear"]):
                route, risk = "missing_info", "low"
            elif any(w in text for w in ["timeout", "error", "crash", "failure", "unavailable", "system failure"]):
                route, risk = "error", "low"
            else:
                route, risk = "simple", "low"

            # Build result matching schema
            if hasattr(self._schema, "model_fields"):
                return self._schema(route=route, risk_level=risk, reasoning="Mock classification based on keywords")
            return {"route": route, "risk_level": risk, "reasoning": "Mock classification"}

    class MockLLM:
        """Mock LLM for offline/CI use."""

        def with_structured_output(self, schema):
            return MockStructuredLLM(schema)

        def invoke(self, messages: list) -> MockMessage:
            text = ""
            for msg in messages:
                if isinstance(msg, dict):
                    text += " " + str(msg.get("content", ""))
                elif hasattr(msg, "content"):
                    text += " " + str(msg.content)
            return MockMessage(content=f"[Mock response] I can help you with: {text[:100]}")

    return MockLLM()
