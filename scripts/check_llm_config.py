"""Sanity-check the configured LLM provider/model without leaking secrets.

Usage:
  .venv/bin/python -m scripts.check_llm_config
  .venv/bin/python -m scripts.check_llm_config --ping
"""

from __future__ import annotations

import argparse
import os

from core.config import get_config_value, get_default_model, get_timeout_seconds
from core.llm_factory import get_sync_llm_client


def _key_name_for_provider(provider: str) -> str | None:
    provider = provider.lower()
    if provider.startswith("openai"):
        return "OPENAI_API_KEY"
    if provider == "claude":
        return "ANTHROPIC_API_KEY"
    if provider == "gemini":
        return "GOOGLE_API_KEY"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ping",
        action="store_true",
        help="Make a live LLM call (requires network + valid API key).",
    )
    args = parser.parse_args()

    provider = (get_config_value("LLM_PROVIDER", "openai") or "openai").lower()
    model = get_default_model()
    timeout = get_timeout_seconds()

    print(f"LLM_PROVIDER={provider}")
    print(f"LLM_MODEL={model}")
    print(f"LLM_TIMEOUT_SECONDS={timeout}")

    max_out = get_config_value("LLM_MAX_OUTPUT_TOKENS") or get_config_value("GEMINI_MAX_TOKENS")
    if max_out:
        print(f"LLM_MAX_OUTPUT_TOKENS={max_out}")

    key_name = _key_name_for_provider(provider)
    if key_name:
        key_in_config = bool(get_config_value(key_name))
        key_in_env = bool(os.getenv(key_name))
        print(f"{key_name}: configured={key_in_config} env_set={key_in_env}")
    else:
        print("API key: unknown provider mapping")

    llm = get_sync_llm_client(provider=provider)
    print(f"LLM client: {llm.__class__.__name__}")
    effective_max = getattr(llm, "_max_tokens", None)
    if isinstance(effective_max, int):
        print(f"effective_max_output_tokens={effective_max}")

    if args.ping:
        resp = llm.chat(
            messages=[
                {"role": "system", "content": "Reply with a single word."},
                {"role": "user", "content": "ping"},
            ],
            model=model,
            temperature=1.0,
        )
        print("Ping response preview:", (resp or "").strip()[:100])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
