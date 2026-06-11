"""Thin Ollama HTTP client for embeddings and chat (no SDK dependency)."""

import requests

import config


def embed(texts: list[str]) -> list[list[float]]:
    """Return embedding vectors for a list of texts using the Ollama embed API."""
    vectors = []
    for text in texts:
        r = requests.post(
            f"{config.OLLAMA_HOST}/api/embeddings",
            json={"model": config.EMBED_MODEL, "prompt": text},
            timeout=120,
        )
        r.raise_for_status()
        vectors.append(r.json()["embedding"])
    return vectors


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def chat(messages: list[dict], stream: bool = False):
    """Call the Ollama chat API.

    If stream=False, returns the full assistant string.
    If stream=True, yields content tokens as they arrive.
    """
    payload = {"model": config.CHAT_MODEL, "messages": messages, "stream": stream}

    if not stream:
        r = requests.post(f"{config.OLLAMA_HOST}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _gen():
        import json
        with requests.post(
            f"{config.OLLAMA_HOST}/api/chat", json=payload, stream=True, timeout=300
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break

    return _gen()
