"""
LLM abstraction supporting Ollama (local) and OpenAI-compatible APIs (Groq, OpenAI, etc.)
Set LLM_PROVIDER=groq and GROQ_API_KEY for cloud deployment.
Set LLM_PROVIDER=ollama (default) for local Ollama.
"""

import os
import json
import httpx

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _active_provider() -> str:
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        return "groq"
    if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        return "openai"
    return "ollama"


async def generate(prompt: str) -> str:
    provider = _active_provider()
    if provider == "groq":
        return await _openai_compat(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            prompt=prompt,
        )
    if provider == "openai":
        return await _openai_compat(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            prompt=prompt,
        )
    return await _ollama(prompt)


async def stream(prompt: str):
    provider = _active_provider()
    if provider in ("groq", "openai"):
        base_url = "https://api.groq.com/openai/v1" if provider == "groq" else OPENAI_BASE_URL
        api_key = GROQ_API_KEY if provider == "groq" else OPENAI_API_KEY
        model = GROQ_MODEL if provider == "groq" else OPENAI_MODEL
        async for chunk in _openai_stream(base_url, api_key, model, prompt):
            yield chunk
    else:
        async for chunk in _ollama_stream(prompt):
            yield chunk


async def _ollama(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


async def _ollama_stream(prompt: str):
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
        ) as resp:
            async for line in resp.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if token := data.get("response"):
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        pass


async def _openai_compat(base_url: str, api_key: str, model: str, prompt: str) -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


async def _openai_stream(base_url: str, api_key: str, model: str, prompt: str):
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                        if content := data["choices"][0]["delta"].get("content"):
                            yield content
                    except (json.JSONDecodeError, KeyError):
                        pass
