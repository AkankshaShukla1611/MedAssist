"""
Thin client for the local LLM, served through Ollama.
Swappable at deploy time purely via env vars — no code change needed to go
from Mistral to Gemma to whatever else `ollama pull <name>` gives you:

    LLM_MODEL=mistral
    LLM_MODEL=gemma2
    LLM_MODEL=llama3

This module never talks to any external/cloud API — everything stays local,
which matters for clinical data privacy.
"""
import httpx
from fastapi import HTTPException

from app.core.config import settings


async def generate(prompt: str) -> str:
    url = f"{settings.LLM_BASE_URL}/api/generate"
    payload = {
        "model": settings.LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},  # low temperature — this is clinical, not creative
    }

    try:
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM did not respond in time. Please try again.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"LLM backend error: {e.response.status_code}")
    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Could not reach the local LLM service. Is Ollama running?",
        )
