"""Optional LLM access. The app runs without a key (deterministic mode); the LLM is only
used when GROQ_API_KEY is set and the groq package is installed."""
import json, os, re
from typing import Optional

MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
_client = None
_tried = False

def get_client():
    global _client, _tried
    if _tried:
        return _client
    _tried = True
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        return None
    try:
        from groq import Groq
        _client = Groq(api_key=key)
    except Exception as e:   # package missing or bad key format
        print(f"[LLM] disabled: {e}")
        _client = None
    return _client

def llm_available() -> bool:
    return get_client() is not None

def chat_json(system: str, user: str, max_tokens: int = 1500) -> Optional[dict]:
    """Return parsed JSON from the LLM, or None on any failure (caller falls back)."""
    c = get_client()
    if c is None:
        return None
    try:
        r = c.chat.completions.create(model=MODEL, temperature=0.0, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        raw = r.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[LLM] call failed: {e}")
        return None
