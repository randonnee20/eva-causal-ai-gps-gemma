"""llm.py — 로컬 Ollama(gemma3:4b) 우선, Gemini API fallback

동작 방식:
  로컬 PC : Ollama(http://localhost:11434)가 살아있으면 gemma3:4b 사용
  Cloud   : Ollama 없음 → st.secrets["GEMINI_API_KEY"]로 Gemini API 사용

환경변수(선택):
  OLLAMA_URL    : 기본 http://localhost:11434
  OLLAMA_MODEL  : 기본 gemma3:4b
  GEMINI_API_KEY: Cloud 배포 시 (st.secrets 우선)
  DAILY_LIMIT   : Gemini 일일 한도 (기본 20)
"""

import os
import json
from datetime import date
from pathlib import Path

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
GEMINI_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
USAGE_FILE = Path("data/usage.json")

_mode = None  # "ollama" | "gemini" | "none"


def _daily_limit():
    return int(os.environ.get("DAILY_LIMIT", "200"))


# ---------------------------------------------------------------------------
# 모드 감지
# ---------------------------------------------------------------------------
def _ollama_alive():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def _get_gemini_key():
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "").strip()


def detect_mode(force=False):
    global _mode
    if _mode is not None and not force:
        return _mode
    if _ollama_alive():
        _mode = "ollama"
    elif _get_gemini_key():
        _mode = "gemini"
    else:
        _mode = "none"
    return _mode


def status_label():
    return {
        "ollama": f"로컬 {OLLAMA_MODEL}",
        "gemini": "Gemini API",
        "none": "LLM 미연결",
    }[detect_mode()]


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
def _gen_ollama(prompt, system, temperature, max_tokens):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip()


# ---------------------------------------------------------------------------
# Gemini (일일 사용량 추적)
# ---------------------------------------------------------------------------
def _check_and_increment():
    limit = _daily_limit()
    if limit >= 9999:
        return
    today = str(date.today())
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    usage = {"date": today, "count": 0}
    if USAGE_FILE.exists():
        try:
            usage = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    if usage.get("date") != today:
        usage = {"date": today, "count": 0}
    if usage["count"] >= limit:
        raise RuntimeError(f"오늘 AI 해석 한도({limit}회)를 모두 사용했습니다. 내일 다시 이용해 주세요.")
    usage["count"] += 1
    USAGE_FILE.write_text(json.dumps(usage, ensure_ascii=False), encoding="utf-8")


def _gen_gemini(prompt, system, temperature, max_tokens):
    from google import genai
    from google.genai import types
    _check_and_increment()
    client = genai.Client(api_key=_get_gemini_key())
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system or None,
    )
    last_err = None
    for model_name in GEMINI_MODELS:
        try:
            resp = client.models.generate_content(model=model_name, contents=prompt, config=cfg)
            return (resp.text or "").strip()
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Gemini 호출 실패: {last_err}")


# ---------------------------------------------------------------------------
# 공개 인터페이스
# ---------------------------------------------------------------------------
def generate(prompt, system="", temperature=0.4, max_tokens=300):
    mode = detect_mode()
    if mode == "ollama":
        return _gen_ollama(prompt, system, temperature, max_tokens)
    if mode == "gemini":
        return _gen_gemini(prompt, system, temperature, max_tokens)
    raise RuntimeError("LLM 사용 불가: Ollama 미실행이고 Gemini 키도 없습니다.")


def usage_info():
    """사이드바 표시용 — Gemini 모드일 때만 카운트 의미 있음"""
    today = str(date.today())
    limit = _daily_limit()
    if detect_mode() != "gemini":
        return {"mode": status_label(), "count": 0, "limit": limit}
    count = 0
    if USAGE_FILE.exists():
        try:
            u = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
            if u.get("date") == today:
                count = u.get("count", 0)
        except Exception:
            pass
    return {"mode": status_label(), "count": count, "limit": limit}
