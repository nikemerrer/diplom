# -*- coding: utf-8 -*-
"""
Минимальный извлекатель полей 1С на базе langextract + Ollama.
— Читаем текст целиком (без собственного чанкинга, библиотека сама разрежет).
— Few-shot примеры берём из utils/examples.py.
— Возвращаем 1С-атрибуты + отладочное сырое содержимое ответа модели, чтобы видеть, что пришло.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import langextract as lx

# Попытаться импортировать провайдер напрямую, иначе через lx.providers
try:
    from langextract.providers.ollama import OllamaLanguageModel
except Exception:
    OllamaLanguageModel = lx.providers.ollama.OllamaLanguageModel  # type: ignore

# ---------------------- ПАРАМЕТРЫ ----------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL_ID = os.getenv("OLLAMA_MODEL_ID", "qwen2.5:7b")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))
TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "32768"))
SEED = int(os.getenv("OLLAMA_SEED", "42"))

MAX_INPUT_CHARS = os.getenv("MAX_INPUT_CHARS")  # пусто -> не обрезаем
MAX_CHAR_BUFFER = os.getenv("MAX_CHAR_BUFFER")  # пусто -> не задаём, иначе используем
MAX_EXAMPLE_CHARS = int(os.getenv("MAX_EXAMPLE_CHARS", "0"))  # 0 = не обрезать примеры

# ------------------ ШАБЛОН ПОЛЕЙ 1С ------------------
EMPTY_PARTY = {"сторона": "", "наименование": "", "подписан": "", "дата": "", "комментарий": ""}
ONEC_TEMPLATE: Dict[str, Any] = {
    "вид_документа": "",
    "заголовок": "",
    "содержание": "",
    "стороны": [EMPTY_PARTY.copy(), EMPTY_PARTY.copy()],
    "комментарий": "",
    "рег_номер": "",
    "дата": "",
    "временный_номер": None,
    "реквизиты": {
        "срок": {"с": None, "по": None},
        "сумма": 0,
        "валюта": None,
        "ндс": None,
    },
    "срок_действия": None,
    "делает_недействующими": False,
    "гриф": None,
    "проект": None,
    "состояние": None,
    "подразделение": None,
    "подготовил": None,
    "ответственный": None,
    "хранение": {"состав": None, "форма": None, "в_дело": None},
    "город": "",
    "объект_наименование": None,
    "объект_адрес": None,
    "часы_охраны": None,
    "периодичность_суммы": None,
}

# ------------------ ПРИМЕРЫ ------------------
def _coerce_examples(raw) -> list:
    out = []
    ExampleData = getattr(lx.data, "ExampleData", None)
    Extraction = getattr(lx.data, "Extraction", None)
    if not ExampleData or not Extraction or not isinstance(raw, list):
        return out

    for ex in raw:
        if hasattr(ex, "text") and hasattr(ex, "extractions"):
            # опционально укорачиваем текст примера, чтобы не раздувать prompt
            if MAX_EXAMPLE_CHARS and isinstance(MAX_EXAMPLE_CHARS, int) and MAX_EXAMPLE_CHARS > 0:
                try:
                    truncated = ex.text[:MAX_EXAMPLE_CHARS]
                    ex = ExampleData(text=truncated, extractions=ex.extractions)
                except Exception:
                    pass
            out.append(ex)
            continue
        if isinstance(ex, dict):
            text = ex.get("text") or ex.get("document") or ex.get("input")
            attrs = ex.get("attrs") or ex.get("attributes")
            title = ex.get("title") or "Документ"
            if isinstance(text, str) and isinstance(attrs, dict):
                if MAX_EXAMPLE_CHARS and isinstance(MAX_EXAMPLE_CHARS, int) and MAX_EXAMPLE_CHARS > 0:
                    text = text[:MAX_EXAMPLE_CHARS]
                out.append(
                    ExampleData(
                        text=text,
                        extractions=[
                            Extraction(
                                extraction_class="onec_fields",
                                extraction_text=title,
                                attributes=attrs,
                            )
                        ],
                    )
                )
    return out


def _load_examples() -> list:
    """Берём примеры из utils/examples.py (режем до 1 шт.), если нет — пустой список."""
    try:
        from .examples import EXAMPLES as USER_EXAMPLES  # type: ignore
        user_ex = _coerce_examples(USER_EXAMPLES)
        if user_ex:
            return user_ex[:1]
    except Exception:
        pass
    return []


# ------------------ ПРОМПТ ------------------
PROMPT_ONEC = """
Ты извлекаешь реквизиты формы 1С «Документооборот». Верни ТОЛЬКО валидный JSON LangExtract:
{"extractions":[{"extraction_class":"onec_fields","extraction_text":"<краткое название/номер>","attributes":{...}}]}

Правила:
- Никакого текста вне JSON, никаких пояснений и Markdown.
- Пустые значения по типу: "" / null / 0. Валюта по умолчанию RUB.
- «подписан» — только ФИО без должности. Даты → YYYY-MM-DD. Суммы → целое число.
- Если данных нет — оставь пустые значения по типу.
- Опирайся на примеры, которые даны.
""".strip()


# ------------------ МОДЕЛЬ ------------------
def _build_model() -> Any:
    try:
        return OllamaLanguageModel(
            model_id=OLLAMA_MODEL_ID,
            model_url=OLLAMA_URL,
            options={"temperature": TEMPERATURE, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX, "seed": SEED},
            timeout=TIMEOUT,
        )
    except TypeError:
        try:
            return OllamaLanguageModel(OLLAMA_MODEL_ID, OLLAMA_URL, options={"temperature": TEMPERATURE, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX, "seed": SEED})
        except Exception:
            return OllamaLanguageModel(OLLAMA_MODEL_ID, OLLAMA_URL)


# ------------------ ОСНОВНАЯ ФУНКЦИЯ ------------------
def extract_onec_fields(text: str) -> Dict[str, Any]:
    # Обрезаем при необходимости (опционально через env)
    if MAX_INPUT_CHARS:
        try:
            limit = int(MAX_INPUT_CHARS)
            if limit > 0 and len(text) > limit:
                text = text[:limit]
        except Exception:
            pass

    examples = _load_examples()
    prompt_parts: List[str] = [PROMPT_ONEC]
    if examples:
        ex = examples[0]
        ex_json = {
            "extractions": [
                {
                    "extraction_class": "onec_fields",
                    "extraction_text": getattr(ex.extractions[0], "extraction_text", "") if ex.extractions else "Документ",
                    "attributes": getattr(ex.extractions[0], "attributes", {}) if ex.extractions else {},
                }
            ]
        }
        prompt_parts.append("\n\nПример:\nТекст:\n" + ex.text + "\nJSON:\n" + json.dumps(ex_json, ensure_ascii=False))
    prompt_parts.append("\n\nДокумент:\n" + text)
    prompt_parts.append("\nОтвет: только JSON.")
    prompt = "\n".join(prompt_parts)

    body = {
        "model": OLLAMA_MODEL_ID,
        "prompt": prompt,
        "stream": False,
        "format": "json",  # просим чистый JSON без Markdown
        "options": {"temperature": TEMPERATURE, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX, "seed": SEED},
    }

    raw_response = None
    try:
        import requests

        resp = requests.post(OLLAMA_URL.rstrip("/") + "/api/generate", json=body, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        raw_response = data.get("response") or data.get("data") or ""
        parsed = _parse_first_json(raw_response)
        if isinstance(parsed, dict):
            exs = parsed.get("extractions") or []
            for ex in exs:
                if isinstance(ex, dict) and ex.get("extraction_class") == "onec_fields" and isinstance(ex.get("attributes"), dict):
                    return {"data": ex["attributes"], "_debug_raw": raw_response}
    except Exception as e:
        return {"_error": str(e), "data": ONEC_TEMPLATE, "_debug_raw": raw_response}

    return {"data": ONEC_TEMPLATE, "_debug_raw": raw_response}


# Совместимость
def extract_entities_from_text(text: str) -> Dict[str, Any]:
    return extract_onec_fields(text)


# ------------------ ВСПОМОГАТЕЛЬНОЕ ------------------
_JSON_RE = None
def _strip_markdown_fences(raw: str) -> str:
    import re

    if not raw:
        return raw
    cleaned = raw.strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned


def _parse_first_json(raw: str) -> Dict[str, Any]:
    global _JSON_RE
    if _JSON_RE is None:
        _JSON_RE = __import__("re").compile(r"\{[\s\S]*\}")
    text = _strip_markdown_fences(raw or "")
    for m in _JSON_RE.finditer(text):
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict) and "extractions" in data:
                return data
        except Exception:
            continue
    return {}
