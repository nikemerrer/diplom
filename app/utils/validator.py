# app/utils/validator.py
# -*- coding: utf-8 -*-
import os
from typing import Any, Dict, Tuple, List
from jsonschema import Draft7Validator

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["doc_type", "title", "number", "date", "parties", "amount", "currency", "subject"],
    "properties": {
        "doc_type": {"type": "string"},
        "title": {"type": "string"},
        "number": {"type": "string"},
        "date": {"type": "string"},  # ISO YYYY-MM-DD или пустая строка
        "parties": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {
                "type": "object",
                "required": ["role", "name"],
                "properties": {
                    "role": {"type": "string"},
                    "name": {"type": "string"}
                }
            }
        },
        "amount": {"type": "integer", "minimum": 0},
        "currency": {"type": "string"},
        "subject": {"type": "string"}
    },
    "additionalProperties": False,
}

USE_LLM_VALIDATOR = os.getenv("USE_LLM_VALIDATOR", "0") == "1"

def validate_against_schema(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    v = Draft7Validator(SCHEMA)
    errors = [f"{e.path}: {e.message}" for e in v.iter_errors(data)]
    return (len(errors) == 0, errors)

def llm_check_or_repair(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Заготовка. По умолчанию просто возвращает вход.
    Позже можно подключить Ollama и попросить LLM:
    - проверить соответствие схеме
    - при необходимости подправить формат (например, дату -> ISO)
    """
    if not USE_LLM_VALIDATOR:
        return data

    # TODO: реализовать вызов через langextract.create_model(...).generate(...)
    # и вернуть "исправленный" JSON.
    return data

def validate_or_repair(data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, List[str]]:
    ok, errs = validate_against_schema(data)
    if ok:
        return data, True, []

    # Пробуем LLM-ремонт (если включён)
    repaired = llm_check_or_repair(data)
    ok2, errs2 = validate_against_schema(repaired)
    if ok2:
        return repaired, True, []

    # Возвращаем исходник с ошибками
    return data, False, (errs if errs else errs2)
