# -*- coding: utf-8 -*-
"""
Извлечение сущностей для 1С:Документооборот на базе LangExtract + Ollama.

Результат:
- onec_fields: только то, что реально пишется в 1С
- extra_findings: дополнительные сущности для вывода в сообщениях
- _meta: служебная информация

Важно:
- комментарий НЕ извлекается
- examples реально ограничиваются через MAX_EXAMPLES и MAX_EXAMPLE_CHARS
- входной текст ограничивается через MAX_INPUT_CHARS
"""
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import langextract as lx
except Exception as exc:
    raise RuntimeError(
        "Не установлен langextract. Добавь зависимость 'langextract' в requirements.txt "
        "и пересобери контейнер."
    ) from exc


# =========================================================
# КОНФИГ
# =========================================================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MODEL_ID = os.getenv("OLLAMA_MODEL_ID") or os.getenv("MODEL_ID") or "qwen2.5:1.5b"

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "900"))
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

EXTRACTION_PASSES = int(os.getenv("LANGEXTRACT_PASSES", "1"))
MAX_WORKERS = int(os.getenv("LANGEXTRACT_MAX_WORKERS", "1"))

MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "2500"))
MAX_CHAR_BUFFER = int(os.getenv("LANGEXTRACT_MAX_CHAR_BUFFER", "2500"))
MAX_EXAMPLE_CHARS = int(os.getenv("MAX_EXAMPLE_CHARS", "450"))
MAX_EXAMPLES = max(int(os.getenv("MAX_EXAMPLES", "1")), 0)

SAVE_LANGEXTRACT_ARTIFACTS = os.getenv("SAVE_LANGEXTRACT_ARTIFACTS", "0").lower() in {"1", "true", "yes"}
LANGEXTRACT_ARTIFACT_DIR = os.getenv("LANGEXTRACT_ARTIFACT_DIR", "langextract_artifacts")


PROMPT_DESCRIPTION = """
Извлекай сущности из русского делового документа в порядке появления в тексте.

Главные правила:
1. extraction_text должен быть дословным фрагментом исходного документа.
2. Не выдумывай сущности, которых нет в тексте.
3. Извлекай только реально встречающиеся значения.
4. Чётко различай:
   - customer_organization = организация-заказчик
   - executor_organization = организация-исполнитель
   - department = подразделение, только если оно явно указано в документе
5. person_full_name извлекай только для реальных физических лиц, явно упомянутых в документе.
6. Если ФИО в косвенном падеже, в normalized_full_name приводи к форме, пригодной для поиска в 1С.
7. Если заголовок документа разбит на несколько строк, в canonical_title собери его в одну строку.
8. Извлекай:
   - сущности для автозаполнения 1С
   - дополнительные полезные сущности для вывода пользователю
9. Не извлекай пустые сущности.
""".strip()


# =========================================================
# ПУБЛИЧНАЯ ФУНКЦИЯ
# =========================================================
def extract_onec_fields(text: str) -> Dict[str, Any]:
    prepared_text = _prepare_input_text(text)
    if not prepared_text:
        raise RuntimeError("На вход извлекателя пришёл пустой текст документа")

    examples = _load_examples()
    annotated = _run_langextract(prepared_text, examples)
    normalized_entities = _normalize_annotated_document(annotated)
    result = _assemble_result(normalized_entities)

    if SAVE_LANGEXTRACT_ARTIFACTS:
        _save_artifacts(annotated)

    return result


# совместимость со старым названием
extract_entities_from_text = extract_onec_fields


# =========================================================
# ПОДГОТОВКА ВХОДНОГО ТЕКСТА
# =========================================================
def _prepare_input_text(text: str) -> str:
    cleaned = (text or "").replace("\x00", " ").strip()
    if MAX_INPUT_CHARS > 0 and len(cleaned) > MAX_INPUT_CHARS:
        return cleaned[:MAX_INPUT_CHARS]
    return cleaned


# =========================================================
# FEW-SHOT ПРИМЕРЫ
# =========================================================
def _load_examples():
    from .examples import EXAMPLES
    return [EXAMPLES[0]]

# =========================================================
# ВЫЗОВ LANGEXTRACT
# =========================================================
def _run_langextract(text: str, examples: List[Any]) -> Any:
    kwargs: Dict[str, Any] = {
        "text_or_documents": text,
        "prompt_description": PROMPT_DESCRIPTION,
        "examples": examples,
        "model_id": MODEL_ID,
        "model_url": OLLAMA_URL,
        "fence_output": False,
        "use_schema_constraints": False,
        "temperature": TEMPERATURE,
        "extraction_passes": EXTRACTION_PASSES,
        "max_workers": MAX_WORKERS,
        "max_char_buffer": MAX_CHAR_BUFFER,
        "language_model_params": {
            "num_ctx": NUM_CTX,
            "timeout": TIMEOUT,
        },
    }

    try:
        return lx.extract(**kwargs)
    except TypeError:
        # совместимость со старыми версиями langextract
        kwargs.pop("language_model_params", None)
        return lx.extract(**kwargs)


# =========================================================
# НОРМАЛИЗАЦИЯ РЕЗУЛЬТАТА LANGEXTRACT
# =========================================================
def _normalize_annotated_document(annotated: Any) -> List[Dict[str, Any]]:
    extractions: Iterable[Any]

    if hasattr(annotated, "extractions"):
        extractions = getattr(annotated, "extractions") or []
    elif isinstance(annotated, list) and len(annotated) == 1 and hasattr(annotated[0], "extractions"):
        extractions = getattr(annotated[0], "extractions") or []
    else:
        extractions = []

    normalized: List[Dict[str, Any]] = []

    for item in extractions:
        extraction_class = getattr(item, "extraction_class", "") or ""
        extraction_text = getattr(item, "extraction_text", "") or ""
        attributes = getattr(item, "attributes", {}) or {}
        char_interval = getattr(item, "char_interval", None)

        start_pos = None
        end_pos = None
        if char_interval is not None:
            start_pos = getattr(char_interval, "start_pos", None)
            end_pos = getattr(char_interval, "end_pos", None)

        if not extraction_class or not extraction_text:
            continue

        normalized.append(
            {
                "class": str(extraction_class),
                "text": str(extraction_text).strip(),
                "attributes": dict(attributes) if isinstance(attributes, dict) else {},
                "start_pos": start_pos,
                "end_pos": end_pos,
            }
        )

    return normalized


# =========================================================
# СБОРКА ИТОГОВОГО JSON
# =========================================================
def _assemble_result(entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    onec_fields = {
        "заголовок": "",
        "организация_заказчик": "",
        "подразделение": "",
        "физические_лица": [],
    }

    extra_findings = {
        "вид_документа": "",
        "номер_документа": "",
        "дата_документа": "",
        "город": "",
        "организация_исполнитель": "",
        "контрагент": "",
        "объект_наименование": "",
        "объект_адрес": "",
        "стоимость_услуг": "",
        "режим_охраны": "",
        "лицензия": "",
        "подписанты": [],
    }

    seen_persons = set()
    seen_signatories = set()

    for entity in entities:
        cls = entity["class"]
        text = entity["text"]
        attrs = entity.get("attributes") or {}

        if cls == "document_title":
            candidate = _clean_string(attrs.get("canonical_title") or text)
            onec_fields["заголовок"] = _pick_better_title(onec_fields["заголовок"], candidate)
            continue

        if cls == "customer_organization":
            if not onec_fields["организация_заказчик"]:
                onec_fields["организация_заказчик"] = _clean_string(attrs.get("normalized_name") or text)
            continue

        if cls == "department":
            if not onec_fields["подразделение"]:
                onec_fields["подразделение"] = _clean_string(attrs.get("normalized_name") or text)
            continue

        if cls == "person_full_name":
            fio = _clean_string(attrs.get("normalized_full_name") or text)
            if fio and fio not in seen_persons:
                seen_persons.add(fio)
                onec_fields["физические_лица"].append(fio)

            role = _clean_string(attrs.get("role"))
            signatory_value = fio or _clean_string(text)
            if signatory_value and signatory_value not in seen_signatories:
                seen_signatories.add(signatory_value)
                extra_findings["подписанты"].append(
                    {
                        "фио": signatory_value,
                        "роль": role,
                    }
                )
            continue

        if cls == "document_type" and not extra_findings["вид_документа"]:
            extra_findings["вид_документа"] = _clean_string(attrs.get("normalized_value") or text)
            continue

        if cls == "document_number" and not extra_findings["номер_документа"]:
            extra_findings["номер_документа"] = _clean_string(text)
            continue

        if cls == "document_date" and not extra_findings["дата_документа"]:
            extra_findings["дата_документа"] = _clean_string(attrs.get("iso_date") or text)
            continue

        if cls == "city" and not extra_findings["город"]:
            extra_findings["город"] = _clean_string(text)
            continue

        if cls == "executor_organization" and not extra_findings["организация_исполнитель"]:
            extra_findings["организация_исполнитель"] = _clean_string(attrs.get("normalized_name") or text)
            continue

        if cls == "counterparty_organization" and not extra_findings["контрагент"]:
            extra_findings["контрагент"] = _clean_string(attrs.get("normalized_name") or text)
            continue

        if cls == "object_name" and not extra_findings["объект_наименование"]:
            extra_findings["объект_наименование"] = _clean_string(attrs.get("normalized_value") or text).strip('«»')
            continue

        if cls == "object_address" and not extra_findings["объект_адрес"]:
            extra_findings["объект_адрес"] = _clean_string(text)
            continue

        if cls == "service_cost" and not extra_findings["стоимость_услуг"]:
            amount = attrs.get("normalized_amount")
            if amount not in (None, "", 0):
                currency = _clean_string(attrs.get("currency")) or "RUB"
                frequency = _clean_string(attrs.get("frequency"))
                suffix = f" {frequency}" if frequency else ""
                extra_findings["стоимость_услуг"] = f"{amount} {currency}{suffix}".strip()
            else:
                extra_findings["стоимость_услуг"] = _clean_string(text)
            continue

        if cls == "guard_schedule" and not extra_findings["режим_охраны"]:
            extra_findings["режим_охраны"] = _clean_string(text)
            continue

        if cls == "license_info" and not extra_findings["лицензия"]:
            extra_findings["лицензия"] = _clean_string(text)
            continue

    return {
        "onec_fields": onec_fields,
        "extra_findings": extra_findings,
        "_meta": {
            "engine": "langextract",
            "model_id": MODEL_ID,
            "entity_count": len(entities),
        },
    }


# =========================================================
# УТИЛИТЫ
# =========================================================
def _pick_better_title(current_value: str, new_value: str) -> str:
    current_value = _clean_string(current_value)
    new_value = _clean_string(new_value)

    if not new_value:
        return current_value
    if not current_value:
        return new_value

    return new_value if len(new_value) > len(current_value) else current_value


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


def _save_artifacts(annotated: Any) -> None:
    artifact_dir = Path(LANGEXTRACT_ARTIFACT_DIR)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    try:
        lx.io.save_annotated_documents(
            [annotated],
            output_name="latest_extractions.jsonl",
            output_dir=str(artifact_dir),
        )
    except Exception:
        return

    try:
        html_content = lx.visualize(str(artifact_dir / "latest_extractions.jsonl"))
        html_path = artifact_dir / "latest_visualization.html"
        with html_path.open("w", encoding="utf-8") as f:
            if hasattr(html_content, "data"):
                f.write(html_content.data)
            else:
                f.write(str(html_content))
    except Exception:
        return