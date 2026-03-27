# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MODEL_ID = os.getenv("OLLAMA_MODEL_ID") or os.getenv("MODEL_ID") or "qwen2.5:1.5b"

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
NUM_PREDICT_CHUNK = int(os.getenv("OLLAMA_NUM_PREDICT_CHUNK", "900"))
NUM_PREDICT_VERIFY = int(os.getenv("OLLAMA_NUM_PREDICT_VERIFY", "700"))
NUM_PREDICT_NAME_NORMALIZER = int(os.getenv("OLLAMA_NUM_PREDICT_NAME_NORMALIZER", "320"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "320"))
HEADER_FOCUS_CHARS = int(os.getenv("HEADER_FOCUS_CHARS", "2600"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "10"))
VERIFY_ENABLED = os.getenv("VERIFY_ENABLED", "1").lower() in {"1", "true", "yes"}
VERIFY_MAX_EVIDENCE_CHARS = int(os.getenv("VERIFY_MAX_EVIDENCE_CHARS", "2600"))


ProgressCallback = Callable[[Dict[str, Any]], None]


def extract_onec_fields(
    text: str, progress_callback: Optional[ProgressCallback] = None
) -> Dict[str, Any]:
    full_text = _prepare_text(text)
    if not full_text:
        raise RuntimeError("На вход извлекателя пришёл пустой текст документа")

    _emit_progress(
        progress_callback,
        {
            "stage": "prepare_text",
            "message": "Подготовка текста документа",
            "text_length": len(full_text),
        },
    )

    chunks = _build_chunks(full_text)
    if not chunks:
        raise RuntimeError("Не удалось сформировать чанки документа")

    chunk_results: List[Dict[str, Any]] = []
    failed_chunks = 0

    print(f"[LLM] full_text_len={len(full_text)} chunk_count={len(chunks)}")
    _emit_progress(
        progress_callback,
        {
            "stage": "extract_chunks",
            "message": "Извлечение сущностей по чанкам",
            "chunk_count": len(chunks),
            "chunks_completed": 0,
            "failed_chunks": 0,
        },
    )

    for idx, chunk in enumerate(chunks, start=1):
        print(f"[LLM][chunk {idx}/{len(chunks)}] len={len(chunk)}")
        _emit_progress(
            progress_callback,
            {
                "stage": "extract_chunks",
                "message": f"Обработка чанка {idx} из {len(chunks)}",
                "chunk_index": idx,
                "chunk_count": len(chunks),
                "chunk_length": len(chunk),
                "chunks_completed": idx - 1,
                "failed_chunks": failed_chunks,
            },
        )
        chunk_result, ok = _extract_chunk(chunk, idx, len(chunks))
        if not ok:
            failed_chunks += 1
        chunk_results.append(chunk_result)
        _emit_progress(
            progress_callback,
            {
                "stage": "extract_chunks",
                "message": f"Чанк {idx} из {len(chunks)} обработан",
                "chunk_index": idx,
                "chunk_count": len(chunks),
                "chunk_length": len(chunk),
                "chunk_ok": ok,
                "chunks_completed": idx,
                "failed_chunks": failed_chunks,
            },
        )

    if failed_chunks >= len(chunks):
        raise RuntimeError("Не удалось извлечь данные: все чанки вернули некорректный JSON")

    _emit_progress(
        progress_callback,
        {
            "stage": "merge_chunks",
            "message": "Сборка итогового результата из чанков",
            "chunk_count": len(chunks),
            "chunks_completed": len(chunks),
            "failed_chunks": failed_chunks,
        },
    )
    merged_raw, evidence_map = _merge_chunk_results(full_text, chunk_results)

    if VERIFY_ENABLED:
        _emit_progress(
            progress_callback,
            {
                "stage": "verify_result",
                "message": "Проверка итоговых сущностей по evidence",
                "chunk_count": len(chunks),
                "chunks_completed": len(chunks),
                "failed_chunks": failed_chunks,
            },
        )
        verified = _verify_result(merged_raw, evidence_map)
    else:
        verified = merged_raw

    _emit_progress(
        progress_callback,
        {
            "stage": "finalize_result",
            "message": "Финальная нормализация результата",
            "chunk_count": len(chunks),
            "chunks_completed": len(chunks),
            "failed_chunks": failed_chunks,
        },
    )
    result = _normalize_final_result(verified, full_text)
    result["_meta"] = {
        "engine": "custom_llm_chunked_verifier",
        "model_id": MODEL_ID,
        "chunk_count": len(chunks),
        "failed_chunks": failed_chunks,
        "verify_enabled": VERIFY_ENABLED,
        "stateless_requests": True,
    }
    return result


extract_entities_from_text = extract_onec_fields


def _emit_progress(
    callback: Optional[ProgressCallback], payload: Dict[str, Any]
) -> None:
    if callback is None:
        return
    callback(payload)


def _prepare_text(text: str) -> str:
    cleaned = (text or "").replace("\x00", " ")
    cleaned = cleaned.replace("\uf0ad", "-").replace("\uf0b7", "-").replace("\xa0", " ")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _build_chunks(text: str) -> List[str]:
    chunks: List[str] = []

    header_chunk = text[:HEADER_FOCUS_CHARS].strip()
    if header_chunk:
        chunks.append(header_chunk)

    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + CHUNK_SIZE, text_len)
        if end < text_len:
            split_pos = text.rfind("\n", start + int(CHUNK_SIZE * 0.65), end)
            if split_pos > start + 400:
                end = split_pos
        chunk = text[start:end].strip()
        if chunk and chunk not in chunks:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
        if MAX_CHUNKS > 0 and len(chunks) >= MAX_CHUNKS:
            break

    return chunks


def _extract_chunk(chunk: str, chunk_no: int, total_chunks: int) -> Tuple[Dict[str, Any], bool]:
    prompt = f"""
Ты извлекаешь сущности из русского делового документа для автозаполнения 1С.

Это чанк {chunk_no} из {total_chunks}. Извлекай ТОЛЬКО то, что явно видно в этом куске.
Не выдумывай. Для каждого найденного значения верни короткую ДОСЛОВНУЮ цитату в поле evidence.

Важно:
- НЕ возвращай пустые поля.
- НЕ заполняй поля-заглушки.
- Если в чанке ничего полезного нет, верни пустой объект:
  {{"onec_fields": {{}}, "extra_findings": {{}}}}

Верни только JSON без markdown и без пояснений.

Формат ответа:
{{
  "onec_fields": {{
    "заголовок": {{"value": "", "evidence": ""}},
    "организация_заказчик": {{"value": "", "evidence": ""}},
    "подразделение": {{"value": "", "evidence": ""}},
    "физические_лица": [
      {{"value": "", "evidence": ""}}
    ]
  }},
  "extra_findings": {{
    "вид_документа": {{"value": "", "evidence": ""}},
    "номер_документа": {{"value": "", "evidence": ""}},
    "дата_документа": {{"value": "", "evidence": ""}},
    "город": {{"value": "", "evidence": ""}},
    "организация_исполнитель": {{"value": "", "evidence": ""}},
    "контрагент": {{"value": "", "evidence": ""}},
    "объект_наименование": {{"value": "", "evidence": ""}},
    "объект_адрес": {{"value": "", "evidence": ""}},
    "стоимость_услуг": {{"value": "", "evidence": ""}},
    "режим_охраны": {{"value": "", "evidence": ""}},
    "лицензия": {{"value": "", "evidence": ""}},
    "подписанты": [
      {{"фио": "", "роль": "", "evidence": ""}}
    ]
  }}
}}

Правила:
1. "заголовок" собирай в одну строку.
2. "организация_заказчик" — только организация-заказчик.
3. "подразделение" — только если явно указано.
4. В "физические_лица" включай только реальных людей.
5. Все ФИО в полях "физические_лица" и "подписанты.фио" возвращай строго в ИМЕНИТЕЛЬНОМ ПАДЕЖЕ.
6. Если в документе ФИО стоит в косвенном падеже, преобразуй его в форму для поиска в 1С.
7. Примеры:
   - "Петрова Ивана Ивановича" -> "Петров Иван Иванович"
   - "Сидорова Петра Петровича" -> "Сидоров Петр Петрович"
   - "Свободенюка Никиты Владимировича" -> "Свободенюк Никита Владимирович"
8. evidence оставляй дословным фрагментом документа.
9. value можно нормализовать, evidence менять нельзя.
10. Если в evidence стоит "Петрова Ивана Ивановича", то value должно быть "Петров Иван Иванович".

Чанк документа:
{chunk}
""".strip()

    try:
        raw = _call_ollama(prompt, num_predict=NUM_PREDICT_CHUNK)
        parsed = _extract_json(raw)
        print(f"[LLM][chunk {chunk_no}/{total_chunks}] ok response_len={len(raw)}")
        return parsed, True
    except Exception as exc:
        print(f"[LLM][chunk {chunk_no}/{total_chunks}] parse_error={exc}")

    retry_prompt = f"""
Извлеки только найденные данные из этого чанка документа.
Если данных нет, верни:
{{"onec_fields": {{}}, "extra_findings": {{}}}}

Строго верни только валидный JSON без пояснений.

Для строковых полей используй:
{{"value":"...", "evidence":"..."}}

Для массивов:
"физические_лица": [{{"value":"...", "evidence":"..."}}]
"подписанты": [{{"фио":"...", "роль":"...", "evidence":"..."}}]

Все ФИО в value/фио верни в ИМЕНИТЕЛЬНОМ ПАДЕЖЕ.
Примеры:
- Петрова Ивана Ивановича -> Петров Иван Иванович
- Сидорова Петра Петровича -> Сидоров Петр Петрович
- Свободенюка Никиты Владимировича -> Свободенюк Никита Владимирович

Чанк:
{chunk}
""".strip()

    try:
        raw = _call_ollama(retry_prompt, num_predict=max(NUM_PREDICT_CHUNK, 1200))
        parsed = _extract_json(raw)
        print(f"[LLM][chunk {chunk_no}/{total_chunks}] retry_ok response_len={len(raw)}")
        return parsed, True
    except Exception as exc:
        print(f"[LLM][chunk {chunk_no}/{total_chunks}] retry_failed={exc}")
        return _empty_chunk_result(), False


def _empty_chunk_result() -> Dict[str, Any]:
    return {"onec_fields": {}, "extra_findings": {}}


def _merge_chunk_results(full_text: str, chunk_results: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    string_fields = [
        ("onec_fields", "заголовок"),
        ("onec_fields", "организация_заказчик"),
        ("onec_fields", "подразделение"),
        ("extra_findings", "вид_документа"),
        ("extra_findings", "номер_документа"),
        ("extra_findings", "дата_документа"),
        ("extra_findings", "город"),
        ("extra_findings", "организация_исполнитель"),
        ("extra_findings", "контрагент"),
        ("extra_findings", "объект_наименование"),
        ("extra_findings", "объект_адрес"),
        ("extra_findings", "стоимость_услуг"),
        ("extra_findings", "режим_охраны"),
        ("extra_findings", "лицензия"),
    ]

    candidates: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    persons: List[Dict[str, str]] = []
    signers: List[Dict[str, str]] = []
    evidence_map: Dict[str, List[str]] = {}

    for chunk_idx, chunk_result in enumerate(chunk_results):
        onec = chunk_result.get("onec_fields") or {}
        extra = chunk_result.get("extra_findings") or {}

        for section, field in string_fields:
            source = onec if section == "onec_fields" else extra
            item = source.get(field) or {}
            if isinstance(item, dict):
                value = _s(item.get("value"))
                evidence = _s(item.get("evidence"))
            else:
                value = _s(item)
                evidence = ""
            if value:
                norm = _normalize_key_like(value)
                candidates.setdefault((section, field), []).append(
                    {
                        "value": value,
                        "norm": norm,
                        "evidence": evidence,
                        "chunk_idx": chunk_idx,
                    }
                )
                if evidence:
                    evidence_map.setdefault(f"{section}.{field}", []).append(evidence)

        for person in onec.get("физические_лица") or []:
            if not isinstance(person, dict):
                continue
            value = _clean_person_name(_s(person.get("value")))
            evidence = _s(person.get("evidence"))
            if value:
                persons.append({"value": value, "evidence": evidence})
                if evidence:
                    evidence_map.setdefault("onec_fields.физические_лица", []).append(evidence)

        for signer in extra.get("подписанты") or []:
            if not isinstance(signer, dict):
                continue
            fio = _clean_person_name(_s(signer.get("фио")))
            role = _s(signer.get("роль"))
            evidence = _s(signer.get("evidence"))
            if fio:
                signers.append({"фио": fio, "роль": role, "evidence": evidence})
                if evidence:
                    evidence_map.setdefault("extra_findings.подписанты", []).append(evidence)

    merged = {
        "onec_fields": {
            "заголовок": _choose_best_string(candidates.get(("onec_fields", "заголовок"), []), mode="title", full_text=full_text),
            "организация_заказчик": _choose_best_string(candidates.get(("onec_fields", "организация_заказчик"), []), mode="org", full_text=full_text),
            "подразделение": _choose_best_string(candidates.get(("onec_fields", "подразделение"), []), mode="dept", full_text=full_text),
            "физические_лица": _dedupe_persons(persons),
        },
        "extra_findings": {
            "вид_документа": _choose_best_string(candidates.get(("extra_findings", "вид_документа"), []), mode="simple", full_text=full_text),
            "номер_документа": _choose_best_string(candidates.get(("extra_findings", "номер_документа"), []), mode="simple", full_text=full_text),
            "дата_документа": _choose_best_string(candidates.get(("extra_findings", "дата_документа"), []), mode="date", full_text=full_text),
            "город": _choose_best_string(candidates.get(("extra_findings", "город"), []), mode="simple", full_text=full_text),
            "организация_исполнитель": _choose_best_string(candidates.get(("extra_findings", "организация_исполнитель"), []), mode="org", full_text=full_text),
            "контрагент": _choose_best_string(candidates.get(("extra_findings", "контрагент"), []), mode="org", full_text=full_text),
            "объект_наименование": _choose_best_string(candidates.get(("extra_findings", "объект_наименование"), []), mode="simple", full_text=full_text),
            "объект_адрес": _choose_best_string(candidates.get(("extra_findings", "объект_адрес"), []), mode="simple", full_text=full_text),
            "стоимость_услуг": _choose_best_string(candidates.get(("extra_findings", "стоимость_услуг"), []), mode="simple", full_text=full_text),
            "режим_охраны": _choose_best_string(candidates.get(("extra_findings", "режим_охраны"), []), mode="simple", full_text=full_text),
            "лицензия": _choose_best_string(candidates.get(("extra_findings", "лицензия"), []), mode="simple", full_text=full_text),
            "подписанты": _dedupe_signers(signers),
        },
    }

    return merged, evidence_map


def _verify_result(merged_result: Dict[str, Any], evidence_map: Dict[str, List[str]]) -> Dict[str, Any]:
    compact_evidence = {
        key: _truncate_join(values, VERIFY_MAX_EVIDENCE_CHARS // 4)
        for key, values in evidence_map.items()
        if values
    }

    prompt = f"""
Проверь извлечённые данные для 1С.

Тебе даны:
1. итоговый JSON
2. доказательства (короткие цитаты из документа)

Нельзя использовать домыслы.
Если поле не подтверждается доказательствами, очисти его.
Для всех ФИО в onec_fields.физические_лица и extra_findings.подписанты оставляй ИМЕНИТЕЛЬНЫЙ ПАДЕЖ.
Если evidence дано в косвенном падеже, evidence не меняй, но value/фио приведи к именительному падежу.
evidence не меняй, оно должно быть дословной цитатой.
Сохрани тот же формат JSON.

Итоговый JSON:
{json.dumps(merged_result, ensure_ascii=False)}

Доказательства:
{json.dumps(compact_evidence, ensure_ascii=False)}
""".strip()

    try:
        raw = _call_ollama(prompt, num_predict=NUM_PREDICT_VERIFY)
        parsed = _extract_json(raw)

        if not isinstance(parsed, dict):
            return merged_result
        if "onec_fields" not in parsed or "extra_findings" not in parsed:
            return merged_result
        print(f"[LLM][verify] ok response_len={len(raw)}")
        return parsed
    except Exception as exc:
        print(f"[LLM][verify] failed={exc}")
        return merged_result


def _call_ollama(prompt: str, num_predict: int) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL_ID,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": TEMPERATURE,
                "num_ctx": NUM_CTX,
                "num_predict": num_predict,
            },
            "keep_alive": 300,
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    return data.get("response", "")


def _extract_json(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()

    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()

    for candidate in _json_candidates(raw):
        fixed = _fix_common_json_issues(candidate)
        try:
            parsed = json.loads(fixed)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    raise RuntimeError("Не удалось распарсить JSON модели")


def _json_candidates(raw: str) -> List[str]:
    candidates: List[str] = []
    if raw:
        candidates.append(raw)

    balanced = _extract_balanced_json_object(raw)
    if balanced and balanced not in candidates:
        candidates.append(balanced)

    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        value = match.group(0)
        if value not in candidates:
            candidates.append(value)

    return candidates


def _extract_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]

    return text[start:]


def _fix_common_json_issues(text: str) -> str:
    text = text.strip()
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u201e", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r",\s*([}\]])", r"\1", text)

    stack: List[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()

    if not in_string and stack:
        text = text + "".join(reversed(stack))

    return text


def _normalize_final_result(data: Dict[str, Any], full_text: str) -> Dict[str, Any]:
    onec = data.get("onec_fields") or {}
    extra = data.get("extra_findings") or {}

    result = {
        "onec_fields": {
            "заголовок": _s(onec.get("заголовок")),
            "организация_заказчик": _s(onec.get("организация_заказчик")),
            "подразделение": _s(onec.get("подразделение")),
            "физические_лица": _arr_str(onec.get("физические_лица")),
        },
        "extra_findings": {
            "вид_документа": _s(extra.get("вид_документа")),
            "номер_документа": _s(extra.get("номер_документа")),
            "дата_документа": _s(extra.get("дата_документа")),
            "город": _s(extra.get("город")),
            "организация_исполнитель": _s(extra.get("организация_исполнитель")),
            "контрагент": _s(extra.get("контрагент")),
            "объект_наименование": _s(extra.get("объект_наименование")),
            "объект_адрес": _s(extra.get("объект_адрес")),
            "стоимость_услуг": _s(extra.get("стоимость_услуг")),
            "режим_охраны": _s(extra.get("режим_охраны")),
            "лицензия": _s(extra.get("лицензия")),
            "подписанты": _signers(extra.get("подписанты")),
        },
    }

    _apply_python_fallbacks(result, full_text)
    result = _expand_initials_from_text(result, full_text)
    result = _normalize_person_names_with_llm(result)
    result = _prefer_full_names_over_initials(result)
    return result



def _expand_initials_from_text(result: Dict[str, Any], full_text: str) -> Dict[str, Any]:
    people = result.get("onec_fields", {}).get("физические_лица", []) or []
    signers = result.get("extra_findings", {}).get("подписанты", []) or []

    expanded_people: List[str] = []
    for fio in people:
        value = _clean_person_name(_s(fio))
        if _is_initials_only_name(value):
            expanded = _find_full_name_for_initials(value, full_text)
            expanded_people.append(expanded or value)
        else:
            expanded_people.append(value)
    result["onec_fields"]["физические_лица"] = expanded_people

    expanded_signers: List[Dict[str, str]] = []
    for signer in signers:
        if not isinstance(signer, dict):
            continue
        fio = _clean_person_name(_s(signer.get("фио")))
        role = _s(signer.get("роль"))
        if _is_initials_only_name(fio):
            expanded = _find_full_name_for_initials(fio, full_text)
            fio = expanded or fio
        expanded_signers.append({"фио": fio, "роль": role})
    result["extra_findings"]["подписанты"] = expanded_signers
    return result


def _prefer_full_names_over_initials(result: Dict[str, Any]) -> Dict[str, Any]:
    persons = result.get("onec_fields", {}).get("физические_лица", []) or []
    signers = result.get("extra_findings", {}).get("подписанты", []) or []

    best_by_signature: Dict[str, str] = {}
    for fio in persons:
        value = _clean_person_name(_s(fio))
        if not value:
            continue
        sig = _person_signature(value)
        if not sig:
            continue
        prev = best_by_signature.get(sig, "")
        if _person_score(value) > _person_score(prev):
            best_by_signature[sig] = value

    filtered_people: List[str] = []
    seen_people = set()
    for fio in persons:
        value = _clean_person_name(_s(fio))
        sig = _person_signature(value)
        chosen = best_by_signature.get(sig, value) if sig else value
        key = _normalize_key_like(chosen)
        if chosen and key not in seen_people:
            seen_people.add(key)
            filtered_people.append(chosen)
    result["onec_fields"]["физические_лица"] = filtered_people

    filtered_signers: List[Dict[str, str]] = []
    seen_signers = set()
    for signer in signers:
        if not isinstance(signer, dict):
            continue
        fio = _clean_person_name(_s(signer.get("фио")))
        role = _s(signer.get("роль"))
        sig = _person_signature(fio)
        chosen = best_by_signature.get(sig, fio) if sig else fio
        key = (_normalize_key_like(chosen), _normalize_key_like(role))
        if chosen and key not in seen_signers:
            seen_signers.add(key)
            filtered_signers.append({"фио": chosen, "роль": role})
    result["extra_findings"]["подписанты"] = filtered_signers

    return result


def _find_full_name_for_initials(short_name: str, full_text: str) -> str:
    parsed = _parse_initials_name(short_name)
    if not parsed:
        return ""
    surname, i1, i2 = parsed

    candidates = re.findall(r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\b", full_text)
    best = ""
    best_score = -1.0

    for cand in candidates:
        cand_clean = _clean_person_name(cand)
        parts = cand_clean.split()
        if len(parts) != 3:
            continue
        c_surname, c_name, c_pat = parts
        if c_name[:1].upper() != i1 or c_pat[:1].upper() != i2:
            continue

        score = 0.0
        if _surname_base(c_surname) == _surname_base(surname):
            score += 10.0
        elif _surname_base(c_surname).startswith(_surname_base(surname)[:4]) or _surname_base(surname).startswith(_surname_base(c_surname)[:4]):
            score += 6.0
        else:
            continue

        if not _is_initials_only_name(cand_clean):
            score += 5.0
        score += len(cand_clean) / 100.0

        if score > best_score:
            best_score = score
            best = cand_clean

    return best


def _parse_initials_name(value: str) -> tuple[str, str, str] | None:
    value = _clean_person_name(value)
    m = re.match(r"^([А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?)\s+([А-ЯЁ])\.?\s*([А-ЯЁ])\.?$", value)
    if not m:
        return None
    return m.group(1), m.group(2).upper(), m.group(3).upper()


def _is_initials_only_name(value: str) -> bool:
    return _parse_initials_name(value) is not None


def _surname_base(value: str) -> str:
    v = _normalize_key_like(value)
    v = v.replace('"', '')
    # грубая нормализация русских фамилий по падежным окончаниям
    for suffix in ("ыми", "ими", "ого", "его", "ому", "ему", "ой", "ей", "ым", "им", "ом", "ем", "а", "я", "у", "ю", "ы", "и"):
        if len(v) > 4 and v.endswith(suffix):
            v = v[: -len(suffix)]
            break
    return v


def _person_signature(value: str) -> str:
    value = _clean_person_name(value)
    if not value:
        return ""
    parsed = _parse_initials_name(value)
    if parsed:
        surname, i1, i2 = parsed
        return f"{_surname_base(surname)}|{i1}|{i2}"

    parts = value.split()
    if len(parts) >= 3:
        surname, name, pat = parts[0], parts[1], parts[2]
        return f"{_surname_base(surname)}|{name[:1].upper()}|{pat[:1].upper()}"
    return _normalize_key_like(value)


def _person_score(value: str) -> float:
    value = _clean_person_name(value)
    if not value:
        return -1.0
    score = 0.0
    parts = value.split()
    if len(parts) == 3:
        score += 10.0
    if not _is_initials_only_name(value):
        score += 10.0
    score += len(value) / 100.0
    return score


def _normalize_person_names_with_llm(result: Dict[str, Any]) -> Dict[str, Any]:
    source_names: List[str] = []

    for fio in result.get("onec_fields", {}).get("физические_лица", []):
        s = _clean_person_name(_s(fio))
        if s:
            source_names.append(s)

    for signer in result.get("extra_findings", {}).get("подписанты", []):
        if isinstance(signer, dict):
            s = _clean_person_name(_s(signer.get("фио")))
            if s:
                source_names.append(s)

    uniq_names: List[str] = []
    seen = set()
    for name in source_names:
        key = _normalize_key_like(name)
        if key and key not in seen:
            seen.add(key)
            uniq_names.append(name)

    if not uniq_names:
        return result

    mapping = _normalize_name_list_via_llm(uniq_names)
    if not mapping:
        return result

    fixed_persons: List[str] = []
    seen_persons = set()
    for fio in result.get("onec_fields", {}).get("физические_лица", []):
        src = _clean_person_name(_s(fio))
        dst = _clean_person_name(_s(mapping.get(src, src)))
        key = _normalize_key_like(dst)
        if dst and key not in seen_persons:
            seen_persons.add(key)
            fixed_persons.append(dst)
    result["onec_fields"]["физические_лица"] = fixed_persons

    fixed_signers: List[Dict[str, str]] = []
    seen_signers = set()
    for signer in result.get("extra_findings", {}).get("подписанты", []):
        if not isinstance(signer, dict):
            continue
        src = _clean_person_name(_s(signer.get("фио")))
        dst = _clean_person_name(_s(mapping.get(src, src)))
        role = _s(signer.get("роль"))
        key = (_normalize_key_like(dst), _normalize_key_like(role))
        if dst and key not in seen_signers:
            seen_signers.add(key)
            fixed_signers.append({
                "фио": dst,
                "роль": role,
            })
    result["extra_findings"]["подписанты"] = fixed_signers

    return result


def _normalize_name_list_via_llm(names: List[str]) -> Dict[str, str]:
    prompt = f"""
Приведи русские ФИО к ИМЕНИТЕЛЬНОМУ ПАДЕЖУ.
Верни только JSON без пояснений.

Правила:
1. Не сокращай ФИО.
2. Не выдумывай новые имена.
3. Если ФИО уже в именительном падеже — оставь как есть.
4. Верни JSON строго такого вида:
{{
  "items": [
    {{"source": "Петрова Ивана Ивановича", "normalized": "Петров Иван Иванович"}}
  ]
}}

Примеры:
- "Петрова Ивана Ивановича" -> "Петров Иван Иванович"
- "Сидорова Петра Петровича" -> "Сидоров Петр Петрович"
- "Свободенюка Никиты Владимировича" -> "Свободенюк Никита Владимирович"

ФИО:
{json.dumps(names, ensure_ascii=False)}
""".strip()

    try:
        raw = _call_ollama(prompt, num_predict=NUM_PREDICT_NAME_NORMALIZER)
        parsed = _extract_json(raw)
    except Exception as exc:
        print(f"[LLM][name_normalizer] failed={exc}")
        return {}

    items = parsed.get("items") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return {}

    mapping: Dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        source = _clean_person_name(_s(item.get("source")))
        normalized = _clean_person_name(_s(item.get("normalized")))
        if source and normalized:
            mapping[source] = normalized

    print(f"[LLM][name_normalizer] ok count={len(mapping)}")
    return mapping


def _apply_python_fallbacks(result: Dict[str, Any], full_text: str) -> None:
    text_head = full_text[:3000]

    if not result["extra_findings"]["номер_документа"]:
        m = re.search(r"(?:ДОГОВОР|АКТ|СОГЛАШЕНИЕ|ЗАЯВКА|ПРИКАЗ|СЛУЖЕБНАЯ ЗАПИСКА)[^\n№]{0,30}№\s*([A-Za-zА-Яа-я0-9\-\/]+)", text_head, re.I)
        if m:
            result["extra_findings"]["номер_документа"] = _s(m.group(1))

    if not result["extra_findings"]["дата_документа"]:
        m = re.search(r"(«?\d{1,2}»?\s+[А-Яа-я]+\s+\d{4}\s*г?\.?)|(\d{2}\.\d{2}\.\d{4})", text_head)
        if m:
            result["extra_findings"]["дата_документа"] = _s(m.group(0))

    if not result["extra_findings"]["город"]:
        m = re.search(r"\bг\.\s*([А-ЯA-ZЁ][А-Яа-яA-Za-zЁё\-\s]{1,40})", text_head)
        if m:
            result["extra_findings"]["город"] = _s(m.group(1))

    if not result["onec_fields"]["заголовок"]:
        lines = [ln.strip() for ln in text_head.splitlines() if ln.strip()]
        header_lines = lines[:4]
        if header_lines:
            result["onec_fields"]["заголовок"] = _s(" ".join(header_lines[:2]))

    if not result["extra_findings"]["вид_документа"]:
        m = re.search(r"\b(ДОГОВОР|АКТ|СОГЛАШЕНИЕ|СЛУЖЕБНАЯ ЗАПИСКА|ПРИКАЗ|ЗАЯВКА)\b", text_head, re.I)
        if m:
            result["extra_findings"]["вид_документа"] = _s(m.group(1))

    dept = result["onec_fields"]["подразделение"]
    if dept and re.match(r"^(г\.|город|дата|договор|акт)\b", dept, re.I):
        result["onec_fields"]["подразделение"] = ""

    if (
        result["onec_fields"]["организация_заказчик"]
        and result["extra_findings"]["организация_исполнитель"]
        and _normalize_key_like(result["onec_fields"]["организация_заказчик"])
        == _normalize_key_like(result["extra_findings"]["организация_исполнитель"])
    ):
        result["extra_findings"]["организация_исполнитель"] = ""


def _choose_best_string(items: List[Dict[str, Any]], mode: str, full_text: str) -> str:
    if not items:
        return ""

    counts: Dict[str, int] = {}
    for item in items:
        counts[item["norm"]] = counts.get(item["norm"], 0) + 1

    best_score = None
    best_value = ""

    for item in items:
        value = _s(item["value"])
        norm = item["norm"]
        evidence = _s(item.get("evidence"))
        if not value:
            continue

        score = 0.0
        score += counts.get(norm, 0) * 5.0
        score += min(len(value), 120) / 25.0
        score += max(0, 5 - item.get("chunk_idx", 0)) * 0.8

        if evidence:
            score += 2.0

        if mode == "title":
            if re.search(r"\b(ДОГОВОР|АКТ|СОГЛАШЕНИЕ|ЗАПИСКА|ПРИКАЗ|ЗАЯВКА)\b", value, re.I):
                score += 6.0
            if 8 <= len(value) <= 180:
                score += 3.0
        elif mode == "org":
            if re.search(r"\b(ООО|АО|ПАО|ИП|Общество|Компания|ЧОП)\b", value, re.I):
                score += 4.0
        elif mode == "dept":
            if re.search(r"\b(департамент|отдел|управление|служба|подразделение)\b", value, re.I):
                score += 5.0
        elif mode == "date":
            if re.search(r"\d{2}\.\d{2}\.\d{4}|«?\d{1,2}»?\s+[А-Яа-я]+\s+\d{4}", value):
                score += 4.0

        if best_score is None or score > best_score:
            best_score = score
            best_value = value

    return _s(best_value)


def _dedupe_persons(persons: List[Dict[str, str]]) -> List[str]:
    result: List[str] = []
    seen = set()
    for person in persons:
        value = _clean_person_name(_s(person.get("value")))
        key = _normalize_key_like(value)
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _dedupe_signers(signers: List[Dict[str, str]]) -> List[Dict[str, str]]:
    result: List[Dict[str, str]] = []
    seen = set()
    for signer in signers:
        fio = _clean_person_name(_s(signer.get("фио")))
        role = _s(signer.get("роль"))
        key = (_normalize_key_like(fio), _normalize_key_like(role))
        if fio and key not in seen:
            seen.add(key)
            result.append({"фио": fio, "роль": role})
    return result


def _clean_person_name(value: str) -> str:
    value = _s(value)
    value = re.sub(r"\b(генерального директора|генеральный директор|директора|директор|подписал|в лице|со стороны)\b", "", value, flags=re.I)
    value = " ".join(value.split())
    return value.strip(" ,.;:-")


def _normalize_key_like(value: str) -> str:
    value = _s(value).lower()
    value = value.replace("«", '"').replace("»", '"')
    value = re.sub(r"[^a-zа-яё0-9\" ]+", " ", value, flags=re.I)
    value = " ".join(value.split())
    return value


def _truncate_join(values: List[str], limit: int) -> str:
    uniq: List[str] = []
    seen = set()
    for value in values:
        s = _s(value)
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    joined = " | ".join(uniq)
    return joined[:limit]


def _s(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split()).strip()


def _arr_str(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    seen = set()
    for item in value:
        s = _clean_person_name(_s(item))
        key = _normalize_key_like(s)
        if s and key not in seen:
            seen.add(key)
            result.append(s)
    return result


def _signers(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []

    result: List[Dict[str, str]] = []
    seen = set()

    for item in value:
        if not isinstance(item, dict):
            continue
        fio = _clean_person_name(_s(item.get("фио")))
        role = _s(item.get("роль"))
        key = (_normalize_key_like(fio), _normalize_key_like(role))
        if fio and key not in seen:
            seen.add(key)
            result.append({"фио": fio, "роль": role})

    return result
