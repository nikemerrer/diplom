# -*- coding: utf-8 -*-
"""
Короткий few-shot пример для LangExtract.
Оставлен один компактный пример, чтобы не перегружать prompt и не упираться в timeout Ollama.
"""
from __future__ import annotations

import langextract as lx

ExampleData = lx.data.ExampleData
Extraction = lx.data.Extraction


EX1_TEXT = """ДОГОВОР № 145-01
ООО «Первый Плюс» и ЧОП «Мурзилка».
Подписал Иванов Иван Иванович.
"""

EX1_EXTRACTIONS = [
    Extraction(
        extraction_class="document_type",
        extraction_text="ДОГОВОР",
        attributes={"normalized_value": "договор", "bucket": "extra_findings"},
    ),
    Extraction(
        extraction_class="document_number",
        extraction_text="145-01",
        attributes={"bucket": "extra_findings"},
    ),
    Extraction(
        extraction_class="customer_organization",
        extraction_text="ООО «Первый Плюс»",
        attributes={
            "normalized_name": "ООО «Первый Плюс»",
            "bucket": "onec_fields",
            "onec_key": "организация_заказчик",
            "role": "заказчик",
        },
    ),
    Extraction(
        extraction_class="executor_organization",
        extraction_text="ЧОП «Мурзилка»",
        attributes={
            "normalized_name": "ЧОП «Мурзилка»",
            "bucket": "extra_findings",
            "role": "исполнитель",
        },
    ),
    Extraction(
        extraction_class="person_full_name",
        extraction_text="Иванов Иван Иванович",
        attributes={
            "normalized_full_name": "Иванов Иван Иванович",
            "bucket": "onec_fields",
            "onec_key": "физические_лица",
            "role": "подписант_заказчика",
        },
    ),
]

EXAMPLES = [
    ExampleData(text=EX1_TEXT, extractions=EX1_EXTRACTIONS),
]
