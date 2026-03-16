# -*- coding: utf-8 -*-
"""
Few-shot примеры для LangExtract под двухконтурную схему:
1) onec_fields  -> только то, что реально идет в автозаполнение 1С
2) extra_findings -> всё остальное, что можно показать в сообщениях пользователю

Важное правило LangExtract:
- extraction_text должен быть ВЕРБАТИМ из текста примера
- экстракции должны идти в порядке появления в тексте
"""

from __future__ import annotations

import langextract as lx

ExampleData = lx.data.ExampleData
Extraction = lx.data.Extraction



# ---------------------------------------------------------------------------
# Пример 1. Договор охранных услуг
# ---------------------------------------------------------------------------
EX1_TEXT = """ДОГОВОР № 145-01
об оказании охранных услуг

г. Москва                                                                 «05» мая 2025 г.

ООО «Первый Плюс», именуемое в дальнейшем «Заказчик», в лице генерального директора Иванова Ивана Ивановича,
действующего на основании устава, с одной стороны и Общество с ограниченной ответственностью Частное охранное
предприятие «Мурзилка» (лицензия № 223344 зарегистрирована 01.01.2025 и действительна до 31.12.2025),
именуемое в дальнейшем «Исполнитель», в лице Генерального директора Сидорова Петра Петровича,
действующего на основании Устава, с другой стороны, а вместе именуемые «Стороны», заключили настоящий Договор.

1.1. Заказчик поручает, а Исполнитель принимает на себя обязательства по оказанию следующих услуг:
охрана имущества Заказчика на Объекте – «Дом труда», расположенном по адресу: улица Труда, д.1.

2.1.1. Для выполнения обязательств по настоящему Договору выставить на Объекте Заказчика посты с режимом работы:
с 10:00 до 21:00 ежедневно.

5.1. Стоимость услуг, оказываемых Исполнителем по настоящему Договору, составляет:
560 000 (пятьсот шестьдесят) рублей 00 копеек ежемесячно.
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
        extraction_class="document_title",
        extraction_text="ДОГОВОР № 145-01\nоб оказании охранных услуг",
        attributes={
            "canonical_title": "ДОГОВОР № 145-01 об оказании охранных услуг",
            "bucket": "onec_fields",
            "onec_key": "заголовок",
        },
    ),
    Extraction(
        extraction_class="city",
        extraction_text="Москва",
        attributes={"bucket": "extra_findings"},
    ),
    Extraction(
        extraction_class="document_date",
        extraction_text="«05» мая 2025 г.",
        attributes={"iso_date": "2025-05-05", "bucket": "extra_findings"},
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
        extraction_class="person_full_name",
        extraction_text="Иванова Ивана Ивановича",
        attributes={
            "normalized_full_name": "Иванов Иван Иванович",
            "bucket": "onec_fields",
            "onec_key": "физические_лица",
            "role": "подписант_заказчика",
        },
    ),
    Extraction(
        extraction_class="executor_organization",
        extraction_text="Общество с ограниченной ответственностью Частное охранное\nпредприятие «Мурзилка»",
        attributes={
            "normalized_name": "ЧОП «Мурзилка»",
            "bucket": "extra_findings",
            "role": "исполнитель",
        },
    ),
    Extraction(
        extraction_class="license_info",
        extraction_text="лицензия № 223344 зарегистрирована 01.01.2025 и действительна до 31.12.2025",
        attributes={"bucket": "extra_findings"},
    ),
    Extraction(
        extraction_class="person_full_name",
        extraction_text="Сидорова Петра Петровича",
        attributes={
            "normalized_full_name": "Сидоров Петр Петрович",
            "bucket": "onec_fields",
            "onec_key": "физические_лица",
            "role": "подписант_исполнителя",
        },
    ),
    Extraction(
        extraction_class="object_name",
        extraction_text="«Дом труда»",
        attributes={"normalized_value": "Дом труда", "bucket": "extra_findings"},
    ),
    Extraction(
        extraction_class="object_address",
        extraction_text="улица Труда, д.1",
        attributes={"bucket": "extra_findings"},
    ),
    Extraction(
        extraction_class="guard_schedule",
        extraction_text="с 10:00 до 21:00 ежедневно",
        attributes={"bucket": "extra_findings"},
    ),
    Extraction(
        extraction_class="service_cost",
        extraction_text="560 000 (пятьсот шестьдесят) рублей 00 копеек ежемесячно",
        attributes={
            "normalized_amount": 560000,
            "currency": "RUB",
            "frequency": "ежемесячно",
            "bucket": "extra_findings",
        },
    ),
]

# ---------------------------------------------------------------------------
# Пример 2. Служебная записка с явным подразделением
# ---------------------------------------------------------------------------
EX2_TEXT = """СЛУЖЕБНАЯ ЗАПИСКА

Департамент информационных технологий

О продлении доступа к информационной системе

г. Санкт-Петербург                                                     12.02.2025

Прошу продлить доступ сотрудников к внутренней информационной системе компании.
Ответственный за сопровождение вопроса — Петров Петр Петрович.
Согласовано с ООО «Северный Контур».
"""

EX2_EXTRACTIONS = [
    Extraction(
        extraction_class="document_type",
        extraction_text="СЛУЖЕБНАЯ ЗАПИСКА",
        attributes={"normalized_value": "служебная записка", "bucket": "extra_findings"},
    ),
    Extraction(
        extraction_class="department",
        extraction_text="Департамент информационных технологий",
        attributes={
            "normalized_name": "Департамент информационных технологий",
            "bucket": "onec_fields",
            "onec_key": "подразделение",
        },
    ),
    Extraction(
        extraction_class="document_title",
        extraction_text="О продлении доступа к информационной системе",
        attributes={
            "canonical_title": "О продлении доступа к информационной системе",
            "bucket": "onec_fields",
            "onec_key": "заголовок",
        },
    ),
    Extraction(
        extraction_class="city",
        extraction_text="Санкт-Петербург",
        attributes={"bucket": "extra_findings"},
    ),
    Extraction(
        extraction_class="document_date",
        extraction_text="12.02.2025",
        attributes={"iso_date": "2025-02-12", "bucket": "extra_findings"},
    ),
    Extraction(
        extraction_class="person_full_name",
        extraction_text="Петров Петр Петрович",
        attributes={
            "normalized_full_name": "Петров Петр Петрович",
            "bucket": "onec_fields",
            "onec_key": "физические_лица",
            "role": "упомянутое_физлицо",
        },
    ),
    Extraction(
        extraction_class="counterparty_organization",
        extraction_text="ООО «Северный Контур»",
        attributes={"normalized_name": "ООО «Северный Контур»", "bucket": "extra_findings"},
    ),
]

# ---------------------------------------------------------------------------
# Пример 3. Акт без подразделения
# ---------------------------------------------------------------------------
EX3_TEXT = """АКТ № 18
об оказанных услугах

г. Москва                                                            31.05.2025

ООО «Первый Плюс» и ЧОП «Мурзилка» подтверждают, что услуги охраны за май 2025 года оказаны в полном объеме.
Со стороны Заказчика акт подписал Иванов Иван Иванович.
Со стороны Исполнителя акт подписал Сидоров Петр Петрович.
Стоимость услуг за отчетный период: 560 000 рублей 00 копеек.
"""

EX3_EXTRACTIONS = [
    Extraction(
        extraction_class="document_type",
        extraction_text="АКТ",
        attributes={"normalized_value": "акт", "bucket": "extra_findings"},
    ),
    Extraction(
        extraction_class="document_number",
        extraction_text="18",
        attributes={"bucket": "extra_findings"},
    ),
    Extraction(
        extraction_class="document_title",
        extraction_text="АКТ № 18\nоб оказанных услугах",
        attributes={
            "canonical_title": "АКТ № 18 об оказанных услугах",
            "bucket": "onec_fields",
            "onec_key": "заголовок",
        },
    ),
    Extraction(
        extraction_class="city",
        extraction_text="Москва",
        attributes={"bucket": "extra_findings"},
    ),
    Extraction(
        extraction_class="document_date",
        extraction_text="31.05.2025",
        attributes={"iso_date": "2025-05-31", "bucket": "extra_findings"},
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
        attributes={"normalized_name": "ЧОП «Мурзилка»", "bucket": "extra_findings", "role": "исполнитель"},
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
    Extraction(
        extraction_class="person_full_name",
        extraction_text="Сидоров Петр Петрович",
        attributes={
            "normalized_full_name": "Сидоров Петр Петрович",
            "bucket": "onec_fields",
            "onec_key": "физические_лица",
            "role": "подписант_исполнителя",
        },
    ),
    Extraction(
        extraction_class="service_cost",
        extraction_text="560 000 рублей 00 копеек",
        attributes={
            "normalized_amount": 560000,
            "currency": "RUB",
            "bucket": "extra_findings",
        },
    ),
]

EXAMPLES = [
    ExampleData(text=EX1_TEXT, extractions=EX1_EXTRACTIONS),
    ExampleData(text=EX2_TEXT, extractions=EX2_EXTRACTIONS),
    ExampleData(text=EX3_TEXT, extractions=EX3_EXTRACTIONS),
]