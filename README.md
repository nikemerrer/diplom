# Извлечение реквизитов договора для 1С

Сервис извлекает сущности из неструктурированных документов, приводит результат к JSON-формату для 1С и поддерживает асинхронную обработку через очередь задач.

Проект состоит из двух частей:

- основной FastAPI-сервис в каталоге `app/`, который читает документы, режет текст на чанки, обращается к Ollama и отдает результат;
- модуль интеграции с 1С в каталоге `1С/`, который отправляет файл в сервис и заполняет реквизиты документа по результату.

## Что делает сервис

На вход подается файл договора или другого делового документа. Сервис:

1. извлекает текст из `doc`, `docx`, `pdf`, `rtf`, `txt`;
2. нормализует текст;
3. разбивает документ на чанки;
4. вызывает LLM по каждому чанку;
5. объединяет найденные сущности;
6. дополнительно верифицирует итог по evidence;
7. возвращает JSON с блоками `onec_fields`, `extra_findings` и `_meta`.

## Архитектура

### Основной сервис

Файл: [`app/main.py`](/mnt/d/Diplom_finnaly/diplom-6b37957370f03454f22746587876c1a50dfc2cfa/app/main.py)

Основные функции:

- Web UI на `/`
- мониторинг на `/monitor`
- healthcheck на `/health`
- синхронная обработка через `POST /api/extract-file`
- асинхронная обработка через очередь `POST /api/jobs` + `POST /api/jobs/result`
- хранение задач в SQLite `jobs.db`
- фоновый воркер, который берет задачи со статусом `queued`

Статусы задач:

- `queued`
- `running`
- `done`
- `error`

### Извлечение текста

Файл: [`app/utils/file_reader.py`](/mnt/d/Diplom_finnaly/diplom-6b37957370f03454f22746587876c1a50dfc2cfa/app/utils/file_reader.py)

Поддержка форматов:

- `docx`: через `python-docx`, затем fallback на чтение XML из zip-пакета
- `pdf`: через `PyPDF2`, затем `pdfminer.six`, затем OCR через `pdf2image` + `pytesseract`
- `rtf`: через `striprtf`
- `doc`: через внешние утилиты `antiword` и `catdoc`
- `txt`: best-effort decode (`utf-8-sig`, `utf-8`, `cp1251`, `latin-1`)

Если расширение файла неизвестно, сервис пытается распознать формат по сигнатуре содержимого.

### Извлечение сущностей

Файл: [`app/utils/entity_extractor.py`](/mnt/d/Diplom_finnaly/diplom-6b37957370f03454f22746587876c1a50dfc2cfa/app/utils/entity_extractor.py)

Логика:

- подготовка текста;
- построение header-чанка и обычных чанков;
- вызов Ollama с JSON-ответом по каждому чанку;
- повторный запрос при плохом JSON;
- merge по набору правил;
- verify-этап по evidence;
- финальная нормализация;
- fallback-эвристики на Python для номера, даты, города, заголовка и части ФИО.

Сервис отслеживает runtime-прогресс задачи:

- `read_file`
- `prepare_extract`
- `prepare_text`
- `extract_chunks`
- `merge_chunks`
- `verify_result`
- `finalize_result`
- `done`
- `error`

## Входные данные

### 1. Синхронный режим

`POST /api/extract-file`

Формат:

- `multipart/form-data`
- поле `file`

Поддерживаемые расширения:

- `.doc`
- `.docx`
- `.pdf`
- `.rtf`
- `.txt`

Пример:

```bash
curl -X POST http://127.0.0.1:8000/api/extract-file \
  -F "file=@test/NDA.docx"
```

### 2. Очередь задач

`POST /api/jobs`

Формат тела:

```json
{
  "filename": "contract.docx",
  "content_base64": "<base64>",
  "meta": {
    "user_id": "demo"
  }
}
```

`meta` сейчас принимается, но в БД не сохраняется и в обработке не используется.

Пример:

```bash
python3 - <<'PY'
import base64, json
from pathlib import Path
payload = {
    "filename": "NDA.docx",
    "content_base64": base64.b64encode(Path("test/NDA.docx").read_bytes()).decode("ascii"),
    "meta": {"user_id": "demo"},
}
print(json.dumps(payload, ensure_ascii=False))
PY
```

## Выходные данные

Основной результат извлечения выглядит так:

```json
{
  "onec_fields": {
    "заголовок": "Договор оказания услуг",
    "организация_заказчик": "ООО Ромашка",
    "подразделение": "Служба безопасности",
    "физические_лица": [
      "Иванов Иван Иванович"
    ]
  },
  "extra_findings": {
    "вид_документа": "договор",
    "номер_документа": "145-01",
    "дата_документа": "12.03.2024",
    "город": "Москва",
    "организация_исполнитель": "ООО Охрана",
    "контрагент": "ООО Ромашка",
    "объект_наименование": "Склад",
    "объект_адрес": "г. Москва, ул. Пример, д. 1",
    "стоимость_услуг": "560000 рублей",
    "режим_охраны": "ежедневно",
    "лицензия": "Лицензия №123",
    "подписанты": [
      {
        "фио": "Петров Петр Петрович",
        "роль": "исполнитель"
      }
    ]
  },
  "_meta": {
    "engine": "custom_llm_chunked_verifier",
    "model_id": "qwen2.5:1.5b",
    "chunk_count": 4,
    "failed_chunks": 0,
    "verify_enabled": true,
    "stateless_requests": true
  }
}
```

### Какие поля реально использует 1С

Модуль [`1С/ИИЗаполнение.bsl`](/mnt/d/Diplom_finnaly/diplom-6b37957370f03454f22746587876c1a50dfc2cfa/1С/ИИЗаполнение.bsl) записывает в документ только `onec_fields`:

- `заголовок`
- `организация_заказчик`
- `подразделение`
- `физические_лица`

Блок `extra_findings` в 1С не маппится в реквизиты, а только выводится сообщениями пользователю.

## API

### `GET /`

Web UI для ручной загрузки файла, предпросмотра текста и скачивания JSON.

### `GET /monitor`

Страница мониторинга очереди и обработки.

Показывает:

- количество задач по статусам;
- активную задачу;
- очередь;
- недавно завершенные задачи;
- ошибки;
- прогресс текущей задачи по чанкам.

### `GET /health`

Пример ответа:

```json
{
  "status": "ok",
  "model_id": "qwen2.5:1.5b",
  "jobs": {
    "queued": 0,
    "running": 0,
    "done": 5,
    "error": 1,
    "total": 6
  }
}
```

### `POST /api/extract-file`

Синхронная обработка файла.

Ответ:

```json
{
  "filename": "NDA.docx",
  "preview_text": "....",
  "data": {
    "onec_fields": {},
    "extra_findings": {},
    "_meta": {}
  },
  "debug_raw": "{...meta...}"
}
```

### `POST /api/jobs`

Создает задачу в очереди.

Ответ:

```json
{
  "job_id": "uuid",
  "status": "queued"
}
```

### `POST /api/jobs/result`

Принимает:

```json
{
  "job_id": "uuid"
}
```

Возвращает статус задачи, результат или ошибку:

```json
{
  "job_id": "uuid",
  "status": "running",
  "result": null,
  "error": null,
  "progress": {
    "stage": "extract_chunks",
    "message": "Обработка чанка 2 из 5",
    "chunk_count": 5,
    "chunks_completed": 1,
    "failed_chunks": 0,
    "progress_percent": 20
  }
}
```

### `GET /api/jobs`

Список задач. Поддерживает:

- `status`
- `limit`
- `offset`

### `GET /api/jobs/{job_id}`

Детальная информация по задаче, включая `progress`.

### `GET /api/jobs/stats`

Сводка по статусам задач.

### `GET /api/monitor/summary`

Главный endpoint для мониторинга.

Возвращает:

- `stats`
- `active_job`
- `running_jobs`
- `queue_preview`
- `recent_done`
- `recent_error`
- `avg_processing_sec_recent`
- `model_id`

## Как работает очередь

Хранение:

- таблица `jobs`
- таблица `payloads`

Фоновый поток:

- периодически ищет первую задачу со статусом `queued`;
- переводит ее в `running`;
- читает `payloads.file_bytes`;
- вызывает `process_file_bytes(...)`;
- сохраняет `result_json` или `error_text`.

При старте сервиса зависшие `running`-задачи переводятся в `error`.

## Мониторинг процесса извлечения сущностей

После доработки в проекте есть live-мониторинг текущего состояния задачи.

Что видно в процессе обработки:

- факт постановки задачи в очередь;
- момент, когда задачу взял воркер;
- чтение и распознавание текста;
- количество чанков;
- номер текущего чанка;
- сколько чанков уже завершено;
- сколько чанков завершились ошибкой JSON-парсинга;
- этап merge;
- этап verify;
- этап финальной нормализации;
- итоговый статус `done` или `error`.

Мониторинг доступен:

- визуально: `http://127.0.0.1:8000/monitor`
- программно: `GET /api/monitor/summary`

## Интеграция с 1С

Каталог: [`1С/`](/mnt/d/Diplom_finnaly/diplom-6b37957370f03454f22746587876c1a50dfc2cfa/1С)

### Основной сценарий

1. Пользователь нажимает кнопку "Заполнить ИИ".
2. 1С выбирает файл и отправляет его в `POST /api/jobs`.
3. Сервис возвращает `job_id`.
4. Пользователь нажимает "Получить результат ИИ".
5. 1С вызывает `POST /api/jobs/result`.
6. Если статус `done`, 1С заполняет реквизиты документа.

### Что важно

- 1С ожидает сервис на `127.0.0.1:8000`;
- создание задачи идет через JSON + base64;
- повторная отправка новой задачи на той же форме блокируется, пока не завершена текущая;
- после `done` или `error` переменная `JobIdИИ` очищается.

### Дополнительный сервис в `1С/app.py`

Файл [`1С/app.py`](/mnt/d/Diplom_finnaly/diplom-6b37957370f03454f22746587876c1a50dfc2cfa/1С/app.py) поднимает отдельный простой FastAPI-сервис для сценария "заявка + комментарий".

Он:

- принимает файл в base64;
- сохраняет его в `1С/storage/`;
- отдает HTML-страницу с textarea для комментария;
- позволяет получить комментарий по `job_id`.

Этот сервис не участвует в основном пайплайне извлечения сущностей и очереди из `app/main.py`.

## Запуск

### Docker Compose

Основной способ запуска:

```bash
docker compose up -d --build
```

После старта доступны:

- `http://localhost:8000/` - Web UI
- `http://localhost:8000/monitor` - мониторинг
- `http://localhost:8000/health` - healthcheck
- `http://localhost:11434/` - API Ollama

### Сервисы Compose

`docker-compose.yml` поднимает:

- `ollama`
- `app`

Особенности:

- для `ollama` включен `gpus: all`;
- модель задается через `MODEL_ID`;
- при старте контейнера `ollama` модель автоматически подтягивается через `ollama pull`, если ее еще нет;
- приложение использует `OLLAMA_URL=http://ollama:11434`.

## Переменные окружения

### Основной сервис

- `HOST` или `APP_HOST`
- `PORT` или `APP_PORT`
- `DB_PATH`
- `JOB_POLL_INTERVAL_SEC`
- `PREVIEW_CHARS`
- `OLLAMA_URL`
- `OLLAMA_MODEL_ID`
- `MODEL_ID`
- `TEMPERATURE`
- `OLLAMA_TIMEOUT`
- `OLLAMA_NUM_CTX`
- `OLLAMA_NUM_PREDICT_CHUNK`
- `OLLAMA_NUM_PREDICT_VERIFY`
- `OLLAMA_NUM_PREDICT_NAME_NORMALIZER`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `HEADER_FOCUS_CHARS`
- `MAX_CHUNKS`
- `VERIFY_ENABLED`
- `VERIFY_MAX_EVIDENCE_CHARS`
- `OCR_LANG`

### Значения по умолчанию в Compose

- `MODEL_ID=qwen2.5:1.5b`
- `TEMPERATURE=0.0`
- `OLLAMA_TIMEOUT=120`
- `OLLAMA_NUM_CTX=4096`
- `OLLAMA_NUM_PREDICT_CHUNK=420`
- `OLLAMA_NUM_PREDICT_VERIFY=520`
- `CHUNK_SIZE=2200`
- `CHUNK_OVERLAP=320`
- `HEADER_FOCUS_CHARS=2600`
- `MAX_CHUNKS=10`
- `VERIFY_ENABLED=1`

## Зависимости

Python-зависимости:

- `fastapi`
- `uvicorn`
- `jinja2`
- `python-multipart`
- `python-docx`
- `pdfminer.six`
- `PyPDF2`
- `striprtf`
- `pdf2image`
- `pytesseract`
- `Pillow`
- `requests`

Системные зависимости внутри контейнера `app`:

- `antiword`
- `catdoc`
- `tesseract-ocr`
- `poppler-utils`

## Ограничения и особенности

- сервис ориентирован на русскоязычные деловые документы;
- качество результата зависит от качества OCR и качества модели в Ollama;
- для больших документов итог ограничивается `MAX_CHUNKS`;
- очередь сейчас однопоточная: один фоновый воркер обрабатывает задачи последовательно;
- `meta` в `POST /api/jobs` пока не используется;
- `app/utils/validator.py` и `app/utils/examples.py` в текущем пайплайне фактически не задействованы;
- файл [`onec_fields.json`](/mnt/d/Diplom_finnaly/diplom-6b37957370f03454f22746587876c1a50dfc2cfa/onec_fields.json) похож на более старый пример выходного формата и не совпадает с текущей структурой ответа `onec_fields/extra_findings`.

## Структура проекта

```text
app/
  main.py                 основной FastAPI-сервис
  start.sh                запуск uvicorn
  templates/
    index.html            web UI
    monitor.html          мониторинг
  static/
    style.css             стили UI и монитора
  utils/
    file_reader.py        извлечение текста из файлов
    entity_extractor.py   LLM-пайплайн извлечения сущностей
    validator.py          черновик schema-валидации
    examples.py           старый few-shot пример

ollama/
  Dockerfile
  start.sh                запуск ollama и авто-pull модели

1С/
  ИИЗаполнение.bsl        модуль формы для интеграции с 1С
  app.py                  отдельный вспомогательный FastAPI-сервис
  storage/                файлы и meta.json вспомогательного сервиса

test/
  NDA.docx
  doc-text.pdf
  Договор - скан.pdf
  Договор на охранные услуги.doc
```

## Быстрая проверка после запуска

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/jobs/stats
curl http://127.0.0.1:8000/api/monitor/summary?limit=20
```

Для ручного теста можно загрузить один из файлов из каталога `test/` через Web UI или через `POST /api/extract-file`.
