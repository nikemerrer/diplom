# Codex Playground

## Структура проекта
- `app/` — FastAPI-приложение для извлечения полей договоров; содержит шаблоны, статические файлы и вспомогательные утилиты.
- `langextract-main/` — локальная копия библиотеки [LangExtract](https://github.com/google/langextract), ставится в editable-режиме и обеспечивает работу извлечения сущностей.
- `ollama/` — образ с сервером Ollama и скриптом загрузки выбранной модели (по умолчанию `gemma3:4b`).
- `docker-compose.yml` — поднимает Ollama и веб-приложение в одном окружении с предустановленными переменными окружения.
- Прочие файлы (`__init__.py`, `README` внутри зависимостей и т.д.) служат для упаковки и документации.

## Запуск через Docker Compose
1. Требуются Docker и поддержка GPU (в compose-файле указано `gpus: all`).
2. При необходимости укажите модель через переменную `MODEL_ID` (по умолчанию `gemma3:4b`).
3. Соберите и поднимите сервисы:
   ```bash
   MODEL_ID=gemma3:4b docker compose up --build
   ```
4. FastAPI будет доступен на `http://localhost:8000`, Ollama — на `http://localhost:11434`.

## Локальный запуск без Docker
1. Установите Python 3.11+, а также системные пакеты `antiword` и `catdoc` (как в `app/Dockerfile`).
2. Создайте виртуальное окружение и поставьте зависимости приложения:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r app/requirements.txt
   ```
3. Установите локальную библиотеку извлечения:
   ```bash
   pip install -e langextract-main
   ```
4. Запустите Ollama (локально или через `docker compose up ollama`) и убедитесь, что нужная модель загружена.
5. Экспортируйте переменные окружения при необходимости:
   ```bash
   export OLLAMA_URL=http://localhost:11434
   export OLLAMA_MODEL_ID=gemma3:4b
   export TEMPERATURE=0.1
   export OLLAMA_NUM_PREDICT=384
   export OLLAMA_TIMEOUT=300
   export CHUNK_SIZE=6000
   export CHUNK_OVERLAP=400
   export MIN_CHUNK_FOR_RETRY=1800
   ```
6. Запустите приложение:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## Используемые технологии
- FastAPI и Uvicorn для API и сервера.
- LangExtract для LLM-ориентированного извлечения данных.
- Ollama для работы с локальными моделями (например, `gemma3:4b`).
- Docker / Docker Compose для контейнеризации и воспроизводимого запуска.
- Дополнительные пакеты: Jinja2, pdfminer.six, python-docx, PyPDF2, striprtf.
