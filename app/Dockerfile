FROM python:3.11-slim

WORKDIR /app

# Системные утилиты для .doc и OCR
RUN apt-get update && \
    apt-get install -y --no-install-recommends antiword catdoc tesseract-ocr poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Python-зависимости веб-приложения
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Локальная библиотека LangExtract
COPY langextract-main /langextract-main
RUN pip install --no-cache-dir -e /langextract-main

# Код FastAPI
COPY app/ .

EXPOSE 8000
RUN chmod +x /app/start.sh
CMD ["/app/start.sh"]
