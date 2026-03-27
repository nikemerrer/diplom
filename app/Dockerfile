FROM python:3.11-slim

WORKDIR /app

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        antiword \
        catdoc \
        tesseract-ocr \
        poppler-utils && \
    rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

EXPOSE 8000

RUN chmod +x /app/start.sh
CMD ["/app/start.sh"]