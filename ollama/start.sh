#!/bin/sh
set -e
ollama serve & SERVER_PID=$!

echo "⏳ Жду готовности ollama-сервера..."
for i in $(seq 1 120); do
  if ollama list >/dev/null 2>&1; then break; fi
  sleep 1
done

MODEL="${MODEL_ID:-qwen2.5:7b}"
if ! ollama show "$MODEL" >/dev/null 2>&1; then
  echo "⬇️  Скачиваю модель $MODEL ..."
  ollama pull "$MODEL"
fi

echo "✅ Ollama готов (порт 11434). Модель: $MODEL"
wait $SERVER_PID
