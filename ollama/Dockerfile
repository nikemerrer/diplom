FROM ollama/ollama:latest

# Кладём стартовый скрипт внутрь образа
COPY start.sh /start.sh

# Чиним возможные CRLF от Windows и даём права на исполнение
RUN sed -i 's/\r$//' /start.sh && chmod +x /start.sh

# Открываем API-порт Ollama
EXPOSE 11434

# Запускаем сервер через наш скрипт
ENTRYPOINT ["/start.sh"]
