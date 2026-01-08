FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all agent code
COPY . .

# Expose port
EXPOSE 8000

# Configuração de concorrência (padrão: 2 execuções simultâneas)
ENV MAX_CONCURRENT_EXECUTIONS=2

# Usar nosso server customizado com controle de concorrência
# Em vez de: langgraph dev --host 0.0.0.0 --port 8000 --no-browser
CMD ["python", "server.py"]
