FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by psycopg2-binary and pymupdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy library sources and the service
COPY libs/ ./libs/
COPY services/ ./services/

EXPOSE 8000

CMD ["uvicorn", "services.chatwithdoc.main:app", "--host", "0.0.0.0", "--port", "8000"]
