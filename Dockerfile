FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt setup.py ./

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install -e .

# Final stage
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /app /app

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Run the installed command with docker run -ti llkms:latest
ENTRYPOINT ["llkms"]