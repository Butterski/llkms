FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY setup.py .
COPY . .

RUN pip install --no-cache-dir -e .

# Option 1: Run the app using the entry-point defined in setup.py (llkms command)
# CMD ["llkms"]

# Option 2: Run main.py directly
CMD ["python", "main.py"]