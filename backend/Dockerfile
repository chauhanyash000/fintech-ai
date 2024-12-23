# Use Python 3.11 slim image
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV FLASK_APP=main.py
ENV PORT=8080


# Run the application
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 3600 main:app