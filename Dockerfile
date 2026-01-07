FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for pandas & sklearn)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose Fly.io required port
EXPOSE 8080

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
