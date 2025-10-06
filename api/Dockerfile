# ==============================
# 🐳 E-Commerce FastAPI Dockerfile
# ==============================

# Base Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency files first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Environment variable (optional)
ENV PYTHONUNBUFFERED=1

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

