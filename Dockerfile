# Use Python 3.10 slim as base image
FROM python:3.10-slim

# Install system dependencies (add more if needed)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install protobuf, sentencepiece, and other dependencies
RUN pip install --no-cache-dir \
    protobuf \
    sentencepiece \
    torch \
    transformers \
    accelerate \
    fastapi \
    uvicorn[standard] \
    safetensors \
    hf_transfer  # Add hf_transfer here if needed

# Set the working directory
WORKDIR /app

# Copy the app file to the container
COPY app.py /app/app.py

# Expose the port that uvicorn will run on
EXPOSE 8000

# Command to run the app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
