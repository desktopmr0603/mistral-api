FROM python:3.10-slim

# system deps (add more if needed)
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

# install python libs
RUN pip install --no-cache-dir torch transformers accelerate fastapi uvicorn[standard] safetensors

# copy app
WORKDIR /app
COPY app.py /app/app.py

# runtime environment variables: MODEL_ID and HF_TOKEN (set these in RunPod)
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
