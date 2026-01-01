# Backend Dockerfile (simple). Note: Detectron2 has platform-specific requirements (CUDA, specific wheels).
# For production with Detectron2, build on a CUDA-enabled base image and follow Detectron2's install guide.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN apt-get update && apt-get install -y build-essential git curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
