# VERSION 4.0 - NO BUILD ISOLATION
FROM python:3.9-slim-bullseye

USER root
ENV DEBIAN_FRONTEND=noninteractive

# 1. System Dependencies (Added wheel and setuptools)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 gcc g++ python3-dev git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Upgrade Pip & Build Tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel ninja

# 3. Install Torch (CPU version)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. INSTALL DETECTRON2 (The Fix)
# --no-build-isolation allows the build to see the 'torch' we just installed
RUN pip install --no-cache-dir --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# 5. Remaining Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]