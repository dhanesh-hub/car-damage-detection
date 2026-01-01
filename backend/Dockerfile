# VERSION 3.0 - FORCED RESET
FROM python:3.9-slim-bullseye

USER root
ENV DEBIAN_FRONTEND=noninteractive

# 1. System Dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 gcc g++ python3-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Upgrade Pip
RUN pip install --no-cache-dir --upgrade pip

# 3. Install Torch First (CPU version)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. THE CACHE BUSTER
# This dummy variable forces Render to stop using old cached layers
ARG CACHEBUST=1 

# 5. Install Detectron2 from GIT (NO BROKEN LINKS)
# We are NOT using the -f link anymore. This builds it from source.
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# 6. Remaining Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]