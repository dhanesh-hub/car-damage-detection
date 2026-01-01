FROM python:3.9-slim-bullseye

USER root
ENV DEBIAN_FRONTEND=noninteractive
# This tells Detectron2 NOT to look for a GPU during the build
ENV FORCE_CPU=1

# 1. System Dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 gcc g++ python3-dev git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Upgrade Pip & Build Tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel ninja

# 3. Install Torch (CPU version)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. INSTALL DETECTRON2 (With build-isolation fix)
RUN pip install --no-cache-dir --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# 5. Remaining Requirements (The one that was failing)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]