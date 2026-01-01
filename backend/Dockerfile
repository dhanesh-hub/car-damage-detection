FROM python:3.9-slim-bullseye

USER root
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system tools + compiler (Detectron2 needs these to unpack)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Upgrade pip and install the "heavy" AI dependencies first
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. INSTALL DETECTRON2 (The missing piece)
# This points to the official pre-built versions for CPU servers
RUN pip install --no-cache-dir detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/index.html

# 4. Install the rest of your requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy code and start
COPY . /app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]