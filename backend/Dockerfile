FROM python:3.9-slim-bullseye

USER root
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system tools + compiler
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Install specific versions of Torch that match the Detectron2 link
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Install Detectron2 using the EXACT matching wheel for Torch 2.1
RUN pip install --no-cache-dir detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/index.html

# 4. Install the rest of your requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy code and start
COPY . /app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]