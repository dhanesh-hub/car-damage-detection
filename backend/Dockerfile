# 1. We switch to 'bullseye' - it is much more stable for OpenCV
FROM python:3.9-slim-bullseye

# 2. Permissions and non-interactive settings
USER root
ENV DEBIAN_FRONTEND=noninteractive

# 3. Use the updated, stable package names for OpenCV
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Standard setup
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app

# 5. Start the server
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]