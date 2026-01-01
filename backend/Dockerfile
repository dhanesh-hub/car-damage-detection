# 1. Use a standard Python image
FROM python:3.9-slim

# 2. Switch to root to ensure we have permissions for system installs
USER root

# 3. Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# 4. Install system dependencies required for OpenCV
# These are essential for the 'import cv2' command to work on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Set the working directory inside the container
WORKDIR /app

# 6. Copy requirements first (improves build speed using cache)
# Since Render Root is 'backend', we just use './' or 'requirements.txt'
COPY requirements.txt ./

# 7. Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 8. Copy the rest of your backend code
# IMPORTANT: Use '.' because we are already inside the 'backend' folder
COPY . /app

# 9. Expose the port (Render usually uses 10000, but 8000 is fine)
EXPOSE 8000

# 10. Command to start the server
# We use 0.0.0.0 so it is accessible from the internet
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]