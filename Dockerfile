FROM python:3.9-slim

# Set environment variables to optimize Python execution
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

WORKDIR /code

# Install system dependencies, curl, and download custom CNN weights from GitHub LFS media CDN
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Create cache directory and set full access
RUN mkdir -p /.cache && chmod -R 777 /.cache

# Download weights directly from GitHub LFS media server
RUN mkdir -p /code/models && \
    curl -L -o /code/models/best_model.pth https://media.githubusercontent.com/media/ShreyasVavley/Facial-Emotion/main/models/best_model.pth

COPY ./src /code/src

# Expose default port (7860 is Hugging Face Spaces standard; 8000 for standard Docker)
EXPOSE 7860

# Run FastAPI app binding to 0.0.0.0
CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT}"]
