FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files & buffer logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/media

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

RUN mkdir -p /app/scripts && \
    cat > /tmp/convert_model.py << 'PYTHON_EOF'
import os
from pathlib import Path

h5_path = Path('media/improved_pest_model.h5')
if h5_path.exists():
    try:
        from scripts.convert_h5_to_keras import convert_h5_to_keras
        convert_h5_to_keras(str(h5_path))
        print(f"Successfully converted {h5_path} to .keras format")
    except Exception as e:
        print(f"Conversion skipped: {e}")
PYTHON_EOF
RUN python /tmp/convert_model.py || true

# Expose Railway's expected port
EXPOSE 8000

CMD ["gunicorn", "capstone.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-"]
