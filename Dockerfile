# Use stable slim image with Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files & buffer logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install essential system libraries (optimized for TensorFlow)
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

    # Convert old model format to new .keras format on startup
RUN mkdir -p /app/media
COPY convert_model.py /app/dashboard/
RUN python /app/convert_model.py || true

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose Railway's expected port
EXPOSE 8000

# Start Gunicorn for Django
CMD ["gunicorn", "capstone.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]
