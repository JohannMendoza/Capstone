# Use lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files and force unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install essential system dependencies for OpenCV, Pillow, and TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libatlas-base-dev \
    libhdf5-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Django project into the container
COPY . .

# Collect static files (optional — useful for production)
# RUN python manage.py collectstatic --noinput

# Expose the Django port
EXPOSE 8000

# Run Gunicorn (production WSGI server)
CMD ["gunicorn", "capstone.wsgi:application", "--bind", "0.0.0.0:8000"]
