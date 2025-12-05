# -------- Build stage --------
FROM python:3.10-slim AS builder

WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PATH=/home/appuser/.local/bin:$PATH

# Create non-root user first
RUN useradd -m -u 1000 appuser

# Install system dependencies (needed for building some packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user for Python installs
USER appuser

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --user torch==2.9.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install --user --no-cache-dir -r requirements.txt

# -------- Runtime stage --------
FROM python:3.10-slim

WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN useradd -m -u 1000 appuser

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopenblas0 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER appuser

# Copy installed packages from builder
COPY --from=builder /home/appuser/.local /home/appuser/.local

# Copy application code
COPY . .

# Create media directory
RUN mkdir -p /app/media

EXPOSE 8000

CMD ["gunicorn", "capstone.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-"]
