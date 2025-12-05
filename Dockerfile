# ============================================
# LANZOFIELDS - DOCKERFILE WITH PWA SUPPORT
# ============================================

# ---- Build Stage ----
FROM python:3.10-slim as builder

WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
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

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch==2.9.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# ✅ CREATE PWA FILES
RUN mkdir -p dashboard/static/dashboard/js dashboard/static/dashboard/img

# Create manifest.json
RUN echo '{
  "name": "LanzoFields",
  "short_name": "LanzoFields",
  "description": "Plant Disease Detection System",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#FFFFFF",
  "theme_color": "#2E7D32",
  "orientation": "any",
  "icons": [
    {
      "src": "/static/dashboard/img/192x192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/static/dashboard/img/512x512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ]
}' > dashboard/static/dashboard/manifest.json

# Create serviceworker.js
RUN echo '// LanzoFields PWA Service Worker
const CACHE_NAME = "lanzofields-pwa-" + new Date().getTime();
const urlsToCache = [
  "/",
  "/offline/",
  "/static/dashboard/img/192x192.png",
  "/static/dashboard/img/512x512.png"
];

self.addEventListener("install", event => {
  console.log("[SW] Installing...");
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log("[SW] Caching app shell");
        return cache.addAll(urlsToCache);
      })
  );
});

self.addEventListener("activate", event => {
  console.log("[SW] Activating...");
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames
          .filter(cacheName => cacheName.startsWith("lanzofields-pwa-"))
          .filter(cacheName => cacheName !== CACHE_NAME)
          .map(cacheName => caches.delete(cacheName))
      );
    })
  );
  return self.clients.claim();
});

self.addEventListener("fetch", event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        if (response) {
          return response;
        }
        return fetch(event.request);
      })
  );
});

console.log("[SW] Service Worker loaded");' > dashboard/static/dashboard/js/serviceworker.js

# ✅ Collect static files
RUN python manage.py collectstatic --noinput

# ---- Runtime Stage ----
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBUG=False

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopenblas0 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local /usr/local

# Copy collected static files
COPY --from=builder /app/staticfiles /app/staticfiles

# Copy application code
COPY --from=builder /app /app

# Create necessary directories
RUN mkdir -p media

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

# Run the app
CMD ["gunicorn", "capstone.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-"]