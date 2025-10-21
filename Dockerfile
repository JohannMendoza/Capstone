FROM python:3.11-slim

# <CHANGE> Minimal system dependencies - opencv-python-headless doesn't need OpenGL
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "capstone.wsgi", "--bind", "0.0.0.0:8000"]
