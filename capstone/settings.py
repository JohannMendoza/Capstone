"""
Django settings for capstone project
Safe for LOCAL (.env) and RAILWAY deployment
"""

from pathlib import Path
import os
from dotenv import load_dotenv
import dj_database_url

import ssl
import certifi

# Fix SSL issues locally
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# ================================================================
# 📂 BASE DIRECTORY & .ENV LOADING
# ================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")  # load local .env if exists

# ================================================================
# 🔐 SECURITY
# ================================================================
SECRET_KEY = os.getenv("SECRET_KEY", "unsafe-dev-key")
DEBUG = False

ALLOWED_HOSTS = [
    "localhost",
    "127.0.0.1",
    "lanzofields.capstoneph.com",
    ".up.railway.app",
]


# ================================================================
# 🧱 APPLICATIONS
# ================================================================
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "dashboard",
    "pwa",  # ✅ ADD PWA SUPPORT
]

AUTH_USER_MODEL = "dashboard.CustomUser"

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",  # static files in prod
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "capstone.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [os.path.join(BASE_DIR, "dashboard", "templates")],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "capstone.wsgi.application"

# ================================================================
# 🗄 DATABASE
# ================================================================
if os.getenv("DATABASE_URL"):
    DATABASES = {
        "default": dj_database_url.config(
            default=os.getenv("DATABASE_URL"),
            conn_max_age=600,
            conn_health_checks=True,
        )
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

# ================================================================
# 🔑 PASSWORD VALIDATION
# ================================================================
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# ================================================================
# 🌍 INTERNATIONALIZATION
# ================================================================
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# ================================================================
# 🖼 STATIC & MEDIA FILES
# ================================================================
STATIC_URL = "/static/"
STATICFILES_DIRS = [os.path.join(BASE_DIR, "static")]
STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

MEDIA_URL = "/media/"
MEDIA_ROOT = os.path.join(BASE_DIR, "media")

MODEL_PATH = os.path.join(MEDIA_ROOT, "best.pt")  # YOLOv8 model

# ================================================================
# 🛡 SECURITY HEADERS (Railway HTTPS)
# ================================================================
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
CSRF_TRUSTED_ORIGINS = [
    "https://*.up.railway.app",
    "https://*.railway.app",
    "https://lanzofields.capstoneph.com",
]

# ================================================================
# 📧 EMAIL CONFIGURATION (SendGrid / Console for DEBUG)
# ================================================================
if DEBUG:
    EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"
    print("⚙️  Using console email backend (development mode)")
else:
    EMAIL_BACKEND = "sendgrid_backend.SendgridBackend"
    SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
    DEFAULT_FROM_EMAIL = os.getenv("DEFAULT_FROM_EMAIL", "aurelioescala15@gmail.com")
    SENDGRID_SANDBOX_MODE_IN_DEBUG = False
    SENDGRID_ECHO_TO_STDOUT = False
    EMAIL_TIMEOUT = 10
    SENDGRID_TRACK_EMAIL_OPENS = False
    SENDGRID_TRACK_CLICKS = False

# ================================================================
# 🚀 UPLOADS & LOGIN
# ================================================================
DATA_UPLOAD_MAX_MEMORY_SIZE = 52_428_800  # 50 MB
LOGIN_URL = "/login/"

# ================================================================
# 📱 PWA CONFIGURATION FOR LANZOFIELDS
# ================================================================
PWA_APP_NAME = 'LanzoFields'
PWA_APP_DESCRIPTION = "Crop Disease Detection using YOLOv8"
PWA_APP_THEME_COLOR = '#2E7D32'  # Green color for agriculture
PWA_APP_BACKGROUND_COLOR = '#FFFFFF'
PWA_APP_DISPLAY = 'standalone'
PWA_APP_SCOPE = '/'
PWA_APP_ORIENTATION = 'any'
PWA_APP_START_URL = '/'
PWA_APP_STATUS_BAR_COLOR = 'default'
PWA_APP_ICONS = [
    {
        'src': '/static/dashboard/img/192x192.png',  # ✅ EXISTING PATH
        'sizes': '192x192',
        'type': 'image/png'
    },
    {
        'src': '/static/dashboard/img/512x512.png',  # ✅ EXISTING PATH
        'sizes': '512x512',
        'type': 'image/png'
    }
]
PWA_APP_ICONS_APPLE = [
    {
        'src': '/static/dashboard/img/192x192.png',  # ✅ EXISTING PATH
        'sizes': '192x192',
        'type': 'image/png'
    }
]
PWA_APP_SPLASH_SCREEN = [
    {
        'src': '/static/dashboard/img/640x1136.png',  # ✅ EXISTING PATH
        'media': '(device-width: 320px) and (device-height: 568px) and (-webkit-device-pixel-ratio: 2)'
    }
]
PWA_SERVICE_WORKER_PATH = os.path.join(BASE_DIR, 'capstone', 'dashboard', 'static', 'dashboard', 'js', 'serviceworker.js')  # ✅ EXISTING PATH