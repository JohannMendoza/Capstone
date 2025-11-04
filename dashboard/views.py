# ✅ UPDATED: Fixed views.py with lazy loading and model caching
import os
import csv
import json
import base64
import logging
import traceback
import threading
import numpy as np
from io import BytesIO
from collections import Counter
from urllib.parse import urljoin
from PIL import Image
from django.conf import settings
from django.utils import timezone
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.template.loader import render_to_string
from django.core.mail import EmailMultiAlternatives, send_mail
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.contrib import messages, auth
from django.contrib.auth import authenticate, login, logout, get_user_model
from django.contrib.auth.decorators import login_required
from django.contrib.auth.tokens import default_token_generator
from django.contrib.auth.views import (
    PasswordResetView, PasswordResetDoneView,
    PasswordResetConfirmView, PasswordResetCompleteView
)
from django.contrib.sites.shortcuts import get_current_site
from django.urls import reverse, reverse_lazy
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.db.models import Count, Q

# Third-party / PDF library
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Local imports
from .forms import RegisterForm, PlantForm
from .models import CustomUser, Plant, TreeAnalysis, LeafImage, PestDetectionSession, PestDetectionResult
from .utils import send_verification_email




# <CHANGE> Removed top-level imports of torch, tensorflow, and ultralytics
# These will be imported inside functions that need them (lazy loading)
model_path = os.path.join(settings.MEDIA_ROOT, 'best.pt')

# <CHANGE> Added global model cache to avoid reloading models
_MODEL_CACHE = {}

# Set up logger
logger = logging.getLogger(__name__)

# ... existing code ...

from django.shortcuts import render
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes
from django.contrib.auth.tokens import default_token_generator
from django.template.loader import render_to_string
from django.conf import settings
from django.core.mail import EmailMultiAlternatives

def send_verification_email(user, request=None):
    """Send email verification link to user"""
    try:
        # Generate token
        uid = urlsafe_base64_encode(force_bytes(user.pk))
        token = default_token_generator.make_token(user)
        
        # Use hardcoded domain for production or get from request for development
        if settings.DEBUG and request:
            domain = get_current_site(request).domain
            protocol = "http"
        else:
            domain = "lanzofields.capstoneph.com"
            protocol = "https"
        
        verification_link = f"{protocol}://{domain}/verify/{uid}/{token}/"
        
        # Render email template
        html_body = render_to_string("dashboard/verify_email_template.html", {
            "user": user,
            "verification_link": verification_link,
            "domain": domain
        })
        
        text_body = f"Hello {user.username},\n\nPlease verify your email by clicking the link below:\n{verification_link}\n\nThis link expires in 24 hours."
        
        # Send email
        email = EmailMultiAlternatives(
            subject="Verify Your Email - Escala Plants & Nursery",
            body=text_body,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[user.email]
        )
        email.attach_alternative(html_body, "text/html")
        email.send(fail_silently=False)
        
        logger.info(f"✅ Verification email sent to {user.email}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to send verification email to {user.email}: {str(e)}")
        return False


def register_view(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.email = form.cleaned_data['email'].lower()
            user.username = user.email.split('@')[0]
            user.role = "client"  # Explicitly set role to client
            user.is_active = False  # Set to False until email verified
            user.save()

            uid = urlsafe_base64_encode(force_bytes(user.pk))
            token = default_token_generator.make_token(user)
            domain = "lanzofields.capstoneph.com"
            verification_link = f"https://{domain}/verify/{uid}/{token}/"

            try:
                body = render_to_string("dashboard/verify_email_template.html", {
                    "user": user,
                    "verification_link": verification_link
                })
            except Exception as e:
                print("[Template Error]", e)
                body = f"<p>Hello {user.username},</p><p>Verify here: <a href='{verification_link}'>Click here</a></p>"

            send_verification_email("Verify Your Email - Escala Plants & Nursery", body, user.email)
            return render(request, "dashboard/register.html", {"form": RegisterForm(), "success": True})
        else:
            print("[DEBUG] Form errors:", form.errors)
            return render(request, "dashboard/register.html", {"form": form})
    else:
        return render(request, "dashboard/register.html", {"form": RegisterForm()})

def login_view(request):
    if request.method == "POST":
        email = request.POST.get("email", "").strip().lower()
        password = request.POST.get("password")

        user = authenticate(request, email=email, password=password)

        if user is not None:
            if not user.is_active:
                return render(request, "dashboard/login.html", {
                    "error": "not_verified"
                })
            login(request, user)
            if user.role == "admin":
                return redirect("admin_dashboard")
            else:
                return redirect("client_dashboard")
        else:
            return render(request, "dashboard/login.html", {
                "error": "invalid_credentials"
            })

    return render(request, "dashboard/login.html")

def verify_email_view(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = CustomUser.objects.get(pk=uid)
        if user and default_token_generator.check_token(user, token):
            user.is_active = True
            user.save()
            return render(request, "dashboard/verify_email.html", {
                "verified": True
            })
        else:
            return render(request, "dashboard/verify_email.html", {
                "verified": False,
                "error": "invalid_token"
            })
    except (TypeError, ValueError, OverflowError, CustomUser.DoesNotExist):
        return render(request, "dashboard/verify_email.html", {
            "verified": False,
            "error": "invalid_request"
        })

    return render(request, "dashboard/login.html")

def login_view(request):
    """User login view - checks email verification status"""
    if request.method == "POST":
        email = request.POST.get("email", "").strip().lower()
        password = request.POST.get("password")

        user = authenticate(request, email=email, password=password)

        if user is not None:
            if not user.is_active:
                logger.warning(f"Login attempt by unverified user: {email}")
                return render(request, "dashboard/login.html", {
                    "error": "not_verified",
                    "message": "Please verify your email before logging in."
                })
            
            login(request, user)
            
            # Route based on role (should always be 'client' for registered users)
            if user.role == "admin":
                logger.info(f"Admin user logged in: {email}")
                return redirect("admin_dashboard")
            else:
                logger.info(f"Client user logged in: {email}")
                return redirect("client_dashboard")
        else:
            logger.warning(f"Failed login attempt for email: {email}")
            return render(request, "dashboard/login.html", {
                "error": "invalid_credentials",
                "message": "Invalid email or password."
            })

    return render(request, "dashboard/login.html")


def verify_email_view(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = CustomUser.objects.get(pk=uid)
        if user and default_token_generator.check_token(user, token):
            user.is_active = True
            user.save()
            return render(request, "dashboard/verify_email.html", {
                "verified": True
            })
        else:
            return render(request, "dashboard/verify_email.html", {
                "verified": False,
                "error": "invalid_token"
            })
    except (TypeError, ValueError, OverflowError, CustomUser.DoesNotExist):
        return render(request, "dashboard/verify_email.html", {
            "verified": False,
            "error": "invalid_request"
        })




# ... existing code ...


def user_list(request):
    if request.method == "POST" and "delete_user_id" in request.POST:
        user_id = request.POST["delete_user_id"]
        user = get_object_or_404(CustomUser, id=user_id)
        user.delete()
        messages.success(request, "User deleted successfully!")
        return redirect("user_list")

    users = CustomUser.objects.all()
    return render(request, "dashboard/user_list.html", {"users": users})

# ... existing code ...

def logout_view(request):
    logout(request)
    return redirect('login')

def home_view(request):
    return render(request, "dashboard/home.html")

# ... existing code ...

@login_required
def admin_dashboard(request):
    if request.user.role != "admin":
        return redirect('home')

    User = get_user_model()
    total_users = User.objects.count()

    plants = Plant.objects.all()
    total_plants = plants.count()
    healthy_plants = plants.filter(health_status="good").count()
    unhealthy_plants = plants.exclude(health_status__in=["good", "undetected"]).count()

    disease_counts = Counter(plant.health_status for plant in plants if plant.health_status not in ["good", "undetected"])
    disease_labels = list(disease_counts.keys())
    disease_values = list(disease_counts.values())

    return render(request, "dashboard/admin_dashboard.html", {
        'total_users': total_users,
        'total_plants': total_plants,
        'total_plants': total_plants,
        'healthy_plants': healthy_plants,
        'unhealthy_plants': unhealthy_plants,
        'disease_labels': disease_labels,
        'disease_values': disease_values,
    })

# ... existing code ...

@login_required
def client_dashboard(request):
    if request.user.role != "client":
        return redirect('home')

    plants = Plant.objects.all()
    total_plants = plants.count()
    healthy_plants = plants.filter(health_status='good').count()
    unhealthy_plants = plants.exclude(health_status='good').count()

    context = {
        'total_plants': total_plants,
        'healthy_plants': healthy_plants,
        'unhealthy_plants': unhealthy_plants,
        'username': request.user.username,
    }

    return render(request, "dashboard/client_dashboard.html", context)

# ... existing code ...

def update_user_view(request, user_id):
    user = get_object_or_404(CustomUser, id=user_id)
    if request.method == "POST":
        form = RegisterForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            messages.success(request, "User updated successfully!")
            return redirect('user_list')
    else:
        form = RegisterForm(instance=user)

    return render(request, "dashboard/update_user.html", {"form": form, "user": user})

# ... existing code ...

class CustomPasswordResetView(PasswordResetView):
    template_name = "dashboard/password_reset.html"
    email_template_name = "dashboard/password_reset_email.txt"
    html_email_template_name = "dashboard/password_reset_email.html"
    subject_template_name = "dashboard/password_reset_subject.txt"
    success_url = reverse_lazy("password_reset_done")

class CustomPasswordResetDoneView(PasswordResetDoneView):
    template_name = "dashboard/password_reset_done.html"

class CustomPasswordResetConfirmView(PasswordResetConfirmView):
    template_name = "dashboard/password_reset_confirm.html"
    success_url = reverse_lazy("password_reset_complete")

class CustomPasswordResetCompleteView(PasswordResetCompleteView):
    template_name = "dashboard/password_reset_complete.html"

# ... existing code ...


from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.shortcuts import render
from dashboard.models import Plant, TreeAnalysis

@login_required
def plant_inventory(request):
    # 🔹 Get all plants (ordered)
    plants_list = Plant.objects.all().order_by('plant_id')
    paginator = Paginator(plants_list, 5)
    page = request.GET.get('page')

    try:
        plants = paginator.page(page)
    except PageNotAnInteger:
        plants = paginator.page(1)
    except EmptyPage:
        plants = paginator.page(paginator.num_pages)

    # 🔹 For each plant, try to fetch the latest TreeAnalysis — whether linked or not
    for plant in plants:
        # Try linked first
        latest_analysis = TreeAnalysis.objects.filter(plant=plant).order_by('-id').first()

        # If not linked, try to find by name pattern like "Plant 27"
        if not latest_analysis:
            latest_analysis = (
                TreeAnalysis.objects
                .filter(name__icontains=f"Plant {plant.plant_id}")
                .order_by('-id')
                .first()
            )

        if latest_analysis:
            # 💾 Use latest actual TreeAnalysis values (not recalculated)
            plant.detection_details = {
                'healthy': round(latest_analysis.healthy_percentage or 0, 1),
                'dried_leaf': round(latest_analysis.dried_leaf_percentage or 0, 1),
                'leaf_rust': round(latest_analysis.leaf_rust_percentage or 0, 1),
                'powdery_mildew': round(latest_analysis.powdery_mildew_percentage or 0, 1),
            }

            # Determine health status
            percentages = {
                'healthy': latest_analysis.healthy_percentage,
                'dried leaf': latest_analysis.dried_leaf_percentage,
                'leaf rust': latest_analysis.leaf_rust_percentage,
                'powdery mildew': latest_analysis.powdery_mildew_percentage,
            }

            most_likely = max(percentages, key=percentages.get)
            max_val = round(percentages[most_likely], 1)

            plant.health_status = 'good' if most_likely == 'healthy' else most_likely
            plant.status_percentage = max_val

            print(f"[DEBUG] Plant {plant.plant_id} using TreeAnalysis ID {latest_analysis.id} → {plant.health_status} ({plant.status_percentage}%)")

        else:
            # No analysis found
            plant.health_status = 'undetected'
            plant.status_percentage = None
            plant.detection_details = None
            print(f"[DEBUG] Plant {plant.plant_id} → No TreeAnalysis found")

    return render(request, 'dashboard/inventory.html', {'plants': plants})




@login_required
def add_plant(request):
    storage = get_messages(request)
    for _ in storage:
        pass

    if request.method == "POST":
        form = PlantForm(request.POST)
        if form.is_valid():
            plant = form.save(commit=False)
            plant.user = request.user
            plant.health_status = "undetected"
            plant.save()
            messages.success(request, "✅ Plant added successfully!")
            return redirect('add_plant')
        else:
            messages.error(request, "❌ Error adding plant. Please check your inputs.")
    else:
        form = PlantForm()

    return render(request, 'dashboard/add_plant.html', {
        'form': form,
        'health_status': "undetected"
    })

# ... existing code ...

@login_required
def update_plant(request, plant_id):
    plant = get_object_or_404(Plant, plant_id=plant_id)
    if request.method == "POST":
        form = PlantForm(request.POST, instance=plant)
        if form.is_valid():
            form.save()
            messages.success(request, "✅ Plant updated successfully!")
            return redirect('inventory')
    else:
        form = PlantForm(instance=plant)

    return render(request, 'dashboard/update_plant.html', {'form': form, 'plant': plant})

# ... existing code ...

def delete_plant(request, plant_id):
    plant = get_object_or_404(Plant, plant_id=plant_id)
    plant.delete()
    messages.success(request, "✅ Plant deleted successfully!")
    return redirect('inventory')

# ... existing code ...

from django.core.paginator import Paginator
from django.shortcuts import render
from dashboard.models import Plant, TreeAnalysis

@login_required
def track_plant_health(request):
    unhealthy_plants_data = []

    # 🔹 Loop through all plants
    for plant in Plant.objects.all().order_by('plant_id'):
        # Try linked TreeAnalysis
        latest_analysis = TreeAnalysis.objects.filter(plant=plant).order_by('-id').first()

        # If not linked, try match by name (e.g. "Plant 27")
        if not latest_analysis:
            latest_analysis = (
                TreeAnalysis.objects
                .filter(name__icontains=f"Plant {plant.plant_id}")
                .order_by('-id')
                .first()
            )

        if latest_analysis:
            # Skip healthy ones, only keep diseased plants
            diseased = any([
                latest_analysis.dried_leaf_percentage > 0,
                latest_analysis.leaf_rust_percentage > 0,
                latest_analysis.powdery_mildew_percentage > 0
            ])

            if diseased:
                plant.detection_details = {
                    'dried_leaf_percentage': round(latest_analysis.dried_leaf_percentage, 1),
                    'leaf_rust_percentage': round(latest_analysis.leaf_rust_percentage, 1),
                    'powdery_mildew_percentage': round(latest_analysis.powdery_mildew_percentage, 1),
                    'total_leaves': latest_analysis.total_leaves,
                    'healthy_percentage': round(latest_analysis.healthy_percentage, 1)
                }
                unhealthy_plants_data.append(plant)

    # 🔹 Paginate results
    paginator = Paginator(unhealthy_plants_data, 5)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, "dashboard/track_plant_health.html", {
        "unhealthy_plants": page_obj,
        "page_obj": page_obj
    })

# ... existing code ...

@login_required
def reports_view(request):
    User = get_user_model()
    
    total_users = User.objects.count()
    active_users = User.objects.filter(is_active=True).count()
    
    total_plants = Plant.objects.count()
    healthy_plants = Plant.objects.filter(health_status="good").count()
    unhealthy_plants = Plant.objects.exclude(health_status="good").count()

    context = {
        "total_users": total_users,
        "active_users": active_users,
        "total_plants": total_plants,
        "healthy_plants": healthy_plants,
        "unhealthy_plants": unhealthy_plants
    }

    return render(request, "dashboard/reports.html", context)

# ... existing code ...

def export_csv(request):
    if request.method == "GET":
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="report.csv"'

        writer = csv.writer(response)

        if "export_users" in request.GET:
            writer.writerow(["User ID", "Username", "Email", "Role", "Active"])
            users = CustomUser.objects.all().values_list("id", "username", "email", "role", "is_active")
            for user in users:
                writer.writerow(user)

        if "export_total_plants" in request.GET:
            writer.writerow(["Total Plants"])
            writer.writerow([Plant.objects.count()])

        if "export_healthy_plants" in request.GET:
            writer.writerow(["Plant ID", "Age", "Health Status", "Symptoms"])
            healthy_plants = Plant.objects.filter(health_status="good").values_list("plant_id", "age", "health_status", "symptoms")
            for plant in healthy_plants:
                writer.writerow(plant)

        if "export_unhealthy_plants" in request.GET:
            writer.writerow(["Plant ID", "Age", "Health Status", "Symptoms"])
            unhealthy_plants = Plant.objects.exclude(health_status="good").values_list("plant_id", "age", "health_status", "symptoms")
            for plant in unhealthy_plants:
                writer.writerow(plant)

        return response
    else:
        return HttpResponse("Invalid request", status=400)

# ... existing code ...

def export_pdf(request):
    if request.method == "POST":
        response = HttpResponse(content_type="application/pdf")
        response["Content-Disposition"] = 'attachment; filename="report.pdf"'
        
        pdf = canvas.Canvas(response, pagesize=letter)
        pdf.setTitle("Report")
        width, height = letter
        y_position = height - 40

        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(200, y_position, "Escala Plants - Reports")
        y_position -= 40

        if "export_users" in request.POST:
            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(30, y_position, "Users List")
            y_position -= 20

            users = CustomUser.objects.all().values_list("id", "username", "email", "role", "is_active")
            pdf.setFont("Helvetica", 10)
            for user in users:
                pdf.drawString(30, y_position, f"ID: {user[0]}, Username: {user[1]}, Email: {user[2]}, Role: {user[3]}, Active: {user[4]}")
                y_position -= 15
            
            y_position -= 20

        if "export_total_plants" in request.POST:
            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(30, y_position, "Total Plants")
            y_position -= 15
            pdf.setFont("Helvetica", 10)
            pdf.drawString(30, y_position, f"Total Plants: {Plant.objects.count()}")
            y_position -= 20

        if "export_healthy_plants" in request.POST:
            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(30, y_position, "Healthy Plants")
            y_position -= 15
            pdf.setFont("Helvetica", 10)
            healthy_plants = Plant.objects.filter(health_status="good").values_list("plant_id", "age", "health_status", "symptoms")
            for plant in healthy_plants:
                pdf.drawString(30, y_position, f"Plant ID: {plant[0]},  Age: {plant[1]}, Status: {plant[2]}, Symptoms: {plant[3]}")
                y_position -= 15
            y_position -= 20

        if "export_unhealthy_plants" in request.POST:
            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(30, y_position, "Unhealthy Plants")
            y_position -= 15
            pdf.setFont("Helvetica", 10)
            unhealthy_plants = Plant.objects.exclude(health_status="good").values_list("plant_id", "age", "health_status", "symptoms")
            for plant in unhealthy_plants:
                pdf.drawString(30, y_position, f"Plant ID: {plant[0]},  Age: {plant[1]}, Status: {plant[2]}, Symptoms: {plant[3]}")
                y_position -= 15

        pdf.save()
        return response
    else:
        return HttpResponse("Invalid request", status=400)

# <CHANGE> Updated YOLO model loading with caching and lazy import
CLASS_NAMES = ['dried leaf', 'healthy', 'leaf rust', 'powdery mildew']

def load_yolo_model():
    """Load YOLO model with caching, better error messages and fallback hints."""
    global _MODEL_CACHE

    if 'yolo_model' in _MODEL_CACHE:
        logger.info("✅ Using cached YOLO model")
        return _MODEL_CACHE['yolo_model']

    model_path = os.path.join(settings.MEDIA_ROOT, 'best.pt')
    if not os.path.exists(model_path):
        logger.error(f"❌ YOLO model file not found at: {model_path}")
        return None

    try:
        # lazy import: will import ultralytics which imports cv2 internally
        from ultralytics import YOLO

        logger.info("🔄 Loading YOLO model for the first time...")
        model = YOLO(model_path)

        _MODEL_CACHE['yolo_model'] = model
        logger.info("✅ YOLO model loaded and cached.")
        try:
            logger.info(f"YOLO classes: {model.names}")
        except Exception:
            logger.debug("Could not retrieve model.names")
        return model

    except ImportError as ie:
        # Most likely cv2 system lib missing or opencv not installed
        logger.error(f"ImportError while loading YOLO (likely cv2/system libs): {ie}")
        logger.error("Suggested fixes: 1) Install libGL (apt-get install -y libgl1 libglib2.0-0) "
                     "or 2) use opencv-python-headless in requirements.txt")
        traceback.print_exc()
        return None

    except Exception as e:
        logger.error(f"❌ Error loading YOLO model: {e}")
        traceback.print_exc()
        return None


# ... existing code ...

def preprocess_image(image):
    """Preprocess the image for prediction"""
    try:
        logger.debug(f"Original image size: {image.size}, mode: {image.mode}")
        
        image = image.resize((150, 150))
        logger.debug(f"Resized image size: {image.size}")
        
        if image.mode != 'RGB':
            logger.debug(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        img_array = np.array(image)
        logger.debug(f"Image array shape: {img_array.shape}, dtype: {img_array.dtype}")
        
        img_array = img_array.astype(np.float32) / 255.0
        logger.debug(f"Normalized array min: {img_array.min()}, max: {img_array.max()}")
        
        img_array = np.expand_dims(img_array, axis=0)
        logger.debug(f"Final array shape: {img_array.shape}")
        
        return img_array
    except Exception as e:
        logger.error(f"Error in preprocess_image: {e}")
        traceback.print_exc()
        raise

# ... existing code ...

ALLOWED_CLASSES = {"healthy", "dried leaf", "leaf rust", "powdery mildew"}
CONF_THRESHOLD = 0.50

@csrf_exempt
def predict(request):
    if request.method != 'POST':
        return JsonResponse({"success": False, "error": "Invalid request method"})

    # <CHANGE> Load model using cached function (lazy loads YOLO)
    model = load_yolo_model()
    if model is None:
        return JsonResponse({"success": False, "error": "YOLO model not loaded"})

    try:
        # <CHANGE> Lazy import cv2 only when needed
        import cv2
        
        frame_file = request.FILES.get('frame')
        plant_id = request.POST.get("plant_id")

        if not frame_file:
            return JsonResponse({"success": False, "error": "No frame received"})

        file_bytes = np.frombuffer(frame_file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results = model.track(frame, tracker="bytetrack.yaml", persist=True)[0]

        detections = []

        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else None
                class_name = model.names[cls].lower().replace('-', ' ')

                if class_name not in ALLOWED_CLASSES or conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "id": track_id,
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class": class_name
                })

        if plant_id:
            from dashboard.models import Plant
            plant = Plant.objects.get(plant_id=plant_id)

            analysis, created = TreeAnalysis.objects.get_or_create(
                plant=plant,
                defaults={"name": f"Analysis for Plant {plant.plant_id}"}
            )

            for det in detections:
                LeafImage.objects.create(
                    image=None,
                    prediction=det["class"].title(),
                    healthy_confidence=det["confidence"] if det["class"] == "healthy" else 0,
                    dried_leaf_confidence=det["confidence"] if det["class"] == "dried leaf" else 0,
                    leaf_rust_confidence=det["confidence"] if det["class"] == "leaf rust" else 0,
                    powdery_mildew_confidence=det["confidence"] if det["class"] == "powdery mildew" else 0,
                    tree_analysis=analysis
                )

            analysis.calculate_health()
            analysis.is_completed = True
            analysis.save()

            plant.tree_analysis = analysis
            plant.health_status = "good" if analysis.overall_health > 70 else "leaf rust"
            plant.save()

        return JsonResponse({
            "success": True,
            "detections": detections,
            "overall_health": analysis.overall_health if plant_id else None,
            "healthy_percentage": analysis.healthy_percentage if plant_id else None,
            "leaf_rust_percentage": analysis.leaf_rust_percentage if plant_id else None,
        })

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})

# ... existing code ...

@login_required
def detector(request):
    plant_id = request.GET.get('plant_id')
    user_role = request.user.role

    if plant_id:
        url = reverse('new_tree_analysis')
        return redirect(f'{url}?plant_id={plant_id}&role={user_role}')

    # <CHANGE> Load model using cached function
    model = load_yolo_model()
    context = {}

    if model is None:
        context['model_error'] = True
        model_path = os.path.join(settings.MEDIA_ROOT, 'best.pt')
        context['model_path'] = model_path
        context['model_exists'] = os.path.exists(model_path)

    context['recent_analyses'] = TreeAnalysis.objects.filter(is_completed=True).order_by('-completed_at')[:5]
    context['user_role'] = user_role

    return render(request, 'dashboard/detector.html', context)

# ... existing code ...

def new_tree_analysis(request):
    plant_id = request.GET.get("plant_id")
    
    if plant_id:
        try:
            plant = Plant.objects.get(plant_id=plant_id)
            if hasattr(plant, 'tree_analysis') and plant.tree_analysis:
                existing_analysis = plant.tree_analysis
                url = reverse('tree_analysis', args=[existing_analysis.id])
                url += f'?plant_id={plant_id}'
                return redirect(url)
            
            tree_analysis = TreeAnalysis.objects.create(
                name=f"Tree Analysis for Plant {plant_id}",
                plant=plant,
                plant_id=plant_id
            )
        except Plant.DoesNotExist:
            tree_analysis = TreeAnalysis.objects.create(
                name="New Tree Analysis",
                plant_id=plant_id
            )
    else:
        tree_analysis = TreeAnalysis.objects.create(name="New Tree Analysis")

    url = reverse('tree_analysis', args=[tree_analysis.id])
    if plant_id:
        url += f'?plant_id={plant_id}'

    return redirect(url)

# ... existing code ...

@login_required
def tree_analysis(request, analysis_id=None):
    """View for tree analysis page"""
    if analysis_id:
        tree_analysis = get_object_or_404(TreeAnalysis, id=analysis_id)
        plant_id = request.GET.get('plant_id') or tree_analysis.plant_id
        if plant_id and not tree_analysis.plant_id:
            tree_analysis.plant_id = plant_id
            tree_analysis.save()
    else:
        plant_id = request.GET.get('plant_id')
        if plant_id:
            return redirect(f"{reverse('new_tree_analysis')}?plant_id={plant_id}")
        return redirect('new_tree_analysis')
    
    leaf_images = tree_analysis.leaf_images.all()
    
    context = {
        'tree_analysis': tree_analysis,
        'leaf_images': leaf_images,
        'plant_id': plant_id,
    }
    
    return render(request, 'dashboard/tree_analysis.html', context)

# ... existing code ...

@csrf_exempt
@require_POST
def complete_analysis(request, analysis_id):
    """Complete a tree analysis and calculate health"""
    try:
        logger.info(f"Completing analysis for ID: {analysis_id}")
        logger.info(f"POST data: {request.POST}")
        
        tree_analysis = get_object_or_404(TreeAnalysis, id=analysis_id)
        logger.info(f"Found tree analysis: {tree_analysis}")
        
        tree_name = request.POST.get('tree_name')
        if tree_name:
            tree_analysis.name = tree_name
        
        healthy_count = int(request.POST.get('healthy_count', 0))
        dried_leaf_count = int(request.POST.get('dried_leaf_count', 0))
        powdery_mildew_count = int(request.POST.get('powdery_mildew_count', 0))
        leaf_rust_count = int(request.POST.get('leaf_rust_count', 0))
        
        tree_analysis.leaf_images.all().delete()
        
        for i in range(healthy_count):
            LeafImage.objects.create(
                tree_analysis=tree_analysis,
                prediction='Healthy',
                healthy_confidence=95.0,
                dried_leaf_confidence=2.0,
                powdery_mildew_confidence=2.0,
                leaf_rust_confidence=1.0
            )
        
        for i in range(dried_leaf_count):
            LeafImage.objects.create(
                tree_analysis=tree_analysis,
                prediction='Dried Leaf',
                healthy_confidence=5.0,
                dried_leaf_confidence=90.0,
                powdery_mildew_confidence=3.0,
                leaf_rust_confidence=2.0
            )
        
        for i in range(powdery_mildew_count):
            LeafImage.objects.create(
                tree_analysis=tree_analysis,
                prediction='Powdery Mildew',
                healthy_confidence=5.0,
                dried_leaf_confidence=5.0,
                powdery_mildew_confidence=85.0,
                leaf_rust_confidence=5.0
            )
        
        for i in range(leaf_rust_count):
            LeafImage.objects.create(
                tree_analysis=tree_analysis,
                prediction='Leaf Rust',
                healthy_confidence=5.0,
                dried_leaf_confidence=10.0,
                powdery_mildew_confidence=5.0,
                leaf_rust_confidence=80.0
            )
        
        logger.info(f"Created {healthy_count + dried_leaf_count + powdery_mildew_count + leaf_rust_count} LeafImage records")
        
        overall_health = tree_analysis.calculate_health()
        
        tree_analysis.is_completed = True
        tree_analysis.completed_at = timezone.now()
        tree_analysis.save()
        
        plant_id = request.POST.get('plant_id') or tree_analysis.plant_id
        if plant_id:
            try:
                plant = Plant.objects.get(plant_id=plant_id)
                plant.tree_analysis = tree_analysis
                if overall_health >= 80:
                    plant.health_status = "good"
                elif tree_analysis.powdery_mildew_percentage > 30:
                    plant.health_status = "amag"
                elif tree_analysis.leaf_rust_percentage > 20:
                    plant.health_status = "leaf rust"
                elif tree_analysis.dried_leaf_percentage > 40:
                    plant.health_status = "dahon"
                else:
                    plant.health_status = "good"
                plant.save()
                logger.info(f"Updated plant {plant_id} health to {plant.health_status}")
            except Plant.DoesNotExist:
                logger.warning(f"Plant ID {plant_id} not found.")

        return JsonResponse({
            'success': True,
            'tree_analysis_id': tree_analysis.id,
            'healthy_percentage': tree_analysis.healthy_percentage,
            'dried_leaf_percentage': tree_analysis.dried_leaf_percentage,
            'leaf_rust_percentage': tree_analysis.leaf_rust_percentage,
            'powdery_mildew_percentage': tree_analysis.powdery_mildew_percentage,
            'overall_health': overall_health,
            'total_detections': healthy_count + dried_leaf_count + powdery_mildew_count + leaf_rust_count,
            'plant_updated': bool(plant_id)
        })

    except Exception as e:
        logger.error(f"Error completing analysis: {e}")
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

# ... existing code ...

@csrf_exempt
def save_analysis(request):
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "Invalid request"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
        analysis_id = data.get("analysis_id")
        plant_id = data.get("plant_id")

        plant = Plant.objects.filter(pk=plant_id).first() if plant_id else None

        analysis, created = TreeAnalysis.objects.get_or_create(
            id=analysis_id if analysis_id else None,
            defaults={
                'plant': plant,
                'name': data.get("tree_name", "Unnamed Tree"),
                'notes': data.get("notes", ""),
                'healthy_percentage': data.get("healthy_percentage", 0),
                'overall_health': data.get("overall_health", 0),
            }
        )

        if not created:
            analysis.plant = plant
            analysis.name = data.get("tree_name", analysis.name)
            analysis.notes = data.get("notes", analysis.notes)
            analysis.healthy_percentage = data.get("healthy_percentage", analysis.healthy_percentage)
            analysis.overall_health = data.get("overall_health", analysis.overall_health)
            analysis.save()

        if plant:
            plant.health_status = (
                "good" if analysis.overall_health > 70 else
                "leaf rust" if analysis.leaf_rust_percentage > 0 else
                "amag" if analysis.powdery_mildew_percentage > 0 else
                "dahon" if hasattr(analysis, "dried_leaf_percentage") and analysis.dried_leaf_percentage > 0 else
                "undetected"
            )
            plant.tree_analysis = analysis
            plant.save()

        return JsonResponse({
            "success": True,
            "analysis_id": analysis.id,
            "plant_id": plant_id,
            "overall_health": analysis.overall_health,
            "healthy_percentage": analysis.healthy_percentage
        }, status=201)

    except Exception as e:
        return JsonResponse({"success": False, "message": str(e)}, status=400)

# ... existing code ...

@csrf_exempt
@require_POST
def remove_leaf(request, leaf_id):
    """Remove a leaf image from a tree analysis"""
    try:
        leaf_image = get_object_or_404(LeafImage, id=leaf_id)
        tree_analysis = leaf_image.tree_analysis
        
        if leaf_image.image:
            if os.path.isfile(leaf_image.image.path):
                os.remove(leaf_image.image.path)
        
        leaf_image.delete()
        
        return JsonResponse({
            'success': True,
            'leaf_count': tree_analysis.leaf_images.count()
        })

    except Exception as e:
        logger.error(f"Error removing leaf: {e}")
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': f'Error removing leaf: {str(e)}'
        })

# ... existing code ...

@csrf_exempt
@require_POST
def clear_leaves(request, analysis_id):
    """Clear all leaf images from a tree analysis"""
    try:
        tree_analysis = get_object_or_404(TreeAnalysis, id=analysis_id)
        
        leaf_images = tree_analysis.leaf_images.all()
        
        for leaf_image in leaf_images:
            if leaf_image.image:
                if os.path.isfile(leaf_image.image.path):
                    os.remove(leaf_image.image.path)
        
        leaf_images.delete()
        
        return JsonResponse({
            'success': True
        })

    except Exception as e:
        logger.error(f"Error clearing leaves: {e}")
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': f'Error clearing leaves: {str(e)}'
        })

# ... existing code ...

@login_required
def history(request):
    """View for analysis history"""
    analyses = TreeAnalysis.objects.filter(is_completed=True).order_by('-completed_at')
    
    context = {
        'analyses': analyses
    }
    
    return render(request, 'dashboard/history.html', context)

# ... existing code ...

@login_required
def analysis_detail(request, analysis_id):
    """View for detailed analysis results"""
    tree_analysis = get_object_or_404(TreeAnalysis, id=analysis_id)
    leaf_images = tree_analysis.leaf_images.all()
    
    context = {
        'tree_analysis': tree_analysis,
        'leaf_images': leaf_images
    }
    
    return render(request, 'dashboard/analysis_detail.html', context)

# ... existing code ...

@csrf_exempt
@require_POST
def delete_analysis(request, analysis_id):
    """Delete a tree analysis"""
    try:
        tree_analysis = get_object_or_404(TreeAnalysis, id=analysis_id)
        
        leaf_images = tree_analysis.leaf_images.all()
        for leaf_image in leaf_images:
            if leaf_image.image:
                if os.path.isfile(leaf_image.image.path):
                    os.remove(leaf_image.image.path)
        
        tree_analysis.delete()
        
        return JsonResponse({
            'success': True
        })

    except Exception as e:
        logger.error(f"Error deleting analysis: {e}")
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': f'Error deleting analysis: {str(e)}'
        })

# ... existing code ...

@csrf_exempt
@require_POST
def delete_multiple_analyses(request):
    """Delete multiple tree analyses"""
    try:
        data = json.loads(request.body)
        analysis_ids = data.get('ids', [])
        
        if not analysis_ids:
            return JsonResponse({
                'success': False,
                'error': 'No analysis IDs provided'
            })
        
        analyses = TreeAnalysis.objects.filter(id__in=analysis_ids)
        
        for analysis in analyses:
            leaf_images = analysis.leaf_images.all()
            for leaf_image in leaf_images:
                if leaf_image.image:
                    if os.path.isfile(leaf_image.image.path):
                        os.remove(leaf_image.image.path)
        
        analyses.delete()
        
        return JsonResponse({
            'success': True,
            'count': len(analysis_ids)
        })

    except Exception as e:
        logger.error(f"Error deleting multiple analyses: {e}")
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': f'Error deleting analyses: {str(e)}'
        })

# ... existing code ...

def export_analysis_pdf(request, analysis_id):
    try:
        tree_analysis = TreeAnalysis.objects.get(id=analysis_id)
    except TreeAnalysis.DoesNotExist:
        return HttpResponse("Analysis not found", status=404)

    leaf_images = tree_analysis.leaf_images.all()
    total_leaves = leaf_images.count()

    healthy_count = leaf_images.filter(prediction='Healthy').count()
    dried_leaf_count = leaf_images.filter(prediction='Dried Leaf').count()
    powdery_mildew_count = leaf_images.filter(prediction='Powdery Mildew').count()
    leaf_rust_count = leaf_images.filter(prediction='Leaf Rust').count()

    healthy_percentage = (healthy_count / total_leaves) * 100 if total_leaves else 0
    dried_leaf_percentage = (dried_leaf_count / total_leaves) * 100 if total_leaves else 0
    powdery_mildew_percentage = (powdery_mildew_count / total_leaves) * 100 if total_leaves else 0
    leaf_rust_percentage = (leaf_rust_count / total_leaves) * 100 if total_leaves else 0

    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="tree_analysis_{analysis_id}.pdf"'

    p = canvas.Canvas(response, pagesize=A4)
    width, height = A4
    x = 50
    y = height - 50
    line_height = 20

    p.setFont("Helvetica-Bold", 16)
    p.drawString(x, y, f"Tree Analysis Report: {tree_analysis.name}")
    y -= line_height * 2

    p.setFont("Helvetica", 12)
    p.drawString(x, y, f"Completed At: {tree_analysis.completed_at.strftime('%Y-%m-%d %H:%M') if tree_analysis.completed_at else 'N/A'}")
    y -= line_height
    p.drawString(x, y, f"Total Leaves Analyzed: {total_leaves}")
    y -= line_height * 2

    p.setFont("Helvetica-Bold", 14)
    p.drawString(x, y, "Detection Summary")
    y -= line_height
    p.setFont("Helvetica", 12)

    stats = [
        ('Healthy Leaves', healthy_count, healthy_percentage),
        ('Dried Leaves', dried_leaf_count, dried_leaf_percentage),
        ('Powdery Mildew Leaves', powdery_mildew_count, powdery_mildew_percentage),
        ('Leaf Rust Leaves', leaf_rust_count, leaf_rust_percentage),
    ]

    for label, count, percent in stats:
        p.drawString(x, y, f"{label}: {count} ({percent:.1f}%)")
        y -= line_height

    y -= line_height
    p.setFont("Helvetica-Bold", 12)
    p.drawString(x, y, f"Overall Health: {tree_analysis.overall_health:.1f}%")

    p.showPage()
    p.save()
    return response

# ... existing code ...

def export_all_analyses(request):
    """Export all tree analyses as CSV"""
    try:
        analyses = TreeAnalysis.objects.filter(is_completed=True).order_by('-completed_at')
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="all-tree-analyses.csv"'
        
        writer = csv.writer(response)
        
        writer.writerow(['Lanzones Tree Analyses Report'])
        writer.writerow(['Generated on', timezone.now().strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow(['Total Analyses', analyses.count()])
        writer.writerow([])
        
        writer.writerow(['ID', 'Tree Name', 'Analysis Date', 'Total Leaves', 'Healthy %', 'Dried Leaf %', 'Powdery Mildew %', 'Leaf Rust %', 'Overall Health'])
        
        for analysis in analyses:
            leaf_count = analysis.leaf_images.count()
            writer.writerow([
                analysis.id,
                analysis.name,
                analysis.completed_at.strftime('%Y-%m-%d %H:%M:%S'),
                leaf_count,
                f"{analysis.healthy_percentage:.1f}%",
                f"{analysis.dried_leaf_percentage:.1f}%",
                f"{analysis.powdery_mildew_percentage:.1f}%",
                f"{analysis.leaf_rust_percentage:.1f}%",
                f"{analysis.overall_health:.1f}%"
            ])
        
        return response

    except Exception as e:
        logger.error(f"Error exporting all analyses: {e}")
        traceback.print_exc()
        return HttpResponse(f"Error exporting analyses: {str(e)}", status=500)

# ... existing code ...

def analysis_detail_view(request, analysis_id):
    try:
        analysis = TreeAnalysis.objects.get(id=analysis_id)
    except TreeAnalysis.DoesNotExist:
        raise Http404("Analysis not found")

    leaf_images = analysis.leaf_images.all()
    total_leaves = leaf_images.count()

    healthy_count = leaf_images.filter(prediction='Healthy').count()
    dried_leaf_count = leaf_images.filter(prediction='Dried Leaf').count()
    powdery_mildew_count = leaf_images.filter(prediction='Powdery Mildew').count()
    leaf_rust_count = leaf_images.filter(prediction='Leaf Rust').count()

    healthy_percentage = (healthy_count / total_leaves) * 100 if total_leaves else 0
    dried_leaf_percentage = (dried_leaf_count / total_leaves) * 100 if total_leaves else 0
    powdery_mildew_percentage = (powdery_mildew_count / total_leaves) * 100 if total_leaves else 0
    leaf_rust_percentage = (leaf_rust_count / total_leaves) * 100 if total_leaves else 0

    leaf_detections = [
        {
            'prediction': leaf.prediction or 'Unknown',
            'image_url': leaf.image.url if leaf.image else '',
            'healthy_confidence': leaf.healthy_confidence,
            'dried_leaf_confidence': leaf.dried_leaf_confidence,
            'powdery_mildew_confidence': leaf.powdery_mildew_confidence,
            'leaf_rust_confidence': leaf.leaf_rust_confidence
        }
        for leaf in leaf_images
    ]

    return JsonResponse({
        'success': True,
        'analysis': {
            'id': analysis.id,
            'name': analysis.name,
            'completed_at': analysis.completed_at.isoformat() if analysis.completed_at else None,
            'overall_health': analysis.overall_health,
            'total_leaf_count': total_leaves,
            'healthy_count': healthy_count,
            'dried_leaf_count': dried_leaf_count,
            'powdery_mildew_count': powdery_mildew_count,
            'leaf_rust_count': leaf_rust_count,
            'healthy_percentage': healthy_percentage,
            'dried_leaf_percentage': dried_leaf_percentage,
            'powdery_mildew_percentage': powdery_mildew_percentage,
            'leaf_rust_percentage': leaf_rust_percentage,
        },
        'leaf_detections': leaf_detections
    })

# ... existing code ...

from django.contrib.auth.decorators import login_required
from .models import TreeAnalysis, PestDetectionSession

@login_required
@login_required
def history_view(request):
    tree_analyses = TreeAnalysis.objects.filter(is_completed=True).order_by('-completed_at')
    pest_sessions = PestDetectionSession.objects.all().order_by('-created_at')

    analyses = []

    # 🟢 Tree Analyses (use saved DB values)
    for analysis in tree_analyses:
        diseased_count = (
            analysis.dried_leaf_count +
            analysis.leaf_rust_count +
            analysis.powdery_mildew_count
        )
        diseased_percentage = (
            (diseased_count / analysis.total_leaves) * 100
            if analysis.total_leaves > 0 else 0
        )

        analyses.append({
            'id': analysis.id,
            'name': analysis.name,
            'created_at': analysis.created_at,
            'completed_at': analysis.completed_at,
            'overall_health': analysis.overall_health,
            'healthy_count': analysis.healthy_count,
            'dried_leaf_count': analysis.dried_leaf_count,
            'leaf_rust_count': analysis.leaf_rust_count,
            'powdery_mildew_count': analysis.powdery_mildew_count,
            'healthy_percentage': analysis.healthy_percentage,
            'dried_leaf_percentage': analysis.dried_leaf_percentage,
            'leaf_rust_percentage': analysis.leaf_rust_percentage,
            'powdery_mildew_percentage': analysis.powdery_mildew_percentage,
            'total_leaves': analysis.total_leaves,
            'diseased_count': diseased_count,  # ✅ ADD THIS
            'diseased_percentage': diseased_percentage,  # ✅ AND THIS
            'type': 'tree_analysis',
        })

    # 🐛 Pest Detection Sessions
    for session in pest_sessions:
        analyses.append({
            'id': session.id,
            'name': session.session_name,
            'completed_at': session.completed_at,
            'total_processed': session.total_processed,
            'no_pest_count': session.no_pest_count,
            'pest_count': session.pest_count,
            'high_risk_count': session.high_risk_count,
            'uncertain_count': session.uncertain_count,
            'avg_confidence': session.avg_confidence,
            'avg_processing_time': session.avg_processing_time,
            'type': 'pest_detection'
        })

    analyses.sort(key=lambda x: x['completed_at'], reverse=True)

    return render(request, 'dashboard/history.html', {'analyses': analyses})


# ... existing code ...

PEST_CLASS_NAMES = ['Adristyrannus', 'Aphids', 'Beetle', 'Bugs', 'Mites', 'Weevil', 'Whitefly']

def preprocess_pest_image(image):
    """Preprocess image for pest detection model"""
    try:
        image = image.resize((224, 224))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        img_array = img_array.astype(np.float32) / 255.0
        
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

# ... existing code ...

# ... existing code at top ...
from .model_loader import load_pest_model, preprocess_image, predict_pest, load_image_from_file

# The model_loader.py has better error handling and batch_shape compatibility


@login_required
def pest_detector(request):
    """Main pest detection page"""
    context = {
        'user_role': request.user.role
    }
    return render(request, 'dashboard/pest_detector.html', context)


@csrf_exempt
@require_POST
def pest_predict(request):
    """API endpoint for pest prediction"""
    try:
        model = load_pest_model()
        if model is None:
            return JsonResponse({
                'success': False, 
                'error': 'Pest detection model not found. Please ensure improved_pest_model.h5 or improved_pest_model.keras is in media folder.'
            })

        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'success': False, 'error': 'No image provided'})

        image = load_image_from_file(image_file)
        if image is None:
            return JsonResponse({'success': False, 'error': 'Failed to load image'})
        
        prediction_result = predict_pest(model, image)
        
        if not prediction_result.get('success', False):
            return JsonResponse(prediction_result)
        
        return JsonResponse({
            'success': True,
            'prediction': prediction_result['predicted_class'],
            'confidence': round(prediction_result['confidence'] * 100, 2),
            'confidence_scores': {
                'Adristyrannus': prediction_result['all_predictions'][0] * 100,
                'Aphids': prediction_result['all_predictions'][1] * 100,
                'Beetle': prediction_result['all_predictions'][2] * 100,
                'Bugs': prediction_result['all_predictions'][3] * 100,
                'Mites': prediction_result['all_predictions'][4] * 100,
                'Weevil': prediction_result['all_predictions'][5] * 100,
                'Whitefly': prediction_result['all_predictions'][6] * 100,
            }
        })
        
    except Exception as e:
        logger.error(f"Error in pest prediction: {e}")
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        })

# ... existing code ...

@csrf_exempt
@require_POST
def save_pest_results(request):
    """Save pest detection results to database"""
    try:
        data = json.loads(request.body)
        results = data.get('results', [])
        summary = data.get('summary', {})
        
        if not results:
            return JsonResponse({'success': False, 'error': 'No results to save'})
        
        session = PestDetectionSession.objects.create(
            user=request.user,
            session_name=f"Pest Detection - {timezone.now().strftime('%Y-%m-%d %H:%M')}",
            total_processed=summary.get('total_processed', 0),
            no_pest_count=summary.get('no_pest_count', 0),
            pest_count=summary.get('pest_count', 0),
            high_risk_count=summary.get('high_risk_count', 0),
            uncertain_count=summary.get('uncertain_count', 0),
            avg_processing_time=float(summary.get('avg_processing_time', 0)),
            avg_confidence=float(summary.get('avg_confidence', 0))
        )
        
        for result in results:
            confidence_scores = result.get('confidence_scores', {})
            
            PestDetectionResult.objects.create(
                session=session,
                filename=result.get('filename', ''),
                prediction=result.get('prediction', ''),
                original_prediction=result.get('original_prediction', ''),
                confidence=float(result.get('confidence', 0)),
                processing_time=float(result.get('processing_time', 0)),
                is_low_confidence=result.get('is_low_confidence', False),
                timestamp=timezone.now(),
                adristyrannus_confidence=float(confidence_scores.get('Adristyrannus', 0)),
                aphids_confidence=float(confidence_scores.get('Aphids', 0)),
                beetle_confidence=float(confidence_scores.get('Beetle', 0)),
                bugs_confidence=float(confidence_scores.get('Bugs', 0)),
                mites_confidence=float(confidence_scores.get('Mites', 0)),
                weevil_confidence=float(confidence_scores.get('Weevil', 0)),
                whitefly_confidence=float(confidence_scores.get('Whitefly', 0))
            )
        
        logger.info(f"Pest detection session saved for user {request.user.username}: {session.id}")
        
        return JsonResponse({
            'success': True,
            'message': 'Results saved successfully!',
            'session_id': session.id
        })
        
    except Exception as e:
        logger.error(f"Error saving pest results: {e}")
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': f'Error saving results: {str(e)}'
        })

# ... existing code ...

@csrf_exempt
@require_POST
def delete_pest_session(request, session_id):
    """Delete a pest detection session"""
    try:
        session = get_object_or_404(PestDetectionSession, id=session_id)
        
        session.results.all().delete()
        
        session.delete()
        
        logger.info(f"Pest detection session {session_id} deleted successfully")
        
        return JsonResponse({
            'success': True,
            'message': 'Pest detection session deleted successfully'
        })

    except Exception as e:
        logger.error(f"Error deleting pest session: {e}")
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': f'Error deleting pest session: {str(e)}'
        })

# ... existing code ...

def export_pest_session_csv(request, session_id):
    """Export pest detection session as CSV"""
    try:
        session = get_object_or_404(PestDetectionSession, id=session_id)
        results = session.results.all()
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="pest_session_{session_id}.csv"'
        
        writer = csv.writer(response)
        
        writer.writerow(['Pest Detection Session Report'])
        writer.writerow(['Session Name', session.session_name])
        writer.writerow(['Date', session.created_at.strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow(['Total Processed', session.total_processed])
        writer.writerow(['No Pest Count', session.no_pest_count])
        writer.writerow(['Pest Count', session.pest_count])
        writer.writerow(['High Risk Count', session.high_risk_count])
        writer.writerow(['Uncertain Count', session.uncertain_count])
        writer.writerow(['Average Confidence', f"{session.avg_confidence:.1f}%"])
        writer.writerow(['Average Processing Time', f"{session.avg_processing_time:.2f}s"])
        writer.writerow([])
        
        writer.writerow(['Individual Results'])
        writer.writerow(['Filename', 'Prediction', 'Confidence', 'Processing Time', 'Low Confidence', 'Timestamp'])
        
        for result in results:
            writer.writerow([
                result.filename,
                result.prediction,
                f"{result.confidence:.1f}%",
                f"{result.processing_time:.2f}s",
                'Yes' if result.is_low_confidence else 'No',
                result.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        logger.info(f"Pest detection session {session_id} exported successfully")
        return response

    except Exception as e:
        logger.error(f"Error exporting pest session: {e}")
        traceback.print_exc()
        return HttpResponse(f"Error exporting pest session: {str(e)}", status=500)

# ... existing code ...

def export_multiple_analyses(request):
    """Export multiple tree analyses as CSV"""
    try:
        ids = request.GET.get('ids', '')
        if not ids:
            return HttpResponse("No analysis IDs provided", status=400)
        
        analysis_ids = [int(id.strip()) for id in ids.split(',') if id.strip()]
        analyses = TreeAnalysis.objects.filter(id__in=analysis_ids, is_completed=True).order_by('-completed_at')
        
        if not analyses:
            return HttpResponse("No analyses found", status=404)
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="selected-tree-analyses.csv"'
        
        writer = csv.writer(response)
        
        writer.writerow(['Selected Tree Analyses Report'])
        writer.writerow(['Generated on', timezone.now().strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow(['Total Analyses', analyses.count()])
        writer.writerow([])
        
        writer.writerow(['ID', 'Tree Name', 'Analysis Date', 'Total Leaves', 'Healthy %', 'Dried Leaf %', 'Powdery Mildew %', 'Leaf Rust %', 'Overall Health'])
        
        for analysis in analyses:
            leaf_count = analysis.leaf_images.count()
            writer.writerow([
                analysis.id,
                analysis.name,
                analysis.completed_at.strftime('%Y-%m-%d %H:%M:%S'),
                leaf_count,
                f"{analysis.healthy_percentage:.1f}%",
                f"{analysis.dried_leaf_percentage:.1f}%",
                f"{analysis.powdery_mildew_percentage:.1f}%",
                f"{analysis.leaf_rust_percentage:.1f}%",
                f"{analysis.overall_health:.1f}%"
            ])
        
        logger.info(f"Multiple tree analyses exported: {len(analysis_ids)} analyses")
        return response

    except Exception as e:
        logger.error(f"Error exporting multiple analyses: {e}")
        traceback.print_exc()
        return HttpResponse(f"Error exporting analyses: {str(e)}", status=500)
    
    
from .models import TreeAnalysis

def tree_analysis_list(request):
    """Display all tree analyses"""
    analyses = TreeAnalysis.objects.all()  # pwede mo lagyan ng filter if needed
    return render(request, 'dashboard/tree_analysis_list.html', {'analyses': analyses})

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.utils import timezone
from datetime import datetime
from .models import TreeAnalysis

@login_required
def update_tree_analysis(request, analysis_id):
    analysis = get_object_or_404(TreeAnalysis, id=analysis_id)

    if request.method == 'POST':
        analysis.name = request.POST.get('name')
        analysis.total_leaves = int(request.POST.get('total_leaves') or 0)
        analysis.healthy_count = int(request.POST.get('healthy_count') or 0)
        analysis.dried_leaf_count = int(request.POST.get('dried_leaf_count') or 0)
        analysis.leaf_rust_count = int(request.POST.get('leaf_rust_count') or 0)
        analysis.powdery_mildew_count = int(request.POST.get('powdery_mildew_count') or 0)
        analysis.notes = request.POST.get('notes')

        # Handle created_at
        created_at_input = request.POST.get('created_at')
        if created_at_input:
            try:
                analysis.created_at = datetime.strptime(created_at_input, "%Y-%m-%dT%H:%M")
            except ValueError:
                messages.error(request, "Invalid Created At format")

        # Calculate percentages
        total = analysis.total_leaves or 0
        if total > 0:
            analysis.healthy_percentage = (analysis.healthy_count / total) * 100
            analysis.dried_leaf_percentage = (analysis.dried_leaf_count / total) * 100
            analysis.leaf_rust_percentage = (analysis.leaf_rust_count / total) * 100
            analysis.powdery_mildew_percentage = (analysis.powdery_mildew_count / total) * 100

            analysis.overall_health = (
                analysis.healthy_percentage * 1 +
                analysis.powdery_mildew_percentage * 0.6 +
                analysis.leaf_rust_percentage * 0.2
            )
        else:
            analysis.healthy_percentage = 0
            analysis.dried_leaf_percentage = 0
            analysis.leaf_rust_percentage = 0
            analysis.powdery_mildew_percentage = 0
            analysis.overall_health = 0

        analysis.completed_at = timezone.now()
        analysis.is_completed = True

        # 🔗 Link to Plant if not already linked
        if not analysis.plant:
            plant_id = request.POST.get('plant_id')
            if plant_id:
                try:
                    from dashboard.models import Plant
                    plant = Plant.objects.get(plant_id=plant_id)
                    analysis.plant = plant
                    plant.tree_analysis = analysis
                    plant.save()
                except Plant.DoesNotExist:
                    print(f"[DEBUG] No matching Plant found for analysis ID {analysis.id}")

        # 💾 Save updated analysis
        analysis.save()

        # ✅ Always ensure Plant link stays updated
        if analysis.plant:
            analysis.plant.tree_analysis = analysis
            analysis.plant.save()

        messages.success(request, "Tree analysis updated successfully!")
        return redirect('tree_analysis_list')

    return render(request, 'dashboard/update_tree_analysis.html', {'analysis': analysis})


def delete_tree_analysis(request, analysis_id):
    analysis = get_object_or_404(TreeAnalysis, id=analysis_id)
    analysis.delete()
    messages.success(request, 'Tree analysis deleted successfully!')
    return redirect('tree_analysis_list')

from django.http import JsonResponse

def analysis_detail_json(request, analysis_id):
    """JSON data for analysis modal"""
    analysis = get_object_or_404(TreeAnalysis, id=analysis_id)

    # Default to 0 if None
    healthy_count = analysis.healthy_count or 0
    dried_leaf_count = analysis.dried_leaf_count or 0
    leaf_rust_count = analysis.leaf_rust_count or 0
    powdery_mildew_count = analysis.powdery_mildew_count or 0

    healthy_percentage = analysis.healthy_percentage or 0
    dried_leaf_percentage = analysis.dried_leaf_percentage or 0
    leaf_rust_percentage = analysis.leaf_rust_percentage or 0
    powdery_mildew_percentage = analysis.powdery_mildew_percentage or 0

    total_leaves = analysis.total_leaves or 0

    # Calculate diseased count and percentage
    diseased_count = dried_leaf_count + leaf_rust_count + powdery_mildew_count
    diseased_percentage = (diseased_count / total_leaves * 100) if total_leaves else 0

    data = {
        'id': analysis.id,
        'name': analysis.name,
        'created_at': analysis.created_at.isoformat() if analysis.created_at else None,
        'completed_at': analysis.completed_at.isoformat() if analysis.completed_at else None,
        'overall_health': analysis.overall_health or 0,
        'total_leaves': total_leaves,
        'healthy_count': healthy_count,
        'diseased_count': diseased_count,
        'diseased_percentage': diseased_percentage,
        'dried_leaf_count': dried_leaf_count,
        'leaf_rust_count': leaf_rust_count,
        'powdery_mildew_count': powdery_mildew_count,
        'healthy_percentage': healthy_percentage,
        'dried_leaf_percentage': dried_leaf_percentage,
        'leaf_rust_percentage': leaf_rust_percentage,
        'powdery_mildew_percentage': powdery_mildew_percentage,
        'notes': analysis.notes or "",
    }

    return JsonResponse({'success': True, 'analysis': data})


