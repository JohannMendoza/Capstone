# ✅ UPDATED: Fixed views.py with lazy loading and model caching
import os
import csv
import json
import time
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

def send_verification_email(subject, body, recipient):
    try:
        email = EmailMessage(
            subject,
            body,
            settings.DEFAULT_FROM_EMAIL,
            [recipient]
        )
        email.content_subtype = "html"  # HTML format
        email.send(fail_silently=False)
        print(f"✅ Email sent successfully to {recipient}")
    except Exception as e:
        print(f"❌ Email send failed: {e}")


from django.shortcuts import render, redirect
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.contrib.auth.tokens import default_token_generator
from django.template.loader import render_to_string
from django.contrib import messages
from django.contrib.auth import authenticate, login
from .forms import RegisterForm
from .utils import send_verification_email

def register_view(request):
    """Register new user and send verification email"""
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            # <CHANGE> Create user with role and is_active properly set
            user = form.save(commit=False)
            user.email = form.cleaned_data['email'].lower()
            user.role = "client"  # <CHANGE> Set role to client
            user.is_active = False  # <CHANGE> Require email verification
            user.save()
            
            print(f"[v0] ✅ User created: {user.email} (ID: {user.id})")

            # <CHANGE> Generate verification token and link
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            token = default_token_generator.make_token(user)
            protocol = 'https' if request.is_secure() else 'http'
            domain = request.get_host()
            verification_link = f"{protocol}://{domain}/verify/{uid}/{token}/"
            
            print(f"[v0] Verification URL: {verification_link}")

            # <CHANGE> Render verification email template
            subject = "Verify Your Email - Escala Plants & Nursery"
            try:
                body = render_to_string("dashboard/verify_email_template.html", {
                    "user": user,
                    "verification_link": verification_link,
                    "domain": domain
                })
                print(f"[v0] ✅ Template rendered successfully")
            except Exception as e:
                print(f"[v0] ❌ Template Error: {e}")
                # Fallback email if template fails
                body = f"""
                <html>
                    <body>
                        <h3>Hello {user.username},</h3>
                        <p>Thank you for registering at Escala Plants & Nursery!</p>
                        <p>Please click the button below to verify your email:</p>
                        <a href='{verification_link}' style='background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;'>Verify Email</a>
                        <p>Or paste this link: {verification_link}</p>
                        <p>This link expires in 24 hours.</p>
                    </body>
                </html>
                """

            # <CHANGE> Send verification email with error handling
            email_sent = send_verification_email(subject, body, user.email)
            
            if email_sent:
                print(f"[v0] ✅ SUCCESS: Verification email sent to {user.email}")
                return render(request, "dashboard/register.html", {
                    "form": RegisterForm(),
                    "success": True,
                    "message": "✅ Registration successful! Check your email to verify your account."
                })
            else:
                print(f"[v0] ⚠️  WARNING: User created but email failed for {user.email}")
                # Still show success but warn user
                return render(request, "dashboard/register.html", {
                    "form": RegisterForm(),
                    "success": True,
                    "warning": "⚠️  Registration successful but we couldn't send verification email. Contact support."
                })
        else:
            print(f"[v0] Form validation failed: {form.errors}")
            return render(request, "dashboard/register.html", {"form": form, "errors": form.errors})
    else:
        form = RegisterForm()
    
    return render(request, "dashboard/register.html", {"form": form})

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
    """Verify email and activate user account"""
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        from django.contrib.auth import get_user_model
        User = get_user_model()
        user = User.objects.get(pk=uid)
        
        print(f"[v0] Verifying email for user: {user.email}")
        
        if default_token_generator.check_token(user, token):
            # <CHANGE> Activate user on successful verification
            user.is_active = True
            user.save()
            print(f"[v0] ✅ Email verified for {user.email}")
            
            return render(request, "dashboard/verify_email.html", {
                "verified": True,
                "message": "✅ Email verified! You can now login."
            })
        else:
            print(f"[v0] ❌ Invalid token for user: {user.email}")
            return render(request, "dashboard/verify_email.html", {
                "verified": False,
                "error": "Token expired or invalid. Please register again."
            })
    except Exception as e:
        print(f"[v0] ❌ Verification error: {str(e)}")
        return render(request, "dashboard/verify_email.html", {
            "verified": False,
            "error": "Invalid verification link."
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


from .forms import UserEditForm  # Add this import

@login_required
def user_list(request):
    if request.user.role != "admin":
        return redirect('home')

    if request.method == "POST" and "delete_user_id" in request.POST:
        user_id = request.POST["delete_user_id"]
        # Prevent admin from deleting themselves
        if int(user_id) != request.user.id:
            user = get_object_or_404(CustomUser, id=user_id)
            username = user.username
            user.delete()
            messages.success(request, f"User '{username}' deleted successfully!")
        else:
            messages.error(request, "You cannot delete your own account!")
        
        return redirect("user_list")

    # Get all users except superusers if needed
    users = CustomUser.objects.all().order_by('-date_joined')
    
    # Add search functionality
    search_query = request.GET.get('search', '')
    if search_query:
        users = users.filter(
            Q(username__icontains=search_query) |
            Q(email__icontains=search_query) |
            Q(role__icontains=search_query)
        )

    return render(request, "dashboard/user_list.html", {
        "users": users,
        "search_query": search_query
    })

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

    # 🔹 GET ALL PLANTS (SAME LOGIC AS PLANT INVENTORY)
    plants_queryset = Plant.objects.select_related('tree_analysis').all()
    
    # 🔹 COUNT PLANTS BY HEALTH STATUS (SAME LOGIC AS PLANT INVENTORY)
    total_plants = plants_queryset.count()
    healthy_plants = 0
    needs_attention_plants = 0
    total_health_score = 0
    plants_with_health = 0

    for plant in plants_queryset:
        latest_analysis = TreeAnalysis.objects.filter(plant=plant).order_by('-id').first()
        
        if not latest_analysis:
            latest_analysis = TreeAnalysis.objects.filter(name__icontains=f"Plant {plant.plant_id}").order_by('-id').first()

        if latest_analysis and latest_analysis.overall_health is not None:
            try:
                overall_health = float(latest_analysis.overall_health)
                total_health_score += overall_health
                plants_with_health += 1
                
                if overall_health >= 70:
                    healthy_plants += 1
                else:
                    needs_attention_plants += 1
            except (ValueError, TypeError):
                needs_attention_plants += 1
        else:
            needs_attention_plants += 1

    # Calculate average health score
    avg_health_score = round(total_health_score / plants_with_health, 1) if plants_with_health > 0 else 0

    # Your original disease distribution
    disease_counts = Counter()
    for plant in plants_queryset:
        if plant.health_status and plant.health_status not in ["good", "undetected"]:
            disease_counts[plant.health_status] += 1
    
    disease_labels = list(disease_counts.keys())
    disease_values = list(disease_counts.values())

    return render(request, "dashboard/admin_dashboard.html", {
        'total_users': total_users,
        'total_plants': total_plants,
        'healthy_plants': healthy_plants,  # CORRECT VARIABLE NAME
        'unhealthy_plants': needs_attention_plants,  # CORRECT VARIABLE NAME  
        'avg_health_score': avg_health_score,
        'disease_labels': disease_labels,
        'disease_values': disease_values,
    })

# ... existing code ...

@login_required
def client_dashboard(request):
    if request.user.role != "client":
        return redirect('home')

    # 🔹 GET ALL PLANTS (SAME LOGIC AS ADMIN DASHBOARD)
    plants_queryset = Plant.objects.select_related('tree_analysis').all()
    
    # 🔹 COUNT PLANTS BY HEALTH STATUS (SAME LOGIC AS ADMIN DASHBOARD)
    total_plants = plants_queryset.count()
    healthy_plants = 0
    needs_attention_plants = 0
    total_health_score = 0
    plants_with_health = 0

    for plant in plants_queryset:
        latest_analysis = TreeAnalysis.objects.filter(plant=plant).order_by('-id').first()
        
        if not latest_analysis:
            latest_analysis = TreeAnalysis.objects.filter(name__icontains=f"Plant {plant.plant_id}").order_by('-id').first()

        if latest_analysis and latest_analysis.overall_health is not None:
            try:
                overall_health = float(latest_analysis.overall_health)
                total_health_score += overall_health
                plants_with_health += 1
                
                if overall_health >= 70:
                    healthy_plants += 1
                else:
                    needs_attention_plants += 1
            except (ValueError, TypeError):
                needs_attention_plants += 1
        else:
            needs_attention_plants += 1

    # Calculate average health score
    avg_health_score = round(total_health_score / plants_with_health, 1) if plants_with_health > 0 else 0

    # Disease distribution (same logic as admin dashboard)
    disease_counts = Counter()
    for plant in plants_queryset:
        if plant.health_status and plant.health_status not in ["good", "undetected"]:
            disease_counts[plant.health_status] += 1
    
    disease_labels = list(disease_counts.keys())
    disease_values = list(disease_counts.values())

    context = {
        'total_plants': total_plants,
        'healthy_plants': healthy_plants,  # CORRECT VARIABLE NAME (same as admin)
        'unhealthy_plants': needs_attention_plants,  # CORRECT VARIABLE NAME (same as admin)
        'avg_health_score': avg_health_score,
        'disease_labels': disease_labels,
        'disease_values': disease_values,
        'username': request.user.username,
    }

    return render(request, "dashboard/client_dashboard.html", context)

# ... existing code ...

@login_required
def update_user_view(request, user_id):
    if request.user.role != "admin":
        return redirect('home')
    
    user = get_object_or_404(CustomUser, id=user_id)
    
    if request.method == "POST":
        form = UserEditForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            messages.success(request, f"User '{user.username}' updated successfully!")
            return redirect('user_list')
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = UserEditForm(instance=user)

    return render(request, "dashboard/update_user.html", {
        "form": form, 
        "user": user
    })

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
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from .models import Plant, TreeAnalysis

@login_required
def plant_inventory(request):
    # 🔹 Get all plants (ordered)
    plants_queryset = Plant.objects.select_related('tree_analysis').all().order_by('plant_id')
    
    # 🔹 Count plants by health status
    total_plants = plants_queryset.count()
    healthy_plants = 0
    needs_attention_plants = 0
    plants_data = []  # Store plant data with calculated fields

    for plant in plants_queryset:
        try:
            latest_analysis = TreeAnalysis.objects.filter(plant=plant).order_by('-id').first()

            if not latest_analysis:
                latest_analysis = (
                    TreeAnalysis.objects
                    .filter(name__icontains=f"Plant {plant.plant_id}")
                    .order_by('-id')
                    .first()
                )

            plant_info = {
                'plant': plant,
                'health_status': 'undetected',
                'health_category': 'Not Analyzed',
                'overall_health': None,
                'status_percentage': None,
                'detection_details': None
            }

            if latest_analysis:
                # ✅ Safely get numeric values
                def safe(val):
                    try:
                        return float(val) if val is not None else 0.0
                    except:
                        return 0.0

                detection_details = {
                    'healthy': round(safe(latest_analysis.healthy_percentage), 1),
                    'dried_leaf': round(safe(latest_analysis.dried_leaf_percentage), 1),
                    'leaf_rust': round(safe(latest_analysis.leaf_rust_percentage), 1),
                    'powdery_mildew': round(safe(latest_analysis.powdery_mildew_percentage), 1),
                    'overall_health': round(safe(latest_analysis.overall_health), 1),
                }

                overall_health = safe(latest_analysis.overall_health)
                
                # 🔹 Determine health category
                if overall_health >= 70:
                    health_category = 'Excellent Health'
                    health_status = 'good'
                    healthy_plants += 1
                elif overall_health >= 40:
                    health_category = 'Moderate Health' 
                    health_status = 'moderate'
                    needs_attention_plants += 1
                else:
                    health_category = 'Poor Health'
                    health_status = 'poor'
                    needs_attention_plants += 1
                    
                plant_info.update({
                    'health_status': health_status,
                    'health_category': health_category,
                    'overall_health': overall_health,
                    'status_percentage': overall_health,
                    'detection_details': detection_details
                })

            else:
                # No analysis found
                needs_attention_plants += 1  # Undetected plants need attention
                plant_info.update({
                    'health_status': 'undetected',
                    'health_category': 'Not Analyzed',
                    'detection_details': None
                })

            plants_data.append(plant_info)

        except Exception as e:
            print(f"[ERROR] Problem with plant ID {plant.plant_id}: {e}")
            plants_data.append({
                'plant': plant,
                'health_status': 'error',
                'health_category': 'Error',
                'overall_health': None,
                'status_percentage': None,
                'detection_details': None
            })
            needs_attention_plants += 1

    # 🔹 Paginate the plants_data list
    paginator = Paginator(plants_data, 10)  # 10 items per page
    page_number = request.GET.get('page')
    
    try:
        plants_page = paginator.page(page_number)
    except PageNotAnInteger:
        plants_page = paginator.page(1)
    except EmptyPage:
        plants_page = paginator.page(paginator.num_pages)

    # 🔹 Calculate average health score
    plants_with_health = [p for p in plants_data if p['overall_health'] is not None]
    avg_health_score = sum(p['overall_health'] for p in plants_with_health) / len(plants_with_health) if plants_with_health else 0

    context = {
        'plants_page': plants_page,  # Use plants_page instead of plants_data
        'total_plants': total_plants,
        'healthy_plants': healthy_plants,
        'needs_attention_plants': needs_attention_plants,
        'avg_health_score': round(avg_health_score, 1)
    }

    return render(request, 'dashboard/inventory.html', context)


@login_required
def add_plant(request):
    """Add a new plant to inventory"""
    if request.method == "POST":
        form = PlantForm(request.POST)
        
        if form.is_valid():
            try:
                plant = form.save(commit=False)
                plant.user = request.user
                plant.health_status = "undetected"
                plant.save()
                
                print(f"[v0] Plant added successfully: Plant #{plant.plant_id}")
                messages.success(request, f"Plant added successfully! Plant ID: {plant.plant_id}")
                return redirect('add_plant')
            
            except Exception as e:
                print(f"[v0] Error saving plant: {str(e)}")
                messages.error(request, f"Error adding plant: {str(e)}")
                return render(request, 'dashboard/add_plant.html', {'form': form})
        else:
            print(f"[v0] Form validation failed: {form.errors}")
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field}: {error}")
            return render(request, 'dashboard/add_plant.html', {'form': form})
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
from io import BytesIO
from datetime import datetime
import os
from PIL import Image as PILImage
from django.http import HttpResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from .models import CustomUser, Plant

def export_pdf(request):
    if request.method != "POST":
        return HttpResponse("Invalid request", status=400)

    response = HttpResponse(content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="Escala_Plants_Report_{datetime.now().strftime("%Y%m%d")}.pdf"'
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=0.7*inch, leftMargin=0.7*inch,
                            topMargin=0.7*inch, bottomMargin=0.7*inch)

    # --- Colors ---
    PRIMARY_COLOR = colors.HexColor("#1B5E20")
    SECONDARY_COLOR = colors.HexColor("#2E7D32")
    HEADER_TEXT = colors.white
    LIGHT_BG = colors.HexColor("#F1F8E9")
    NEUTRAL_TEXT = colors.HexColor("#212121")
    BORDER_COLOR = colors.HexColor("#A5D6A7")

    # --- Styles ---
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Heading1"], fontSize=22, leading=28,
                                 textColor=PRIMARY_COLOR, alignment=1, spaceAfter=12, fontName="Helvetica-Bold")
    section_style_left = ParagraphStyle("SectionLeft", parent=styles["Heading3"], fontSize=12,
                                        textColor=PRIMARY_COLOR, spaceBefore=18, spaceAfter=8, fontName="Helvetica-Bold")
    label_style = ParagraphStyle("Label", parent=styles["Normal"], fontSize=10, textColor=SECONDARY_COLOR,
                                 alignment=1)
    footer_style = ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=SECONDARY_COLOR, alignment=1)

    story = []

    # --- Logo ---
    logo_path = os.path.join('dashboard', 'static', 'dashboard', 'img', 'core-img', 'logo.png')
    if os.path.exists(logo_path):
        img = PILImage.open(logo_path).convert("RGBA")
        bg = PILImage.new("RGBA", img.size, (27, 94, 32, 255))
        bg.paste(img, (0, 0), img)
        temp_logo_path = os.path.join('dashboard', 'static', 'dashboard', 'img', 'core-img', 'logo_with_bg.png')
        bg.save(temp_logo_path)
        logo_img = Image(temp_logo_path, width=1.5*inch, height=0.5*inch)
        logo_img.hAlign = "CENTER"
        story.append(logo_img)
        story.append(Spacer(1, 0.15*inch))

    # --- Title ---
    story.append(Paragraph("Escala Plants & Nursery", label_style))
    story.append(Paragraph("Reports", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y')}", label_style))
    story.append(Spacer(1, 0.3*inch))

    page_width = A4[0] - doc.leftMargin - doc.rightMargin

    # --- Users Table ---
    if "export_users" in request.POST:
        users = CustomUser.objects.all().values_list("id", "username", "email", "role", "is_active")
        if users:
            story.append(Paragraph("Users List", section_style_left))
            story.append(Spacer(1, 0.1*inch))

            columns = ["ID", "Username", "Email", "Role", "Active"]
            data = [columns]
            for u in users:
                data.append([str(u[0]), u[1], u[2], u[3], "Yes" if u[4] else "No"])

            col_widths = [
                0.7 * inch,       # ID
                1.8 * inch,       # Username
                3.0 * inch,       # Email (wider)
                1.2 * inch,       # Role
                0.8 * inch        # Active
            ]

            table = Table(data, colWidths=col_widths, rowHeights=0.6*inch)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), SECONDARY_COLOR),
                ("TEXTCOLOR", (0,0), (-1,0), HEADER_TEXT),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("ALIGN", (0,0), (-1,-1), "CENTER"),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ("GRID", (0,0), (-1,-1), 0.5, BORDER_COLOR),
                ("ROWBACKGROUNDS", (1,1), (-1,-1), [colors.white, LIGHT_BG]),
                ("ALIGN", (2,1), (2,-1), "LEFT")  # Email left-aligned
            ]))
            story.append(table)
            story.append(Spacer(1, 0.3*inch))

    # --- Plants Table ---
    plant_keys = ["export_total_plants", "export_healthy_plants", "export_unhealthy_plants"]
    if any(key in request.POST for key in plant_keys):
        story.append(Paragraph("Plants Summary", section_style_left))
        story.append(Spacer(1, 0.1*inch))

        total_plants = Plant.objects.count()
        healthy_plants = Plant.objects.filter(health_status="good").count()
        unhealthy_plants = Plant.objects.exclude(health_status="good").count()

        plants_data = [["Category", "Count"]]
        if "export_total_plants" in request.POST:
            plants_data.append(["Total Plants", str(total_plants)])
        if "export_healthy_plants" in request.POST:
            plants_data.append(["Healthy Plants", str(healthy_plants)])
        if "export_unhealthy_plants" in request.POST:
            plants_data.append(["Unhealthy Plants", str(unhealthy_plants)])

        col_widths = [
            4.0 * inch,   # Category
            1.8 * inch    # Count
        ]

        table = Table(plants_data, colWidths=col_widths, rowHeights=0.6*inch)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), SECONDARY_COLOR),
            ("TEXTCOLOR", (0,0), (-1,0), HEADER_TEXT),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("GRID", (0,0), (-1,-1), 0.5, BORDER_COLOR),
            ("ROWBACKGROUNDS", (1,1), (-1,-1), [colors.white, LIGHT_BG]),
            ("ALIGN", (0,1), (0,-1), "LEFT")  # Category left-aligned
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3*inch))

    # --- Footer ---
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", footer_style))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    response.write(pdf)
    return response


# ✅ FIXED: Model path and caching
model_path = os.path.join(settings.MEDIA_ROOT, 'best.pt')
_MODEL_CACHE = {}
_YOLO_MODEL = None
_LAST_MODEL_LOAD = 0
_MODEL_LOAD_TIMEOUT = 3600  # 1 hour

# Set up logger
logger = logging.getLogger(__name__)

# ✅ FIXED: Add the missing load_yolo_model function
def load_yolo_model():
    """Load YOLO model with caching, better error messages and fallback hints."""
    global _MODEL_CACHE, _YOLO_MODEL, _LAST_MODEL_LOAD

    current_time = time.time()
    
    # Return cached model if available and not expired
    if (_YOLO_MODEL is not None and 
        current_time - _LAST_MODEL_LOAD < _MODEL_LOAD_TIMEOUT):
        logger.info("✅ Using cached YOLO model")
        return _YOLO_MODEL

    if 'yolo_model' in _MODEL_CACHE:
        logger.info("✅ Using cached YOLO model from _MODEL_CACHE")
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
        _YOLO_MODEL = model
        _LAST_MODEL_LOAD = current_time
        
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

# ✅ FIXED: Add the optimized get_yolo_model function
def get_yolo_model():
    """Improved model loading with better caching"""
    global _YOLO_MODEL, _LAST_MODEL_LOAD
    
    current_time = time.time()
    
    # Return cached model if available and not expired
    if (_YOLO_MODEL is not None and 
        current_time - _LAST_MODEL_LOAD < _MODEL_LOAD_TIMEOUT):
        logger.debug("✅ Using cached YOLO model")
        return _YOLO_MODEL
    
    try:
        from ultralytics import YOLO
        
        model_path = os.path.join(settings.MEDIA_ROOT, 'best.pt')
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            return None
        
        logger.info("🔄 Loading YOLO model...")
        _YOLO_MODEL = YOLO(model_path)
        _LAST_MODEL_LOAD = current_time
        
        # Warm up the model
        dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        _YOLO_MODEL.predict(dummy_input, verbose=False)
        
        logger.info("✅ YOLO model loaded and warmed up successfully")
        return _YOLO_MODEL
        
    except Exception as e:
        logger.error(f"❌ Error loading YOLO model: {e}")
        traceback.print_exc()
        return None

# ... existing code continues ...

CLASS_NAMES = ['dried leaf', 'healthy', 'leaf rust', 'powdery mildew']

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
CONF_THRESHOLD = 0.30  # CHANGED from 0.50 to 0.30 (30% instead of 50%)

@csrf_exempt
def predict(request):
    """
    Improved predict function that detects ALL leaves
    - Lowered confidence threshold to 30%
    - Detects all leaves in frame
    - Better error handling
    """
    if request.method != 'POST':
        return JsonResponse({"success": False, "error": "Invalid request method"})

    model = load_yolo_model()
    if model is None:
        return JsonResponse({"success": False, "error": "YOLO model not loaded"})

    try:
        import cv2
        
        frame_file = request.FILES.get('frame')
        plant_id = request.POST.get("plant_id")

        if not frame_file:
            return JsonResponse({"success": False, "error": "No frame received"})

        file_bytes = np.frombuffer(frame_file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        detections = []

        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls].lower().replace('-', ' ')

                if class_name not in ALLOWED_CLASSES:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class": class_name.title()
                })

        logger.info(f"✅ Detected {len(detections)} leaves in frame")

        # Save to database if plant_id provided
        if plant_id:
            from dashboard.models import Plant
            try:
                plant = Plant.objects.get(plant_id=plant_id)
                analysis, created = TreeAnalysis.objects.get_or_create(
                    plant=plant,
                    defaults={"name": f"Analysis for Plant {plant.plant_id}"}
                )

                for det in detections:
                    LeafImage.objects.create(
                        image=None,
                        prediction=det["class"],
                        healthy_confidence=det["confidence"] if det["class"] == "Healthy" else 0,
                        dried_leaf_confidence=det["confidence"] if det["class"] == "Dried Leaf" else 0,
                        leaf_rust_confidence=det["confidence"] if det["class"] == "Leaf Rust" else 0,
                        powdery_mildew_confidence=det["confidence"] if det["class"] == "Powdery Mildew" else 0,
                        tree_analysis=analysis
                    )

                analysis.calculate_health()
                analysis.is_completed = True
                analysis.save()

                plant.tree_analysis = analysis
                plant.health_status = "good" if analysis.overall_health > 70 else "leaf rust"
                plant.save()

            except Exception as e:
                logger.error(f"Error saving detections: {e}")

        return JsonResponse({
            "success": True,
            "detections": detections,
            "detection_count": len(detections)
        })

    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        return JsonResponse({"success": False, "error": str(e)})

@csrf_exempt
@require_POST
def optimized_predict(request):
    """
    ✅ IMPROVED: Optimized prediction function
    - Faster processing with optimized parameters
    - Better detection with lower confidence threshold
    - Performance monitoring
    """
    start_time = time.time()
    
    model = get_yolo_model()
    if model is None:
        return JsonResponse({"success": False, "error": "YOLO model not available"})

    try:
        import cv2
        
        frame_file = request.FILES.get('frame')
        quality = request.POST.get('quality', 'high')
        plant_id = request.POST.get("plant_id")

        if not frame_file:
            return JsonResponse({"success": False, "error": "No frame received"})

        # ✅ IMPROVED: Faster image decoding
        file_bytes = np.frombuffer(frame_file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            return JsonResponse({"success": False, "error": "Failed to decode frame"})

        # ✅ IMPROVED: Adjust parameters based on quality
        imgsz = 640 if quality == 'low' else 640
        conf_threshold = 0.20 if quality == 'low' else CONF_THRESHOLD

        # ✅ IMPROVED: Optimized prediction parameters
        results = model.predict(
            frame, 
            conf=conf_threshold,
            imgsz=imgsz,
            verbose=False,
            max_det=50,  # Limit maximum detections
            agnostic_nms=True,  # Faster NMS
            half=False  # Use full precision for better accuracy
        )[0]

        detections = []
        detection_count = 0

        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls].lower().replace('-', ' ')

                if class_name not in ALLOWED_CLASSES:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class": class_name.title()
                })
                detection_count += 1

        processing_time = time.time() - start_time
        
        logger.info(f"✅ Detection completed: {detection_count} leaves in {processing_time:.2f}s")

        # ✅ IMPROVED: Save to database if plant_id provided (non-blocking)
        if plant_id and detection_count > 0:
            try:
                from dashboard.models import Plant, TreeAnalysis, LeafImage
                from django.utils import timezone
                
                plant = Plant.objects.get(plant_id=plant_id)
                analysis, created = TreeAnalysis.objects.get_or_create(
                    plant=plant,
                    defaults={"name": f"Analysis for Plant {plant.plant_id}"}
                )

                for det in detections:
                    LeafImage.objects.create(
                        image=None,
                        prediction=det["class"],
                        healthy_confidence=det["confidence"] if det["class"] == "Healthy" else 0,
                        dried_leaf_confidence=det["confidence"] if det["class"] == "Dried Leaf" else 0,
                        leaf_rust_confidence=det["confidence"] if det["class"] == "Leaf Rust" else 0,
                        powdery_mildew_confidence=det["confidence"] if det["class"] == "Powdery Mildew" else 0,
                        tree_analysis=analysis
                    )

                analysis.calculate_health()
                analysis.is_completed = True
                analysis.save()

                plant.tree_analysis = analysis
                plant.health_status = "good" if analysis.overall_health > 70 else "leaf rust"
                plant.save()
                
                logger.info(f"✅ Saved {detection_count} detections to plant {plant_id}")

            except Exception as e:
                logger.error(f"Error saving detections: {e}")
                # Don't fail the request if saving fails

        return JsonResponse({
            "success": True,
            "detections": detections,
            "detection_count": detection_count,
            "processing_time": round(processing_time, 2)
        })

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        traceback.print_exc()
        return JsonResponse({"success": False, "error": str(e)})


@csrf_exempt
@require_POST
def fast_predict(request):
    """
    Ultra-fast prediction for real-time detection
    - Minimal processing
    - Fastest possible response
    """
    start_time = time.time()
    
    model = get_yolo_model()
    if model is None:
        return JsonResponse({"success": False, "error": "Model unavailable"})

    try:
        frame_file = request.FILES.get('frame')
        if not frame_file:
            return JsonResponse({"success": False, "error": "No frame"})

        # Ultra-fast decoding
        file_bytes = np.frombuffer(frame_file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JsonResponse({"success": False, "error": "Invalid image"})

        # Minimal prediction parameters
        results = model.predict(
            frame, 
            conf=0.20,  # Very low threshold for maximum detection
            imgsz=480,  # Smaller size for speed
            verbose=False,
            max_det=30,
            agnostic_nms=True
        )[0]

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls].lower().replace('-', ' ')
                
                if class_name not in ALLOWED_CLASSES:
                    continue
                    
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class": class_name.title()
                })

        processing_time = time.time() - start_time
        
        return JsonResponse({
            "success": True,
            "detections": detections,
            "detection_count": len(detections),
            "processing_time": round(processing_time, 3)  # Millisecond precision
        })

    except Exception as e:
        logger.error(f"❌ Fast prediction error: {e}")
        return JsonResponse({"success": False, "error": str(e)})

# ... existing code ...

@login_required
def detector(request):
    plant_id = request.GET.get('plant_id')
    user_role = request.user.role

    # If plant_id is provided, redirect to new_tree_analysis
    if plant_id:
        url = reverse('new_tree_analysis')
        return redirect(f'{url}?plant_id={plant_id}')

    # ✅ FIXED: Now load_yolo_model is defined
    model = load_yolo_model()
    context = {}

    if model is None:
        context['model_error'] = True
        model_path = os.path.join(settings.MEDIA_ROOT, 'best.pt')
        context['model_path'] = model_path
        context['model_exists'] = os.path.exists(model_path)

    context['recent_analyses'] = TreeAnalysis.objects.filter(is_completed=True).order_by('-completed_at')[:5]
    context['user_role'] = user_role  # Pass user role to template

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

# ✅ FIXED: complete_analysis function in views.py
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
        
        # ✅ FIXED: Handle plant_id properly
        plant_id = request.POST.get('plant_id')
        plant_updated = False
        
        if plant_id and plant_id != 'null' and plant_id != 'undefined':
            try:
                # Convert to integer if it's a numeric string
                if plant_id.isdigit():
                    plant_id_int = int(plant_id)
                    plant = Plant.objects.get(plant_id=plant_id_int)
                    plant.tree_analysis = tree_analysis
                    
                    # Update plant health status based on analysis
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
                    plant_updated = True
                    logger.info(f"Updated plant {plant_id} health to {plant.health_status}")
                else:
                    logger.warning(f"Invalid plant_id format: {plant_id}")
                    
            except Plant.DoesNotExist:
                logger.warning(f"Plant ID {plant_id} not found.")
            except ValueError:
                logger.warning(f"Invalid plant_id: {plant_id}")
        else:
            logger.info("No valid plant_id provided, skipping plant update")

        return JsonResponse({
            'success': True,
            'tree_analysis_id': tree_analysis.id,
            'healthy_percentage': tree_analysis.healthy_percentage,
            'dried_leaf_percentage': tree_analysis.dried_leaf_percentage,
            'leaf_rust_percentage': tree_analysis.leaf_rust_percentage,
            'powdery_mildew_percentage': tree_analysis.powdery_mildew_percentage,
            'overall_health': overall_health,
            'total_detections': healthy_count + dried_leaf_count + powdery_mildew_count + leaf_rust_count,
            'plant_updated': plant_updated
        })

    except Exception as e:
        logger.error(f"Error completing analysis: {e}")
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': str(e)
        })
# ... existing code ...

# ✅ ADD THIS TO views.py
@login_required
def check_plants(request):
    """Check if user has any plants in inventory"""
    try:
        plant_count = Plant.objects.count()
        return JsonResponse({
            'has_plants': plant_count > 0,
            'plant_count': plant_count
        })
    except Exception as e:
        logger.error(f"Error checking plants: {e}")
        return JsonResponse({'has_plants': False})

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
from django.core.paginator import Paginator
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from .models import TreeAnalysis, PestDetectionSession

@login_required
def history(request):
    # <CHANGE> Fixed pagination handling with proper page context
    tree_analyses = TreeAnalysis.objects.filter(is_completed=True).order_by('-completed_at')
    pest_sessions = PestDetectionSession.objects.all().order_by('-created_at')

    analyses = []

    for analysis in tree_analyses:
        diseased_count = analysis.dried_leaf_count + analysis.leaf_rust_count + analysis.powdery_mildew_count
        diseased_percentage = (diseased_count / analysis.total_leaves) * 100 if analysis.total_leaves > 0 else 0
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
            'diseased_count': diseased_count,
            'diseased_percentage': diseased_percentage,
            'type': 'tree_analysis',
        })

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

    # <CHANGE> Proper pagination with Paginator
    paginator = Paginator(analyses, 5)
    page_number = request.GET.get('page')
    
    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)

    return render(request, 'dashboard/history.html', {'analyses': page_obj})
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
import os
import logging
from datetime import datetime
from io import BytesIO
from django.conf import settings
from django.http import HttpResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
)
from dashboard.models import TreeAnalysis
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

def export_analysis_pdf(request, analysis_id):
    """Generate a professional Tree Analysis Report PDF with logo on green background."""

    try:
        tree_analysis = TreeAnalysis.objects.get(id=analysis_id)
    except TreeAnalysis.DoesNotExist:
        return HttpResponse("Analysis not found", status=404)

    # --- PDF Setup ---
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = (
        f'attachment; filename="tree_analysis_{analysis_id}_{datetime.now().strftime("%Y%m%d")}.pdf"'
    )

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )

    # --- Colors ---
    PRIMARY_COLOR = colors.HexColor("#1B5E20")
    SECONDARY_COLOR = colors.HexColor("#2E7D32")
    ACCENT_COLOR = colors.HexColor("#43A047")
    WARNING_COLOR = colors.HexColor("#F57F17")
    DANGER_COLOR = colors.HexColor("#C62828")
    LIGHT_BG = colors.HexColor("#F1F8E9")
    NEUTRAL_TEXT = colors.HexColor("#212121")
    SECONDARY_TEXT = colors.HexColor("#616161")
    BORDER_COLOR = colors.HexColor("#A5D6A7")

    # --- Styles ---
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Heading1"], fontSize=28,
        leading=34, textColor=PRIMARY_COLOR, alignment=1,
        spaceAfter=6, fontName="Helvetica-Bold"
    )
    company_style = ParagraphStyle(
        "Company", parent=styles["Heading3"], fontSize=12,
        textColor=SECONDARY_COLOR, alignment=1, spaceAfter=4,
        fontName="Helvetica-Bold"
    )
    section_heading = ParagraphStyle(
        "SectionHeading", parent=styles["Heading3"], fontSize=12,
        textColor=PRIMARY_COLOR, spaceBefore=16, spaceAfter=12,
        fontName="Helvetica-Bold"
    )
    body_style = ParagraphStyle(
        "BodyText", parent=styles["Normal"], fontSize=10,
        leading=14, textColor=NEUTRAL_TEXT, spaceAfter=6
    )
    label_style = ParagraphStyle(
        "Label", parent=styles["Normal"], fontSize=9,
        textColor=SECONDARY_TEXT, fontName="Helvetica"
    )
    footer_style = ParagraphStyle(
        "Footer", parent=styles["Normal"], fontSize=8,
        textColor=SECONDARY_TEXT, alignment=1
    )

    story = []

    # --- Prepare Logo with Green Background ---
    logo_path = os.path.join(settings.BASE_DIR, 'dashboard', 'static', 'dashboard', 'img', 'core-img', 'logo.png')
    logo_temp_path = os.path.join(settings.BASE_DIR, 'dashboard', 'static', 'dashboard', 'img', 'core-img', 'logo_with_bg.png')

    if os.path.exists(logo_path):
        # Flatten transparent logo on green background
        img = PILImage.open(logo_path).convert("RGBA")
        bg = PILImage.new("RGBA", img.size, (27, 94, 32, 255))  # PRIMARY_COLOR
        bg.paste(img, (0,0), img)
        bg.save(logo_temp_path)

        # Add logo to PDF
        logo_img = Image(logo_temp_path, width=1 * inch, height=0.4 * inch)
        logo_img.hAlign = "CENTER"
        story.append(logo_img)
        story.append(Spacer(1, 0.15 * inch))
    else:
        logger.warning(f"Logo not found at path: {logo_path}")

    # --- Title Section ---
    story.append(Paragraph("Escala Plants & Nursery", company_style))
    story.append(Paragraph("Tree Health Analysis Report", title_style))
    story.append(Paragraph(
        f"Report ID: TREE-{analysis_id} | {datetime.now().strftime('%B %d, %Y')}", label_style
    ))
    story.append(Spacer(1, 0.2 * inch))

    # --- Analysis Overview ---
    story.append(Paragraph("Analysis Overview", section_heading))
    meta_data = [
        ["Analysis Name", tree_analysis.name],
        ["Generated Date", datetime.now().strftime("%B %d, %Y at %I:%M %p")],
        ["Analysis Date", tree_analysis.completed_at.strftime("%B %d, %Y") if tree_analysis.completed_at else "Pending"],
        ["Total Leaves Analyzed", str(tree_analysis.total_leaves or 0)],
    ]
    meta_table = Table(meta_data, colWidths=[2.2 * inch, 4.3 * inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BG),
        ("BACKGROUND", (1, 0), (1, -1), colors.white),
        ("TEXTCOLOR", (0, 0), (0, -1), PRIMARY_COLOR),
        ("TEXTCOLOR", (1, 0), (1, -1), NEUTRAL_TEXT),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 1, BORDER_COLOR),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.25 * inch))

    # --- Health Status Summary ---
    story.append(Paragraph("Health Status Summary", section_heading))
    overall_health = tree_analysis.overall_health or 0
    if overall_health >= 80:
        health_status, health_color, indicator = "Excellent", ACCENT_COLOR, "HEALTHY"
    elif overall_health >= 50:
        health_status, health_color, indicator = "Moderate", WARNING_COLOR, "CAUTION"
    else:
        health_status, health_color, indicator = "Poor", DANGER_COLOR, "CRITICAL"

    summary_data = [
        ["Metric", "Score", "Status"],
        ["Overall Tree Health", f"{overall_health:.1f}%", f"{health_status} • {indicator}"],
    ]
    summary_table = Table(summary_data, colWidths=[2.5 * inch, 1.5 * inch, 2.5 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BACKGROUND", (0, 1), (-1, 1), health_color),
        ("TEXTCOLOR", (0, 1), (-1, 1), colors.white),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, 1), 11),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 1, colors.white),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.25 * inch))

    # --- Leaf Analysis Breakdown ---
    story.append(Paragraph("Leaf Analysis Breakdown", section_heading))
    detection_data = [
        ["Category", "Count", "Percentage", "Status"],
        ["Healthy Leaves", str(tree_analysis.healthy_count or 0), f"{tree_analysis.healthy_percentage or 0:.1f}%", "GOOD"],
        ["Dried Leaves", str(tree_analysis.dried_leaf_count or 0), f"{tree_analysis.dried_leaf_percentage or 0:.1f}%", "WARN"],
        ["Powdery Mildew", str(tree_analysis.powdery_mildew_count or 0), f"{tree_analysis.powdery_mildew_percentage or 0:.1f}%", "WARN"],
        ["Leaf Rust", str(tree_analysis.leaf_rust_count or 0), f"{tree_analysis.leaf_rust_percentage or 0:.1f}%", "CRITICAL"],
    ]
    detection_table = Table(detection_data, colWidths=[2 * inch, 1.2 * inch, 1.3 * inch, 1.7 * inch])
    detection_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SECONDARY_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 1, BORDER_COLOR),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
    ]))
    story.append(detection_table)
    story.append(Spacer(1, 0.25 * inch))

    # --- Recommendations ---
    story.append(Paragraph("Recommendations", section_heading))
    if overall_health >= 80:
        recommendation, recommendation_color = (
            "Tree health is excellent. Continue regular monitoring and maintain current care practices to ensure sustained vitality.",
            ACCENT_COLOR
        )
    elif overall_health >= 50:
        recommendation, recommendation_color = (
            "Tree shows moderate concerns. Apply preventive treatments, increase monitoring frequency to weekly inspections, and consider professional consultation.",
            WARNING_COLOR
        )
    else:
        recommendation, recommendation_color = (
            "Tree health requires immediate attention. Professional expert consultation is strongly recommended. Implement intensive treatment protocols without delay.",
            DANGER_COLOR
        )

    rec_table = Table([[Paragraph(recommendation, body_style)]], colWidths=[6.3 * inch])
    rec_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F5F5")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("GRID", (0, 0), (-1, -1), 1, recommendation_color),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 0.3 * inch))

    # --- Footer ---
    footer_text = f"Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} | Report ID: TREE-{analysis_id}"
    story.append(Paragraph(footer_text, footer_style))

    # --- Build PDF ---
    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    response.write(pdf)

    logger.info(f"Tree analysis PDF exported: {analysis_id}")
    return response

# ... existing code ...

def export_all_analyses(request):
    """Export all completed tree analyses as professional PDF reports, each on a separate page"""
    try:
        analyses = TreeAnalysis.objects.filter(is_completed=True).order_by('-completed_at')

        # --- PDF Setup ---
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="all-tree-analyses.pdf"'

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=0.7*inch,
            leftMargin=0.7*inch,
            topMargin=0.7*inch,
            bottomMargin=0.7*inch
        )

        # --- Colors ---
        PRIMARY_COLOR = colors.HexColor("#1B5E20")
        SECONDARY_COLOR = colors.HexColor("#2E7D32")
        ACCENT_COLOR = colors.HexColor("#43A047")
        WARNING_COLOR = colors.HexColor("#F57F17")
        DANGER_COLOR = colors.HexColor("#C62828")
        LIGHT_BG = colors.HexColor("#F1F8E9")
        NEUTRAL_TEXT = colors.HexColor("#212121")
        SECONDARY_TEXT = colors.HexColor("#616161")
        BORDER_COLOR = colors.HexColor("#A5D6A7")

        # --- Styles ---
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("ReportTitle", parent=styles["Heading1"], fontSize=22,
                                     leading=26, textColor=PRIMARY_COLOR, alignment=1,
                                     spaceAfter=6, fontName="Helvetica-Bold")
        company_style = ParagraphStyle("Company", parent=styles["Heading3"], fontSize=12,
                                       textColor=SECONDARY_COLOR, alignment=1, spaceAfter=4,
                                       fontName="Helvetica-Bold")
        section_heading = ParagraphStyle("SectionHeading", parent=styles["Heading3"], fontSize=12,
                                        textColor=PRIMARY_COLOR, spaceBefore=16, spaceAfter=12,
                                        fontName="Helvetica-Bold")
        body_style = ParagraphStyle("BodyText", parent=styles["Normal"], fontSize=10,
                                    leading=14, textColor=NEUTRAL_TEXT, spaceAfter=6)
        label_style = ParagraphStyle("Label", parent=styles["Normal"], fontSize=9,
                                     textColor=SECONDARY_TEXT, fontName="Helvetica")
        footer_style = ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8,
                                      textColor=SECONDARY_TEXT, alignment=1)

        story = []

        # --- Logo paths ---
        logo_path = os.path.join(settings.BASE_DIR, 'dashboard', 'static', 'dashboard', 'img', 'core-img', 'logo.png')
        logo_temp_path = os.path.join(settings.BASE_DIR, 'dashboard', 'static', 'dashboard', 'img', 'core-img', 'logo_with_bg.png')

        if os.path.exists(logo_path):
            img = PILImage.open(logo_path).convert("RGBA")
            bg = PILImage.new("RGBA", img.size, (27, 94, 32, 255))  # PRIMARY_COLOR
            bg.paste(img, (0,0), img)
            bg.save(logo_temp_path)

        # --- Loop through each analysis ---
        for idx, analysis in enumerate(analyses):
            # --- Logo ---
            if os.path.exists(logo_temp_path):
                logo_img = Image(logo_temp_path, width=1*inch, height=0.4*inch)
                logo_img.hAlign = "CENTER"
                story.append(logo_img)
                story.append(Spacer(1, 0.15*inch))

            # --- Title Section ---
            story.append(Paragraph("Escala Plants & Nursery", company_style))
            story.append(Paragraph("Tree Health Analysis Report", title_style))
            story.append(Paragraph(
                f"Report ID: TREE-{analysis.id} | Generated: {datetime.now().strftime('%B %d, %Y')}", label_style
            ))
            story.append(Spacer(1, 0.2*inch))

            # --- Analysis Overview ---
            story.append(Paragraph("Analysis Overview", section_heading))
            meta_data = [
                ["Analysis Name", analysis.name],
                ["Generated Date", datetime.now().strftime("%B %d, %Y at %I:%M %p")],
                ["Analysis Date", analysis.completed_at.strftime("%B %d, %Y") if analysis.completed_at else "Pending"],
                ["Total Leaves Analyzed", str(analysis.total_leaves or 0)],
            ]
            meta_table = Table(meta_data, colWidths=[2.2 * inch, 4.3 * inch])
            meta_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), LIGHT_BG),
                ("BACKGROUND", (1, 0), (1, -1), colors.white),
                ("TEXTCOLOR", (0, 0), (0, -1), PRIMARY_COLOR),
                ("TEXTCOLOR", (1, 0), (1, -1), NEUTRAL_TEXT),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 1, BORDER_COLOR),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ]))
            story.append(meta_table)
            story.append(Spacer(1, 0.25*inch))

            # --- Health Status Summary ---
            story.append(Paragraph("Health Status Summary", section_heading))
            overall_health = analysis.overall_health or 0
            if overall_health >= 80:
                health_status, health_color, indicator = "Excellent", ACCENT_COLOR, "HEALTHY"
            elif overall_health >= 50:
                health_status, health_color, indicator = "Moderate", WARNING_COLOR, "CAUTION"
            else:
                health_status, health_color, indicator = "Poor", DANGER_COLOR, "CRITICAL"

            summary_data = [
                ["Metric", "Score", "Status"],
                ["Overall Tree Health", f"{overall_health:.1f}%", f"{health_status} • {indicator}"],
            ]
            summary_table = Table(summary_data, colWidths=[2.5 * inch, 1.5 * inch, 2.5 * inch])
            summary_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_COLOR),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 1), (-1, 1), health_color),
                ("TEXTCOLOR", (0, 1), (-1, 1), colors.white),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 1), (-1, 1), 11),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 1, colors.white),
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 0.25*inch))

            # --- Leaf Analysis Breakdown ---
            story.append(Paragraph("Leaf Analysis Breakdown", section_heading))
            detection_data = [
                ["Category", "Count", "Percentage", "Status"],
                ["Healthy Leaves", str(analysis.healthy_count or 0), f"{analysis.healthy_percentage or 0:.1f}%", "GOOD"],
                ["Dried Leaves", str(analysis.dried_leaf_count or 0), f"{analysis.dried_leaf_percentage or 0:.1f}%", "WARN"],
                ["Powdery Mildew", str(analysis.powdery_mildew_count or 0), f"{analysis.powdery_mildew_percentage or 0:.1f}%", "WARN"],
                ["Leaf Rust", str(analysis.leaf_rust_count or 0), f"{analysis.leaf_rust_percentage or 0:.1f}%", "CRITICAL"],
            ]
            detection_table = Table(detection_data, colWidths=[2 * inch, 1.2 * inch, 1.3 * inch, 1.7 * inch])
            detection_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), SECONDARY_COLOR),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 1, BORDER_COLOR),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ]))
            story.append(detection_table)
            story.append(Spacer(1, 0.25*inch))

            # --- Recommendations ---
            story.append(Paragraph("Recommendations", section_heading))
            if overall_health >= 80:
                recommendation, recommendation_color = (
                    "Tree health is excellent. Continue regular monitoring and maintain current care practices to ensure sustained vitality.",
                    ACCENT_COLOR
                )
            elif overall_health >= 50:
                recommendation, recommendation_color = (
                    "Tree shows moderate concerns. Apply preventive treatments, increase monitoring frequency to weekly inspections, and consider professional consultation.",
                    WARNING_COLOR
                )
            else:
                recommendation, recommendation_color = (
                    "Tree health requires immediate attention. Professional expert consultation is strongly recommended. Implement intensive treatment protocols without delay.",
                    DANGER_COLOR
                )

            rec_table = Table([[Paragraph(recommendation, body_style)]], colWidths=[6.3 * inch])
            rec_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F5F5")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("GRID", (0, 0), (-1, -1), 1, recommendation_color),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(rec_table)
            story.append(Spacer(1, 0.3*inch))

            # --- Footer ---
            footer_text = f"Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} | Report ID: TREE-{analysis.id}"
            story.append(Paragraph(footer_text, footer_style))

            # --- Page break except last analysis ---
            if idx < len(analyses) - 1:
                story.append(PageBreak())

        # --- Build PDF ---
        doc.build(story)
        pdf = buffer.getvalue()
        buffer.close()
        response.write(pdf)

        logger.info("All tree analyses PDF exported successfully.")
        return response

    except Exception as e:
        logger.error(f"Error exporting all analyses PDF: {e}")
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

def export_pest_session_pdf(request, session_id):
    """Export pest detection session as PDF with professional formatting."""
    try:
        session = get_object_or_404(PestDetectionSession, id=session_id)
        results = session.results.all()

        # --- PDF Setup ---
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="pest_session_{session_id}_{datetime.now().strftime("%Y%m%d")}.pdf"'
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=0.7*inch,
            leftMargin=0.7*inch,
            topMargin=0.7*inch,
            bottomMargin=0.7*inch,
        )

        # --- Colors ---
        PRIMARY_COLOR = colors.HexColor("#1B5E20")
        SECONDARY_COLOR = colors.HexColor("#2E7D32")
        HEADER_TEXT = colors.white
        LIGHT_BG = colors.HexColor("#F1F8E9")
        NEUTRAL_TEXT = colors.HexColor("#212121")
        BORDER_COLOR = colors.HexColor("#A5D6A7")

        # --- Styles ---
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "Title", parent=styles["Heading1"], fontSize=22, leading=28,
            textColor=PRIMARY_COLOR, alignment=1, spaceAfter=8, fontName="Helvetica-Bold"
        )
        section_style = ParagraphStyle(
            "Section", parent=styles["Heading3"], fontSize=12, textColor=PRIMARY_COLOR,
            spaceBefore=16, spaceAfter=8, fontName="Helvetica-Bold"
        )
        body_style = ParagraphStyle(
            "Body", parent=styles["Normal"], fontSize=10, leading=14,
            textColor=NEUTRAL_TEXT, spaceAfter=6
        )
        label_style = ParagraphStyle(
            "Label", parent=styles["Normal"], fontSize=10, textColor=SECONDARY_COLOR,
            alignment=1  # center
        )
        footer_style = ParagraphStyle(
            "Footer", parent=styles["Normal"], fontSize=8, textColor=SECONDARY_COLOR, alignment=1
        )
        cell_style = ParagraphStyle(
            "CellStyle", parent=styles["Normal"], fontSize=9, leading=10, alignment=1, textColor=NEUTRAL_TEXT
        )

        story = []

        # --- Logo ---
        logo_path = os.path.join('dashboard', 'static', 'dashboard', 'img', 'core-img', 'logo.png')
        if os.path.exists(logo_path):
            img = PILImage.open(logo_path).convert("RGBA")
            bg = PILImage.new("RGBA", img.size, (27, 94, 32, 255))  # Green background
            bg.paste(img, (0,0), img)
            temp_logo_path = os.path.join('dashboard', 'static', 'dashboard', 'img', 'core-img', 'logo_with_bg.png')
            bg.save(temp_logo_path)
            logo_img = Image(temp_logo_path, width=1*inch, height=0.4*inch)
            logo_img.hAlign = "CENTER"
            story.append(logo_img)
            story.append(Spacer(1, 0.15*inch))

        # --- Title ---
        story.append(Paragraph("Escala Plants & Nursery", label_style))
        story.append(Paragraph("Pest Detection Session Report", title_style))
        story.append(Paragraph(f"Session ID: PEST-{session_id} | {datetime.now().strftime('%B %d, %Y')}", label_style))
        story.append(Spacer(1, 0.2*inch))

        # --- Session Overview ---
        story.append(Paragraph("Session Overview", section_style))
        overview_data = [
            ["Session Name", session.session_name],
            ["Date", session.created_at.strftime("%B %d, %Y at %I:%M %p")],
            ["Total Processed", str(session.total_processed)],
            ["No Pest Count", str(session.no_pest_count)],
            ["Pest Count", str(session.pest_count)],
            ["High Risk Count", str(session.high_risk_count)],
            ["Uncertain Count", str(session.uncertain_count)],
            ["Average Confidence", f"{session.avg_confidence:.1f}%"],
            ["Average Processing Time", f"{session.avg_processing_time:.2f}s"],
        ]
        overview_table = Table(overview_data, colWidths=[2.3*inch, 4.2*inch])
        overview_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), LIGHT_BG),
            ("BACKGROUND", (1,0), (1,-1), colors.white),
            ("TEXTCOLOR", (0,0), (0,-1), PRIMARY_COLOR),
            ("TEXTCOLOR", (1,0), (1,-1), NEUTRAL_TEXT),
            ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
            ("RIGHTPADDING", (0,0), (-1,-1), 8),
            ("TOPPADDING", (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("GRID", (0,0), (-1,-1), 1, BORDER_COLOR),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT_BG])
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 0.25*inch))

        # --- Individual Results ---
        story.append(Paragraph("Individual Results", section_style))
        results_data = [["Filename", "Prediction", "Confidence", "Processing Time", "Low Confidence", "Timestamp"]]
        for r in results:
            filename_para = Paragraph(r.filename, cell_style)
            results_data.append([
                filename_para,
                Paragraph(r.prediction, cell_style),
                Paragraph(f"{r.confidence:.1f}%", cell_style),
                Paragraph(f"{r.processing_time:.2f}s", cell_style),
                Paragraph("Yes" if r.is_low_confidence else "No", cell_style),
                Paragraph(r.timestamp.strftime("%Y-%m-%d %H:%M:%S"), cell_style)
            ])
        results_table = Table(results_data, colWidths=[2.0*inch, 1.3*inch, 1*inch, 1.2*inch, 1*inch, 1.5*inch])
        results_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), SECONDARY_COLOR),
            ("TEXTCOLOR", (0,0), (-1,0), HEADER_TEXT),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("ALIGN", (0,0), (-1,0), "CENTER"),
            ("ALIGN", (1,1), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("GRID", (0,0), (-1,-1), 0.5, BORDER_COLOR),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT_BG])
        ]))
        story.append(results_table)
        story.append(Spacer(1, 0.25*inch))

        # --- Footer ---
        footer_text = f"Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} | Session ID: PEST-{session_id}"
        story.append(Paragraph(footer_text, footer_style))

        # --- Build PDF ---
        doc.build(story)
        pdf = buffer.getvalue()
        buffer.close()
        response.write(pdf)

        logger.info(f"Pest session PDF exported: {session_id}")
        return response

    except Exception as e:
        logger.error(f"Error exporting pest session PDF: {e}")
        return HttpResponse(f"Error exporting pest session: {str(e)}", status=500)

# ... existing code ...

def export_multiple_analyses(request):
    """Export selected tree analyses as professional PDF reports, each on a separate page"""
    try:
        ids = request.GET.get('ids', '')
        if not ids:
            return HttpResponse("No analysis IDs provided", status=400)

        analysis_ids = [int(i.strip()) for i in ids.split(',') if i.strip()]
        analyses = TreeAnalysis.objects.filter(id__in=analysis_ids, is_completed=True).order_by('-completed_at')

        if not analyses:
            return HttpResponse("No analyses found for the selected IDs", status=404)

        # --- PDF Setup ---
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="selected-tree-analyses.pdf"'

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=0.7*inch,
            leftMargin=0.7*inch,
            topMargin=0.7*inch,
            bottomMargin=0.7*inch
        )

        # --- Colors and Styles ---
        PRIMARY_COLOR = colors.HexColor("#1B5E20")
        SECONDARY_COLOR = colors.HexColor("#2E7D32")
        ACCENT_COLOR = colors.HexColor("#43A047")
        WARNING_COLOR = colors.HexColor("#F57F17")
        DANGER_COLOR = colors.HexColor("#C62828")
        LIGHT_BG = colors.HexColor("#F1F8E9")
        NEUTRAL_TEXT = colors.HexColor("#212121")
        SECONDARY_TEXT = colors.HexColor("#616161")
        BORDER_COLOR = colors.HexColor("#A5D6A7")

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("ReportTitle", parent=styles["Heading1"], fontSize=22,
                                     leading=26, textColor=PRIMARY_COLOR, alignment=1,
                                     spaceAfter=6, fontName="Helvetica-Bold")
        company_style = ParagraphStyle("Company", parent=styles["Heading3"], fontSize=12,
                                       textColor=SECONDARY_COLOR, alignment=1, spaceAfter=4,
                                       fontName="Helvetica-Bold")
        section_heading = ParagraphStyle("SectionHeading", parent=styles["Heading3"], fontSize=12,
                                        textColor=PRIMARY_COLOR, spaceBefore=16, spaceAfter=12,
                                        fontName="Helvetica-Bold")
        body_style = ParagraphStyle("BodyText", parent=styles["Normal"], fontSize=10,
                                    leading=14, textColor=NEUTRAL_TEXT, spaceAfter=6)
        label_style = ParagraphStyle("Label", parent=styles["Normal"], fontSize=9,
                                     textColor=SECONDARY_TEXT, fontName="Helvetica")
        footer_style = ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8,
                                      textColor=SECONDARY_TEXT, alignment=1)

        story = []

        # --- Logo paths ---
        logo_path = os.path.join(settings.BASE_DIR, 'dashboard', 'static', 'dashboard', 'img', 'core-img', 'logo.png')
        logo_temp_path = os.path.join(settings.BASE_DIR, 'dashboard', 'static', 'dashboard', 'img', 'core-img', 'logo_with_bg.png')
        if os.path.exists(logo_path):
            img = PILImage.open(logo_path).convert("RGBA")
            bg = PILImage.new("RGBA", img.size, (27, 94, 32, 255))  # PRIMARY_COLOR
            bg.paste(img, (0,0), img)
            bg.save(logo_temp_path)

        # --- Loop through each selected analysis ---
        for idx, analysis in enumerate(analyses):
            # Logo
            if os.path.exists(logo_temp_path):
                logo_img = Image(logo_temp_path, width=1*inch, height=0.4*inch)
                logo_img.hAlign = "CENTER"
                story.append(logo_img)
                story.append(Spacer(1, 0.15*inch))

            # Title
            story.append(Paragraph("Escala Plants & Nursery", company_style))
            story.append(Paragraph("Tree Health Analysis Report", title_style))
            story.append(Paragraph(f"Report ID: TREE-{analysis.id} | Generated: {datetime.now().strftime('%B %d, %Y')}", label_style))
            story.append(Spacer(1, 0.2*inch))

            # Analysis Overview
            story.append(Paragraph("Analysis Overview", section_heading))
            meta_data = [
                ["Analysis Name", analysis.name],
                ["Generated Date", datetime.now().strftime("%B %d, %Y at %I:%M %p")],
                ["Analysis Date", analysis.completed_at.strftime("%B %d, %Y") if analysis.completed_at else "Pending"],
                ["Total Leaves Analyzed", str(analysis.total_leaves or 0)],
            ]
            meta_table = Table(meta_data, colWidths=[2.2 * inch, 4.3 * inch])
            meta_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), LIGHT_BG),
                ("BACKGROUND", (1, 0), (1, -1), colors.white),
                ("TEXTCOLOR", (0, 0), (0, -1), PRIMARY_COLOR),
                ("TEXTCOLOR", (1, 0), (1, -1), NEUTRAL_TEXT),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 1, BORDER_COLOR),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ]))
            story.append(meta_table)
            story.append(Spacer(1, 0.25*inch))

            # Health Status Summary
            story.append(Paragraph("Health Status Summary", section_heading))
            overall_health = analysis.overall_health or 0
            if overall_health >= 80:
                health_status, health_color, indicator = "Excellent", ACCENT_COLOR, "HEALTHY"
            elif overall_health >= 50:
                health_status, health_color, indicator = "Moderate", WARNING_COLOR, "CAUTION"
            else:
                health_status, health_color, indicator = "Poor", DANGER_COLOR, "CRITICAL"

            summary_data = [["Metric", "Score", "Status"], ["Overall Tree Health", f"{overall_health:.1f}%", f"{health_status} • {indicator}"]]
            summary_table = Table(summary_data, colWidths=[2.5*inch, 1.5*inch, 2.5*inch])
            summary_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_COLOR),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 1), (-1, 1), health_color),
                ("TEXTCOLOR", (0, 1), (-1, 1), colors.white),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 0.25*inch))

            # Leaf Analysis Breakdown
            story.append(Paragraph("Leaf Analysis Breakdown", section_heading))
            detection_data = [
                ["Category", "Count", "Percentage", "Status"],
                ["Healthy Leaves", str(analysis.healthy_count or 0), f"{analysis.healthy_percentage or 0:.1f}%", "GOOD"],
                ["Dried Leaves", str(analysis.dried_leaf_count or 0), f"{analysis.dried_leaf_percentage or 0:.1f}%", "WARN"],
                ["Powdery Mildew", str(analysis.powdery_mildew_count or 0), f"{analysis.powdery_mildew_percentage or 0:.1f}%", "WARN"],
                ["Leaf Rust", str(analysis.leaf_rust_count or 0), f"{analysis.leaf_rust_percentage or 0:.1f}%", "CRITICAL"],
            ]
            detection_table = Table(detection_data, colWidths=[2*inch, 1.2*inch, 1.3*inch, 1.7*inch])
            detection_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), SECONDARY_COLOR),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 1, BORDER_COLOR),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ]))
            story.append(detection_table)
            story.append(Spacer(1, 0.25*inch))

            # Recommendations
            story.append(Paragraph("Recommendations", section_heading))
            if overall_health >= 80:
                recommendation, recommendation_color = (
                    "Tree health is excellent. Continue regular monitoring and maintain current care practices to ensure sustained vitality.",
                    ACCENT_COLOR
                )
            elif overall_health >= 50:
                recommendation, recommendation_color = (
                    "Tree shows moderate concerns. Apply preventive treatments, increase monitoring frequency to weekly inspections, and consider professional consultation.",
                    WARNING_COLOR
                )
            else:
                recommendation, recommendation_color = (
                    "Tree health requires immediate attention. Professional expert consultation is strongly recommended. Implement intensive treatment protocols without delay.",
                    DANGER_COLOR
                )
            rec_table = Table([[Paragraph(recommendation, body_style)]], colWidths=[6.3*inch])
            rec_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F5F5")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("GRID", (0, 0), (-1, -1), 1, recommendation_color),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(rec_table)
            story.append(Spacer(1, 0.3*inch))

            # Footer
            footer_text = f"Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} | Report ID: TREE-{analysis.id}"
            story.append(Paragraph(footer_text, footer_style))

            if idx < len(analyses) - 1:
                story.append(PageBreak())

        doc.build(story)
        pdf = buffer.getvalue()
        buffer.close()
        response.write(pdf)

        logger.info(f"Selected analyses PDF exported: {len(analyses)} analyses")
        return response

    except Exception as e:
        logger.error(f"Error exporting selected analyses PDF: {e}")
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

    # Idagdag ito sa iyong existing views.py
def offline(request):
    return render(request, 'dashboard/offline.html')
