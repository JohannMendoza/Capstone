from django.contrib.auth.models import AbstractUser
from django.db import models
import uuid
import os
from django.core.validators import FileExtensionValidator
from django.conf import settings

def get_image_path(instance, filename):
    """Generate a unique path for uploaded images"""
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('leaf_images', filename)

class LeafImage(models.Model):
    image = models.ImageField(
        upload_to=get_image_path,
        validators=[FileExtensionValidator(allowed_extensions=['jpg', 'jpeg', 'png'])]
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    # Prediction label and confidence scores
    prediction = models.CharField(max_length=50, blank=True, null=True)
    dried_leaf_confidence = models.FloatField(default=0)  # ✅ NEW
    healthy_confidence = models.FloatField(default=0)
    leaf_rust_confidence = models.FloatField(default=0)
    powdery_mildew_confidence = models.FloatField(default=0)
    
    tree_analysis = models.ForeignKey('TreeAnalysis', on_delete=models.CASCADE, related_name='leaf_images')

    def __str__(self):
        return f"Leaf Image {self.id} - {self.prediction}"

    def delete(self, *args, **kwargs):
        if self.image and os.path.isfile(self.image.path):
            os.remove(self.image.path)
        super().delete(*args, **kwargs)

class TreeAnalysis(models.Model):
    """Model to store tree analysis sessions"""
    plant = models.ForeignKey('dashboard.Plant', on_delete=models.CASCADE, related_name='tree_analyses', null=True, blank=True)
    name = models.CharField(max_length=100, default="Unnamed Tree")
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    is_completed = models.BooleanField(default=False)
    dried_leaf_confidence = models.FloatField(default=0)  # ✅ NEW
    healthy_percentage = models.FloatField(default=0)
    leaf_rust_percentage = models.FloatField(default=0)
    powdery_mildew_percentage = models.FloatField(default=0)
    overall_health = models.FloatField(default=0)  # 0-100 scale
    notes = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = "Tree Analyses"

    def __str__(self):
        return f"{self.name} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

    def calculate_health(self):
        """Calculate detailed health percentages and overall health from leaf image predictions."""
        leaf_images = self.leaf_images.all()
        total_leaves = leaf_images.count()

        if total_leaves == 0:
            self.healthy_percentage = 0
            self.dried_leaf_percentage = 0
            self.leaf_rust_percentage = 0
            self.powdery_mildew_percentage = 0
            self.overall_health = 0
            return 0

        # Count leaves per class
        healthy_count = leaf_images.filter(prediction='Healthy').count()
        dried_leaf_count = leaf_images.filter(prediction='Dried Leaf').count()
        leaf_rust_count = leaf_images.filter(prediction='Leaf Rust').count()
        powdery_mildew_count = leaf_images.filter(prediction='Powdery Mildew').count()

        # Compute percentages
        self.healthy_percentage = (healthy_count / total_leaves) * 100
        self.dried_leaf_percentage = (dried_leaf_count / total_leaves) * 100
        self.leaf_rust_percentage = (leaf_rust_count / total_leaves) * 100
        self.powdery_mildew_percentage = (powdery_mildew_count / total_leaves) * 100

        # Compute overall health score using weighted formula
        self.overall_health = (
            self.healthy_percentage * 1 +
            self.powdery_mildew_percentage * 0.6 +
            self.leaf_rust_percentage * 0.2
        )

        return self.overall_health


# ✅ NEW: Pest Detection Session Model
class PestDetectionSession(models.Model):
    """Model to store pest detection sessions"""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='pest_sessions')
    session_name = models.CharField(max_length=100, default="Pest Detection Session")
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(auto_now=True)
    
    # Summary statistics
    total_processed = models.IntegerField(default=0)
    no_pest_count = models.IntegerField(default=0)
    pest_count = models.IntegerField(default=0)
    high_risk_count = models.IntegerField(default=0)
    uncertain_count = models.IntegerField(default=0)
    avg_processing_time = models.FloatField(default=0.0)
    avg_confidence = models.FloatField(default=0.0)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = "Pest Detection Sessions"

    def __str__(self):
        return f"{self.session_name} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

# ✅ NEW: Individual Pest Detection Result Model
class PestDetectionResult(models.Model):
    """Model to store individual pest detection results"""
    session = models.ForeignKey(PestDetectionSession, on_delete=models.CASCADE, related_name='results')
    filename = models.CharField(max_length=255)
    prediction = models.CharField(max_length=50)
    original_prediction = models.CharField(max_length=50, blank=True, null=True)
    confidence = models.FloatField()
    processing_time = models.FloatField()
    is_low_confidence = models.BooleanField(default=False)
    timestamp = models.DateTimeField()
    
    # Confidence scores for all classes
    adristyrannus_confidence = models.FloatField(default=0.0)
    aphids_confidence = models.FloatField(default=0.0)
    beetle_confidence = models.FloatField(default=0.0)
    bugs_confidence = models.FloatField(default=0.0)
    mites_confidence = models.FloatField(default=0.0)
    weevil_confidence = models.FloatField(default=0.0)
    whitefly_confidence = models.FloatField(default=0.0)

    def __str__(self):
        return f"{self.filename} - {self.prediction} ({self.confidence:.1f}%)"

class CustomUser(AbstractUser):
    ROLE_CHOICES = (
        ('admin', 'Admin'),
        ('client', 'Client'),
    )
    
    email = models.EmailField(unique=True)  # Make email required and unique
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='admin')

    USERNAME_FIELD = 'email'  # Use email for authentication
    REQUIRED_FIELDS = ['username']  # Keep username but not required for login

class Plant(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    plant_id = models.AutoField(primary_key=True)
    plant_number = models.IntegerField(default=0)
    age = models.IntegerField()
    health_status = models.CharField(max_length=255, default='undetected')
    symptoms = models.CharField(max_length=255, blank=True, null=True)
    tree_analysis = models.OneToOneField(
    'dashboard.TreeAnalysis',
    null=True,
    blank=True,
    on_delete=models.SET_NULL,
    related_name='linked_plant'  # ✅ Add this
)

    def save(self, *args, **kwargs):
        symptoms_mapping = {
            "leaf rust": "gawang ulan",
            "dahon": "gawa ng puno",
            "amag": "gawa ng kung ano",
            "insecto": "gawa ng insecto",
            "good": "good plant care",
        }
        self.symptoms = symptoms_mapping.get(self.health_status, "Unknown")
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Plant {self.plant_id} - {self.health_status}"

class PlantInventory(models.Model):
    name = models.CharField(max_length=100)
    plant_number = models.IntegerField(default=0)  # New column for plant numbers
    health_status = models.CharField(
        max_length=20,
        choices=[
            ('leaf rust', 'Leaf Rust'),
            ('dahon', 'Dahon'),
            ('amag', 'Amag'),
            ('insecto', 'Insecto'),
            ('good', 'Good')
        ]
    )
    symptoms = models.CharField(max_length=255, blank=True)  # New column for symptoms

    def save(self, *args, **kwargs):
        # Automatically insert symptoms based on health status
        symptoms_mapping = {
            'leaf rust': 'gawang ulan',
            'dahon': 'gawa ng puno',
            'amag': 'gawa ng kung ano',
            'insecto': 'gawa ng insecto',
            'good': 'good plant care'
        }
        self.symptoms = symptoms_mapping.get(self.health_status, "")
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name
