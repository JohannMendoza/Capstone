from django import forms
from .models import CustomUser
from .models import Plant
from django.contrib.auth.forms import PasswordResetForm
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes
from django.contrib.auth.tokens import default_token_generator
from django.contrib.auth import get_user_model

from django import forms
from django.contrib.auth import get_user_model
from django.core.validators import validate_email
from django.core.exceptions import ValidationError

CustomUser = get_user_model()

class RegisterForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput, required=True)
    password2 = forms.CharField(widget=forms.PasswordInput, required=True, label="Confirm Password")
    
    class Meta:
        model = CustomUser
        fields = ['username', 'email']
    
    def clean_email(self):
        email = self.cleaned_data.get('email', '').strip().lower()
        
        # Validate email format
        try:
            validate_email(email)
        except ValidationError:
            raise forms.ValidationError("Please enter a valid email address.")
        
        # Check if email already exists
        if CustomUser.objects.filter(email=email).exists():
            raise forms.ValidationError("This email is already registered.")
        
        return email
    
    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get('password')
        password2 = cleaned_data.get('password2')
        
        if password and password2 and password != password2:
            raise forms.ValidationError("Passwords do not match.")
        
        return cleaned_data
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data['password'])
        # <CHANGE> Add these two critical lines
        user.role = 'client'  # Set default role to client
        user.is_active = False  # Require email verification
        if commit:
            user.save()
        return user

            
class PlantForm(forms.ModelForm):
    class Meta:
        model = Plant
        fields = ['plant_number', 'age']
        widgets = {
            'plant_number': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 rounded-lg bg-white focus:ring-2 focus:ring-green-500',
                'min': '1',
                'step': '1',
                'placeholder': 'Enter plant number',
                'required': 'required'
            }),
            'age': forms.NumberInput(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 rounded-lg bg-white focus:ring-2 focus:ring-green-500',
                'min': '0',
                'step': '1',
                'placeholder': 'Enter age in years',
                'required': 'required'
            })
        }
    
    def clean_plant_number(self):
        plant_number = self.cleaned_data.get('plant_number')
        if plant_number is None or plant_number == '':
            raise forms.ValidationError("Plant number is required.")
        try:
            plant_number = int(plant_number)
        except (ValueError, TypeError):
            raise forms.ValidationError("Plant number must be a valid number.")
        if plant_number < 1:
            raise forms.ValidationError("Plant number must be a positive number (1 or higher).")
        return plant_number
    
    def clean_age(self):
        age = self.cleaned_data.get('age')
        if age is None or age == '':
            raise forms.ValidationError("Age is required.")
        try:
            age = int(age)
        except (ValueError, TypeError):
            raise forms.ValidationError("Age must be a valid number.")
        if age < 0:
            raise forms.ValidationError("Age must be zero or a positive number.")
        if age > 150:
            raise forms.ValidationError("Please enter a realistic age (0-150 years).")
        return age

class CustomPasswordResetForm(PasswordResetForm):
    def send_mail(self, subject_template_name, email_template_name,
                  context, from_email, to_email, html_email_template_name=None):

        subject = "Password Reset | Escala Plants & Nursery"
        text_content = render_to_string('registration/password_reset_email.txt', context)
        html_content = render_to_string('registration/password_reset_email.html', context)

        email_message = EmailMultiAlternatives(subject, text_content, from_email, [to_email])
        email_message.attach_alternative(html_content, "text/html")
        email_message.send()


        # forms.py - Add this form
from django import forms
from django.contrib.auth import get_user_model
from django.core.validators import validate_email
from django.core.exceptions import ValidationError

CustomUser = get_user_model()

class UserEditForm(forms.ModelForm):
    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('client', 'Client'),
    ]
    
    STATUS_CHOICES = [
        (True, 'Active'),
        (False, 'Inactive'),
    ]
    
    role = forms.ChoiceField(
        choices=ROLE_CHOICES,
        widget=forms.Select(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 rounded-lg bg-white focus:ring-2 focus:ring-green-500 focus:border-transparent',
        })
    )
    
    is_active = forms.ChoiceField(
        choices=STATUS_CHOICES,
        widget=forms.Select(attrs={
            'class': 'w-full px-4 py-3 border border-gray-300 rounded-lg bg-white focus:ring-2 focus:ring-green-500 focus:border-transparent',
        }),
        label="Account Status"
    )
    
    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'role', 'is_active']
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 rounded-lg bg-white focus:ring-2 focus:ring-green-500 focus:border-transparent',
                'placeholder': 'Enter username'
            }),
            'email': forms.EmailInput(attrs={
                'class': 'w-full px-4 py-3 border border-gray-300 rounded-lg bg-white focus:ring-2 focus:ring-green-500 focus:border-transparent',
                'placeholder': 'Enter email address'
            }),
        }
    
    def clean_email(self):
        email = self.cleaned_data.get('email', '').strip().lower()
        
        # Validate email format
        try:
            validate_email(email)
        except ValidationError:
            raise forms.ValidationError("Please enter a valid email address.")
        
        # Check if email already exists (excluding current user)
        if CustomUser.objects.filter(email=email).exclude(id=self.instance.id).exists():
            raise forms.ValidationError("This email is already registered to another user.")
        
        return email
    
    def clean_username(self):
        username = self.cleaned_data.get('username')
        
        # Check if username already exists (excluding current user)
        if CustomUser.objects.filter(username=username).exclude(id=self.instance.id).exists():
            raise forms.ValidationError("This username is already taken.")
        
        return username
