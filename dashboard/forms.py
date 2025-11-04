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
        fields = ['plant_number', 'age']  # Changed location to plant_number

User = get_user_model()

class CustomPasswordResetForm(PasswordResetForm):
    def send_mail(self, subject_template_name, email_template_name,
                  context, from_email, to_email, html_email_template_name=None):

        subject = "Password Reset | Escala Plants & Nursery"
        text_content = render_to_string('registration/password_reset_email.txt', context)
        html_content = render_to_string('registration/password_reset_email.html', context)

        email_message = EmailMultiAlternatives(subject, text_content, from_email, [to_email])
        email_message.attach_alternative(html_content, "text/html")
        email_message.send()
