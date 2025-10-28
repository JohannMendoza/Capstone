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

class RegisterForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    confirm_password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password']

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        confirm_password = cleaned_data.get("confirm_password")

        if password and confirm_password and password != confirm_password:
            self.add_error('confirm_password', "Passwords do not match.")
            
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