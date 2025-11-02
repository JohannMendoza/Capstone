from django.core.mail import EmailMessage
from django.contrib.auth.tokens import PasswordResetTokenGenerator
from django.conf import settings

default_token_generator = PasswordResetTokenGenerator()

def send_verification_email(subject, body, recipient):
    """
    Send HTML verification email to user
    """
    try:
        email = EmailMessage(
            subject=subject,
            body=body,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[recipient]
        )
        email.content_subtype = "html"
        email.send(fail_silently=False)
        print(f"[v0] Email sent successfully to {recipient}")
    except Exception as e:
        print(f"[v0] Error sending email to {recipient}: {str(e)}")
