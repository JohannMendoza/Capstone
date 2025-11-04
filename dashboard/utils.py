# utils.py
from django.core.mail import EmailMessage
from django.template.loader import render_to_string
from django.conf import settings

def send_verification_email(subject, body, recipient):
    """
    Send HTML verification email to user via SendGrid
    """
    try:
        email = EmailMessage(
            subject=subject,
            body=body,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[recipient]
        )
        email.content_subtype = "html"
        result = email.send(fail_silently=False)
        print(f"✅ [v0] Email sent successfully to {recipient}")
        return True
    except Exception as e:
        print(f"❌ [v0] Error sending email to {recipient}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
