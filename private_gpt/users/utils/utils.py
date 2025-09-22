import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from private_gpt.users.core.config import settings
from fastapi import HTTPException, status

def send_registration_email(fullname: str, email: str, random_password: str) -> None:
    """
    Send a registration email with a random password.
    """
    subject = "Welcome to QuickRef - Registration Successful"
    body = f"""
        <html>
        <body style="margin:0; padding:0; background-color:#f4f4f7; font-family:Arial, Helvetica, sans-serif;">
            <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f4f4f7; padding:20px 0;">
            <tr>
                <td align="center">
                <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff; border-radius:8px; overflow:hidden; box-shadow:0 2px 6px rgba(0,0,0,0.1);">
                    <!-- Header -->
                    <tr>
                    <td style="background-color:#4f46e5; padding:20px; text-align:center; color:#ffffff; font-size:22px; font-weight:bold;">
                        QuickRef
                    </td>
                    </tr>
                    <!-- Body -->
                    <tr>
                    <td style="padding:30px; color:#333333; font-size:16px; line-height:1.5;">
                        <p style="margin:0 0 16px 0;">Hello {fullname},</p>
                        <p style="margin:0 0 16px 0;">Thank you for registering with <strong>QuickRef</strong>!</p>
                        <p style="margin:0 0 16px 0;">Your temporary password is:</p>
                        <p style="margin:0 0 24px 0; font-size:18px; font-weight:bold; color:#4f46e5;">
                        {random_password}
                        </p>
                        <p style="margin:0 0 24px 0;">Please log in to QuickRef using the button below and consider changing it to a more secure password after logging in.</p>
                        <!-- CTA button -->
                        <p style="text-align:center; margin:0 0 30px 0;">
                        <a href="https://quickref.quickfoxconsulting.com" 
                            style="background-color:#4f46e5; color:#ffffff; padding:12px 24px; text-decoration:none; border-radius:4px; font-weight:bold; display:inline-block;">
                            Log In to QuickRef
                        </a>
                        </p>
                        <p style="margin:0 0 0 0;">Best regards,<br>QuickRef Team</p>
                    </td>
                    </tr>
                    <!-- Footer -->
                    <tr>
                    <td style="background-color:#f0f0f0; padding:15px; text-align:center; font-size:12px; color:#888888;">
                        © {2025} QuickRef · All rights reserved.
                    </td>
                    </tr>
                </table>
                </td>
            </tr>
            </table>
        </body>
        </html>
    """

    msg = MIMEMultipart()
    msg.attach(MIMEText(body, "html"))
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_SENDER_EMAIL
    msg["To"] = email

    print(settings.SMTP_SERVER)
    print(settings.SMTP_PORT)
    
    try:
        with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT) as server:
            server.starttls()
            server.login(settings.SMTP_SENDER_EMAIL, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_SENDER_EMAIL, email, msg.as_string())
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to send email."
        )
    

def validate_password(password: str) -> None:
    """
    Validate the password according to the defined criteria.
    
    Args:
        password (str): The new password to validate.
        
    Raises:
        HTTPException: If the password does not meet the criteria.
    """
    # Define the password validation criteria
    min_length = 6
    require_upper = re.compile(r'[A-Z]')
    require_lower = re.compile(r'[a-z]')
    require_digit = re.compile(r'\d')
    require_special = re.compile(r'[!@#$%^&*()_+=-]')  # Add special characters as needed

    # Check password length
    if len(password) < min_length:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Password must be at least {min_length} characters long.")

    # Check for uppercase letter
    if not require_upper.search(password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password must contain at least one uppercase letter.")

    # Check for lowercase letter
    if not require_lower.search(password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password must contain at least one lowercase letter.")

    # Check for digit
    if not require_digit.search(password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password must contain at least one digit.")

    # Check for special character
    if not require_special.search(password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password must contain at least one special character (e.g., !@#$%^&*()_+=-).")
