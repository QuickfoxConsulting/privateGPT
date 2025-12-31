#!/usr/bin/env python3
"""
Script to generate and validate Fernet encryption keys for Google Drive integration.
Run this to generate a valid INTEGRATION_ENCRYPTION_KEY.
"""

from cryptography.fernet import Fernet
import os
import sys


def generate_key():
    """Generate a new Fernet encryption key."""
    key = Fernet.generate_key()
    return key.decode()


def validate_key(key_string):
    """Validate if a key string is a valid Fernet key."""
    try:
        # Try to create a Fernet cipher with the key
        key_bytes = key_string.encode() if isinstance(key_string, str) else key_string
        cipher = Fernet(key_bytes)
        
        # Test encryption/decryption
        test_data = b"test_encryption"
        encrypted = cipher.encrypt(test_data)
        decrypted = cipher.decrypt(encrypted)
        
        if decrypted == test_data:
            return True, "Valid Fernet key"
        else:
            return False, "Encryption/decryption mismatch"
    except Exception as e:
        return False, f"Invalid key: {str(e)}"


def main():
    print("=" * 70)
    print("Fernet Encryption Key Generator")
    print("=" * 70)
    print()
    
    # Check if there's an existing key in environment
    existing_key = os.getenv("INTEGRATION_ENCRYPTION_KEY")
    
    if existing_key:
        print("📋 Found existing INTEGRATION_ENCRYPTION_KEY in environment")
        print(f"   Value: {existing_key[:20]}..." if len(existing_key) > 20 else f"   Value: {existing_key}")
        print()
        
        is_valid, message = validate_key(existing_key)
        if is_valid:
            print("✅ Current key is VALID and working!")
            print()
            print("Your key is correctly configured. The error might be elsewhere.")
            return 0
        else:
            print(f"❌ Current key is INVALID: {message}")
            print()
    else:
        print("⚠️  No INTEGRATION_ENCRYPTION_KEY found in environment")
        print()
    
    # Generate new key
    print("Generating new encryption key...")
    new_key = generate_key()
    
    print()
    print("=" * 70)
    print("✅ NEW ENCRYPTION KEY GENERATED")
    print("=" * 70)
    print()
    print("Copy this key to your .env file:")
    print()
    print(f"INTEGRATION_ENCRYPTION_KEY={new_key}")
    print()
    print("=" * 70)
    print()
    
    # Validate the new key
    is_valid, message = validate_key(new_key)
    if is_valid:
        print("✅ New key validated successfully!")
    else:
        print(f"❌ Validation failed: {message}")
        return 1
    
    print()
    print("IMPORTANT STEPS:")
    print("1. Copy the key above to your .env file")
    print("2. Restart your application")
    print("3. Keep this key secure and backed up")
    print("4. Never commit this key to version control")
    print()
    print("⚠️  WARNING: If you change this key after storing encrypted data,")
    print("   all previously encrypted tokens will be unrecoverable!")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())