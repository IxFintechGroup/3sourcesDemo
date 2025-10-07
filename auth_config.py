#!/usr/bin/env python3
"""
Authentication configuration for the Multi-Source OHLCV application.
This file contains the user credentials and authentication settings.
"""

import hashlib

# User credentials (username: hashed_password)
# To add a new user, hash their password using: hashlib.sha256("password".encode()).hexdigest()
VALID_USERS = {
    "admin": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",  # "password"
    "user": "ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f",   # "123456"
    "demo": "5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5",    # "hello"
    "guest": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"     # "" (empty password)
}

def hash_password(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def check_credentials(username, password):
    """Check if the provided credentials are valid."""
    if username in VALID_USERS:
        hashed_password = hash_password(password)
        return hashed_password == VALID_USERS[username]
    return False

def get_demo_credentials():
    """Return demo credentials for display purposes."""
    return {
        "admin": "password",
        "user": "123456", 
        "demo": "hello",
        "guest": ""  # empty password
    }
