"""Fernet symmetric encryption for sensitive token storage."""

import os

from cryptography.fernet import Fernet


def _fernet() -> Fernet:
    key = os.environ.get("ENCRYPTION_KEY", "")
    if not key:
        raise RuntimeError("ENCRYPTION_KEY env var is not set — cannot encrypt/decrypt tokens.")
    return Fernet(key.encode())


def encrypt(plaintext: str) -> str:
    """Encrypt a string and return a URL-safe base64 token."""
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    """Decrypt a Fernet token back to plaintext."""
    return _fernet().decrypt(ciphertext.encode()).decode()
