"""
API Key Management for CellType-Agent.

Implements API key authentication for programmatic access.
"""

import hashlib
import logging
import os
import secrets
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("ct.security.api_keys")


@dataclass
class APIKey:
    """API key model."""
    key_id: str
    key_hash: str
    name: str
    user_id: str
    scopes: list[str]
    rate_limit: int  # Requests per minute
    created_at: float
    expires_at: Optional[float] = None
    last_used: Optional[float] = None
    is_active: bool = True


class APIKeyManager:
    """
    API Key management.

    Features:
    - Key generation with prefixes
    - Key hashing for security
    - Scope-based access control
    - Rate limiting per key
    - Key expiration

    Usage:
        manager = APIKeyManager()
        key = manager.create_key("my-app", "user-123", ["read", "write"])
        valid = manager.verify_key(key)
    """

    # Key prefix for identification
    KEY_PREFIX = "ct_"

    def __init__(self):
        """Initialize API key manager."""
        self._keys: dict[str, APIKey] = {}  # key_id -> APIKey
        self._key_hashes: dict[str, str] = {}  # hash -> key_id

    def create_key(
        self,
        name: str,
        user_id: str,
        scopes: Optional[list[str]] = None,
        rate_limit: int = 60,
        expires_days: Optional[int] = None,
    ) -> str:
        """
        Create a new API key.

        Args:
            name: Key name/description
            user_id: Owner user ID
            scopes: Permission scopes
            rate_limit: Requests per minute
            expires_days: Days until expiration (None = no expiration)

        Returns:
            Generated API key (store this securely!)
        """
        # Generate key
        key_secret = secrets.token_hex(32)
        key_id = secrets.token_hex(8)
        full_key = f"{self.KEY_PREFIX}{key_id}_{key_secret}"

        # Hash key for storage
        key_hash = self._hash_key(full_key)

        # Calculate expiration
        expires_at = None
        if expires_days:
            expires_at = time.time() + (expires_days * 86400)

        # Store key
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            scopes=scopes or ["read"],
            rate_limit=rate_limit,
            created_at=time.time(),
            expires_at=expires_at,
        )

        self._keys[key_id] = api_key
        self._key_hashes[key_hash] = key_id

        logger.info(f"Created API key '{name}' for user {user_id}")

        return full_key

    def verify_key(
        self,
        key: str,
        required_scope: Optional[str] = None,
    ) -> Optional[APIKey]:
        """
        Verify an API key.

        Args:
            key: API key string
            required_scope: Optional required scope

        Returns:
            APIKey if valid, None otherwise
        """
        # Check format
        if not key or not key.startswith(self.KEY_PREFIX):
            return None

        # Hash key
        key_hash = self._hash_key(key)

        # Look up by hash
        key_id = self._key_hashes.get(key_hash)
        if not key_id:
            return None

        api_key = self._keys.get(key_id)
        if not api_key:
            return None

        # Check active
        if not api_key.is_active:
            return None

        # Check expiration
        if api_key.expires_at and time.time() > api_key.expires_at:
            return None

        # Check scope
        if required_scope and required_scope not in api_key.scopes:
            return None

        # Update last used
        api_key.last_used = time.time()

        return api_key

    def revoke_key(self, key_id: str) -> bool:
        """
        Revoke an API key.

        Args:
            key_id: Key ID to revoke

        Returns:
            True if successful
        """
        if key_id in self._keys:
            self._keys[key_id].is_active = False
            logger.info(f"Revoked API key {key_id}")
            return True
        return False

    def delete_key(self, key_id: str) -> bool:
        """
        Delete an API key.

        Args:
            key_id: Key ID to delete

        Returns:
            True if successful
        """
        if key_id in self._keys:
            api_key = self._keys[key_id]
            del self._key_hashes[api_key.key_hash]
            del self._keys[key_id]
            logger.info(f"Deleted API key {key_id}")
            return True
        return False

    def list_keys(self, user_id: Optional[str] = None) -> list[APIKey]:
        """
        List API keys.

        Args:
            user_id: Filter by user ID

        Returns:
            List of API keys
        """
        keys = list(self._keys.values())
        if user_id:
            keys = [k for k in keys if k.user_id == user_id]
        return keys

    def _hash_key(self, key: str) -> str:
        """Hash an API key."""
        return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key(name: str = "default") -> str:
    """
    Generate a simple API key.

    Args:
        name: Key name

    Returns:
        Generated API key
    """
    prefix = "ct_"
    random_part = secrets.token_hex(32)
    return f"{prefix}{random_part}"


def validate_api_key_format(key: str) -> bool:
    """
    Validate API key format.

    Args:
        key: API key string

    Returns:
        True if valid format
    """
    if not key:
        return False

    if not key.startswith("ct_"):
        return False

    parts = key[3:].split("_")
    if len(parts) != 2:
        return False

    key_id, key_secret = parts

    if len(key_id) != 16:  # 8 bytes = 16 hex chars
        return False

    if len(key_secret) != 64:  # 32 bytes = 64 hex chars
        return False

    return True