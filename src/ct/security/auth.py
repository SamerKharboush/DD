"""
Authentication Module for CellType-Agent.

Implements JWT-based authentication with:
- Access token generation
- Token verification
- Role-based access control
"""

import datetime
import hashlib
import logging
import os
import secrets
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("ct.security.auth")

# Secret key for JWT signing
JWT_SECRET = os.environ.get("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


@dataclass
class User:
    """User model."""
    user_id: str
    username: str
    email: Optional[str] = None
    role: str = "user"
    is_active: bool = True
    created_at: Optional[datetime.datetime] = None


@dataclass
class TokenData:
    """Token payload data."""
    user_id: str
    username: str
    role: str
    exp: datetime.datetime


class AuthManager:
    """
    Authentication manager.

    Handles:
    - User registration
    - Login/logout
    - Token generation
    - Token verification

    Usage:
        auth = AuthManager()
        token = auth.login("user", "password")
        user = auth.verify(token)
    """

    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize auth manager.

        Args:
            secret_key: JWT secret key (uses env var if not provided)
        """
        self.secret_key = secret_key or JWT_SECRET
        self.algorithm = JWT_ALGORITHM
        self._users: dict[str, dict] = {}  # In-memory user store
        self._tokens: dict[str, str] = {}  # Token blacklist

    def register_user(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
        role: str = "user",
    ) -> User:
        """
        Register a new user.

        Args:
            username: Username
            password: Password
            email: Optional email
            role: User role

        Returns:
            Created User object
        """
        if username in self._users:
            raise ValueError(f"User '{username}' already exists")

        # Hash password
        password_hash = self._hash_password(password)

        user_id = secrets.token_hex(8)
        user_data = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "role": role,
            "password_hash": password_hash,
            "is_active": True,
            "created_at": datetime.datetime.utcnow(),
        }

        self._users[username] = user_data

        return User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            is_active=True,
            created_at=user_data["created_at"],
        )

    def login(self, username: str, password: str) -> Optional[str]:
        """
        Authenticate user and generate token.

        Args:
            username: Username
            password: Password

        Returns:
            JWT token if successful, None otherwise
        """
        user = self._users.get(username)
        if not user:
            return None

        if not user["is_active"]:
            return None

        if not self._verify_password(password, user["password_hash"]):
            return None

        return create_access_token(
            user_id=user["user_id"],
            username=username,
            role=user["role"],
            secret_key=self.secret_key,
        )

    def verify(self, token: str) -> Optional[User]:
        """
        Verify token and get user.

        Args:
            token: JWT token

        Returns:
            User if valid, None otherwise
        """
        # Check blacklist
        if token in self._tokens:
            return None

        token_data = verify_token(token, self.secret_key)
        if not token_data:
            return None

        user_data = self._users.get(token_data.username)
        if not user_data:
            return None

        return User(
            user_id=user_data["user_id"],
            username=user_data["username"],
            email=user_data.get("email"),
            role=user_data["role"],
            is_active=user_data["is_active"],
            created_at=user_data.get("created_at"),
        )

    def logout(self, token: str) -> bool:
        """
        Invalidate a token.

        Args:
            token: JWT token to invalidate

        Returns:
            True if successful
        """
        self._tokens[token] = "revoked"
        return True

    def _hash_password(self, password: str) -> str:
        """Hash a password."""
        salt = secrets.token_hex(16)
        hash_value = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            salt.encode(),
            100000,
        )
        return f"{salt}:{hash_value.hex()}"

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against hash."""
        try:
            salt, stored_hash = password_hash.split(":")
            hash_value = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode(),
                salt.encode(),
                100000,
            )
            return hash_value.hex() == stored_hash
        except Exception:
            return False

    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        user_data = self._users.get(username)
        if not user_data:
            return None

        return User(
            user_id=user_data["user_id"],
            username=user_data["username"],
            email=user_data.get("email"),
            role=user_data["role"],
            is_active=user_data["is_active"],
        )

    def list_users(self) -> list[User]:
        """List all users."""
        return [
            User(
                user_id=u["user_id"],
                username=u["username"],
                email=u.get("email"),
                role=u["role"],
                is_active=u["is_active"],
            )
            for u in self._users.values()
        ]


def create_access_token(
    user_id: str,
    username: str,
    role: str = "user",
    secret_key: Optional[str] = None,
    expires_hours: int = JWT_EXPIRATION_HOURS,
) -> str:
    """
    Create a JWT access token.

    Args:
        user_id: User ID
        username: Username
        role: User role
        secret_key: Secret key for signing
        expires_hours: Token expiration in hours

    Returns:
        JWT token string
    """
    import base64
    import json

    secret = secret_key or JWT_SECRET

    # Create header
    header = {
        "alg": JWT_ALGORITHM,
        "typ": "JWT",
    }

    # Create payload
    now = datetime.datetime.utcnow()
    payload = {
        "user_id": user_id,
        "username": username,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + datetime.timedelta(hours=expires_hours)).timestamp()),
    }

    # Encode header and payload
    header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")

    # Create signature
    message = f"{header_b64}.{payload_b64}"
    signature = hashlib.sha256((message + secret).encode()).hexdigest()
    signature_b64 = base64.urlsafe_b64encode(signature.encode()).decode().rstrip("=")

    return f"{message}.{signature_b64}"


def verify_token(token: str, secret_key: Optional[str] = None) -> Optional[TokenData]:
    """
    Verify a JWT token.

    Args:
        token: JWT token string
        secret_key: Secret key for verification

    Returns:
        TokenData if valid, None otherwise
    """
    import base64
    import json

    try:
        secret = secret_key or JWT_SECRET
        parts = token.split(".")

        if len(parts) != 3:
            return None

        header_b64, payload_b64, signature_b64 = parts

        # Verify signature
        message = f"{header_b64}.{payload_b64}"
        expected_sig = hashlib.sha256((message + secret).encode()).hexdigest()
        expected_sig_b64 = base64.urlsafe_b64encode(expected_sig.encode()).decode().rstrip("=")

        if signature_b64 != expected_sig_b64:
            return None

        # Decode payload
        # Add padding if needed
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding

        payload = json.loads(base64.urlsafe_b64decode(payload_b64))

        # Check expiration
        now = datetime.datetime.utcnow()
        exp = datetime.datetime.fromtimestamp(payload.get("exp", 0))

        if now > exp:
            return None

        return TokenData(
            user_id=payload["user_id"],
            username=payload["username"],
            role=payload.get("role", "user"),
            exp=exp,
        )

    except Exception as e:
        logger.debug(f"Token verification failed: {e}")
        return None


def get_current_user(token: str, auth_manager: AuthManager) -> Optional[User]:
    """
    Get current user from token.

    Args:
        token: JWT token
        auth_manager: Auth manager instance

    Returns:
        User if valid, None otherwise
    """
    return auth_manager.verify(token)


def require_role(role: str):
    """
    Decorator to require a specific role.

    Args:
        role: Required role

    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get user from kwargs or args
            user = kwargs.get("user") or (args[0] if args else None)

            if not user:
                raise PermissionError("Authentication required")

            if not isinstance(user, User):
                raise PermissionError("Invalid user object")

            if user.role != role and user.role != "admin":
                raise PermissionError(f"Role '{role}' required")

            return func(*args, **kwargs)

        return wrapper
    return decorator