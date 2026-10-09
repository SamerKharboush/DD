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
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("ct.security.auth")

# Secret key for JWT signing.
#
# Previously this defaulted to secrets.token_hex(32) *per process*, which is
# worse than having no secret at all: every worker signed with a different key,
# and every restart invalidated all outstanding tokens. A silently-rotating
# signing key produces confusing 401s rather than an obvious failure.
#
# Now it is persisted once to ~/.ct/jwt_secret with 0600 permissions and reused
# across restarts. Set JWT_SECRET in the environment (as docker-compose should)
# to manage it externally instead.
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


def _load_jwt_secret() -> str:
    """Resolve the JWT signing key, persisting a generated one if needed."""
    env_secret = os.environ.get("JWT_SECRET")
    if env_secret:
        return env_secret

    secret_path = Path(os.path.expanduser("~/.ct/jwt_secret"))
    try:
        if secret_path.exists():
            existing = secret_path.read_text(encoding="utf-8").strip()
            if existing:
                return existing
        secret_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        generated = secrets.token_hex(32)
        # Create with 0600 from the start rather than chmod-ing after.
        fd = os.open(str(secret_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(generated)
        logger.info("Generated new JWT signing key at %s", secret_path)
        return generated
    except Exception as exc:
        # Fail loudly: a per-process key is a security-relevant degradation.
        logger.error(
            "Could not persist JWT secret (%s: %s); falling back to a "
            "process-local key. Tokens will not survive a restart.",
            type(exc).__name__, exc,
        )
        return secrets.token_hex(32)


JWT_SECRET = _load_jwt_secret()


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
    """Create a signed JWT access token.

    Uses PyJWT with HS256. The previous implementation hand-rolled the
    signature as ``sha256(header || payload || secret)``, which is not a MAC:
    it is vulnerable to length-extension forgery, did not enforce ``alg`` on
    verification, and compared signatures with ``!=`` (a non-constant-time
    string comparison).

    Returns:
        JWT token string
    """
    import jwt as _jwt

    secret = secret_key or JWT_SECRET

    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "user_id": user_id,
        "username": username,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + datetime.timedelta(hours=expires_hours)).timestamp()),
    }

    return _jwt.encode(payload, secret, algorithm="HS256")


def verify_token(token: str, secret_key: Optional[str] = None) -> Optional[TokenData]:
    """Verify a JWT token.

    Uses PyJWT with an explicit ``algorithms`` allowlist. Passing
    ``algorithms=["HS256"]`` is what prevents an attacker from submitting a
    token whose header claims ``alg: none`` (or an asymmetric algorithm) and
    having it accepted without a signature check.

    Returns:
        TokenData if valid, None otherwise.
    """
    import jwt as _jwt

    try:
        secret = secret_key or JWT_SECRET

        # Raises on: bad signature, expired, malformed, wrong algorithm.
        payload = _jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            options={"require": ["exp", "iat"]},
        )

        exp = datetime.datetime.fromtimestamp(payload["exp"], tz=datetime.timezone.utc)

        return TokenData(
            user_id=payload["user_id"],
            username=payload["username"],
            role=payload.get("role", "user"),
            exp=exp,
        )

    except Exception as e:
        # Never leak the reason to the caller; log server-side only.
        logger.debug(f"Token verification failed: {type(e).__name__}")
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