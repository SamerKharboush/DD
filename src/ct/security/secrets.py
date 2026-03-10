"""
Secrets Management for CellType-Agent.

Provides secure handling of secrets with:
- Environment variable fallbacks
- File-based secret storage
- Vault integration (optional)
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("ct.security.secrets")


@dataclass
class SecretConfig:
    """Secret configuration."""
    name: str
    description: str
    required: bool = True
    default: Optional[str] = None


# Required secrets configuration
REQUIRED_SECRETS = [
    SecretConfig("ANTHROPIC_API_KEY", "Claude API key", required=True),
    SecretConfig("JWT_SECRET", "JWT signing secret", required=False, default=None),
    SecretConfig("NEO4J_PASSWORD", "Neo4j database password", required=True),
    SecretConfig("DATABASE_URL", "PostgreSQL connection string", required=False),
    SecretConfig("REDIS_URL", "Redis connection URL", required=False),
    SecretConfig("ESM3_API_KEY", "ESM3 API key", required=False),
]


class SecretsManager:
    """
    Secrets management with multiple backends.

    Priority order:
    1. Environment variables
    2. Secrets file (~/.ct/secrets.json)
    3. Vault (if configured)
    4. Default values

    Usage:
        secrets = SecretsManager()
        api_key = secrets.get("ANTHROPIC_API_KEY")
        secrets.set("MY_SECRET", "secret_value")
    """

    def __init__(
        self,
        secrets_file: Optional[Path] = None,
        vault_url: Optional[str] = None,
    ):
        """
        Initialize secrets manager.

        Args:
            secrets_file: Path to secrets file
            vault_url: HashiCorp Vault URL (optional)
        """
        self.secrets_file = secrets_file or Path.home() / ".ct" / "secrets.json"
        self.vault_url = vault_url or os.environ.get("VAULT_URL")
        self._file_secrets: dict[str, str] = {}
        self._cache: dict[str, str] = {}

        # Load file secrets
        self._load_file_secrets()

    def get(
        self,
        name: str,
        default: Optional[str] = None,
        required: bool = False,
    ) -> Optional[str]:
        """
        Get a secret value.

        Args:
            name: Secret name
            default: Default value if not found
            required: Raise error if not found

        Returns:
            Secret value or default
        """
        # Check cache first
        if name in self._cache:
            return self._cache[name]

        # Check environment variable
        value = os.environ.get(name)

        # Check file secrets
        if value is None:
            value = self._file_secrets.get(name)

        # Check Vault
        if value is None and self.vault_url:
            value = self._get_from_vault(name)

        # Use default
        if value is None:
            value = default

        if value is None and required:
            raise ValueError(f"Required secret '{name}' not found")

        # Cache value
        if value is not None:
            self._cache[name] = value

        return value

    def set(
        self,
        name: str,
        value: str,
        persist: bool = False,
    ):
        """
        Set a secret value.

        Args:
            name: Secret name
            value: Secret value
            persist: Save to secrets file
        """
        self._cache[name] = value

        if persist:
            self._file_secrets[name] = value
            self._save_file_secrets()

    def delete(self, name: str, persist: bool = False):
        """
        Delete a secret.

        Args:
            name: Secret name
            persist: Remove from secrets file
        """
        self._cache.pop(name, None)

        if persist:
            self._file_secrets.pop(name, None)
            self._save_file_secrets()

    def validate_secrets(self) -> dict[str, bool]:
        """
        Validate all required secrets are available.

        Returns:
            Dictionary of secret name -> is_valid
        """
        results = {}

        for secret in REQUIRED_SECRETS:
            value = self.get(secret.name, default=secret.default)

            if secret.required and not value:
                results[secret.name] = False
                logger.warning(f"Missing required secret: {secret.name}")
            else:
                results[secret.name] = True

        return results

    def list_secrets(self) -> list[str]:
        """
        List available secret names.

        Returns:
            List of secret names
        """
        names = set()
        names.update(self._file_secrets.keys())
        names.update(k for k in os.environ.keys() if k.isupper())
        return sorted(names)

    def _load_file_secrets(self):
        """Load secrets from file."""
        if self.secrets_file.exists():
            try:
                with open(self.secrets_file) as f:
                    self._file_secrets = json.load(f)
                logger.debug(f"Loaded {len(self._file_secrets)} secrets from file")
            except Exception as e:
                logger.warning(f"Failed to load secrets file: {e}")
                self._file_secrets = {}

    def _save_file_secrets(self):
        """Save secrets to file."""
        self.secrets_file.parent.mkdir(parents=True, exist_ok=True)

        # Set restrictive permissions
        with open(self.secrets_file, "w") as f:
            json.dump(self._file_secrets, f, indent=2)

        # Set file permissions (Unix only)
        try:
            os.chmod(self.secrets_file, 0o600)
        except Exception:
            pass

        logger.debug(f"Saved {len(self._file_secrets)} secrets to file")

    def _get_from_vault(self, name: str) -> Optional[str]:
        """Get secret from HashiCorp Vault."""
        if not self.vault_url:
            return None

        try:
            import requests

            vault_token = os.environ.get("VAULT_TOKEN")
            if not vault_token:
                return None

            response = requests.get(
                f"{self.vault_url}/v1/secret/data/{name}",
                headers={"X-Vault-Token": vault_token},
                timeout=5,
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("data", {}).get("data", {}).get("value")

        except Exception as e:
            logger.debug(f"Vault lookup failed for {name}: {e}")

        return None


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to get a secret.

    Args:
        name: Secret name
        default: Default value

    Returns:
        Secret value or default
    """
    manager = SecretsManager()
    return manager.get(name, default=default)


def set_secret(name: str, value: str, persist: bool = False):
    """
    Convenience function to set a secret.

    Args:
        name: Secret name
        value: Secret value
        persist: Save to file
    """
    manager = SecretsManager()
    manager.set(name, value, persist=persist)