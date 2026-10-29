"""AES-GCM storage for datasource credentials.

The key comes from ``DATASOURCE_SECRET_KEY`` (base64 or passphrase) or from a
locally generated key file next to the datasource database. Nothing here is
ever logged; callers expose only a per-field ``configured`` boolean.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from datasource.errors import CredentialsUnreadable

logger = logging.getLogger("ai-service.datasource.secrets")

_NONCE_BYTES = 12
_KEY_BYTES = 32


def _decode_key(value: str) -> bytes:
    raw = value.strip()
    try:
        decoded = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    except Exception:
        decoded = b""
    if len(decoded) == _KEY_BYTES:
        return decoded
    return hashlib.sha256(raw.encode("utf-8")).digest()


class SecretBox:
    def __init__(self, key: bytes, key_path: Path | None = None) -> None:
        if len(key) != _KEY_BYTES:
            raise ValueError("secret key must be 32 bytes")
        self._aes = AESGCM(key)
        self.key_path = key_path

    @classmethod
    def from_settings(cls, secret_key: str = "", key_path: Path | str | None = None) -> "SecretBox":
        if secret_key:
            return cls(_decode_key(secret_key))
        if key_path is None:
            raise ValueError("key_path is required when DATASOURCE_SECRET_KEY is empty")
        path = Path(key_path)
        if path.is_file():
            data = path.read_bytes()
            if len(data) != _KEY_BYTES:
                raise CredentialsUnreadable(f"secret key file has invalid length: {path}")
            return cls(data, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        key = os.urandom(_KEY_BYTES)
        path.write_bytes(key)
        try:
            os.chmod(path, 0o600)
        except OSError:
            logger.warning("could not restrict datasource key file permissions: %s", path)
        return cls(key, path)

    @classmethod
    def from_key_bytes(cls, key: bytes) -> "SecretBox":
        return cls(key)

    def encrypt(self, values: Mapping[str, Any]) -> str:
        payload = json.dumps(dict(values), ensure_ascii=False, sort_keys=True).encode("utf-8")
        nonce = os.urandom(_NONCE_BYTES)
        ciphertext = self._aes.encrypt(nonce, payload, None)
        return base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")

    def decrypt(self, token: str) -> dict[str, Any]:
        try:
            raw = base64.urlsafe_b64decode(token.encode("ascii"))
            nonce, ciphertext = raw[:_NONCE_BYTES], raw[_NONCE_BYTES:]
            payload = self._aes.decrypt(nonce, ciphertext, None)
            data = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise CredentialsUnreadable("凭据无法解密，请重新填写") from exc
        if not isinstance(data, dict):
            raise CredentialsUnreadable("凭据格式无效，请重新填写")
        return data
