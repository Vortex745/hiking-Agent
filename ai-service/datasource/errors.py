"""Stable connector error codes.

Vendor response bodies and tokens must never reach user-visible fields; the
engine records these codes and a short message instead.
"""

from __future__ import annotations


class DatasourceError(Exception):
    code = "datasource_error"

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        self.message = message
        self.code = code or self.code
        super().__init__(message)


class AuthError(DatasourceError):
    code = "auth_failed"


class PermissionDenied(DatasourceError):
    code = "permission_denied"


class RateLimited(DatasourceError):
    code = "rate_limited"


class NotFoundError(DatasourceError):
    code = "not_found"


class ConflictError(DatasourceError):
    code = "conflict"


class ValidationError(DatasourceError):
    code = "validation_error"


class NetworkError(DatasourceError):
    code = "network_error"


class VendorError(DatasourceError):
    code = "vendor_error"


class UnsupportedType(DatasourceError):
    code = "unsupported_type"


class CredentialsUnreadable(DatasourceError):
    code = "credentials_unreadable"


class IngestFailed(DatasourceError):
    code = "ingest_failed"


class EnumerationIncomplete(DatasourceError):
    code = "enumeration_incomplete"
