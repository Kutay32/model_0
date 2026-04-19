"""Load PEM material for Flower gRPC TLS (TLS 1.2+ / 1.3 via gRPC + OpenSSL)."""

from __future__ import annotations

import os
from pathlib import Path


def read_pem_file(path: str) -> bytes:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"PEM file not found: {p}")
    return p.read_bytes()


def load_server_certificates_from_paths(
    ca_path: str,
    server_cert_path: str,
    server_key_path: str,
) -> tuple[bytes, bytes, bytes]:
    """
    Flower `start_server(certificates=...)` order:
    CA certificate, server certificate, server private key (PEM bytes).
    """
    return (
        read_pem_file(ca_path),
        read_pem_file(server_cert_path),
        read_pem_file(server_key_path),
    )


def resolve_server_tls_from_env() -> tuple[bytes, bytes, bytes] | None:
    """HARMONIA_TLS_CA_PATH, HARMONIA_TLS_SERVER_CERT, HARMONIA_TLS_SERVER_KEY."""
    ca = os.environ.get("HARMONIA_TLS_CA_PATH", "").strip()
    cert = os.environ.get("HARMONIA_TLS_SERVER_CERT", "").strip()
    key = os.environ.get("HARMONIA_TLS_SERVER_KEY", "").strip()
    if not ca and not cert and not key:
        return None
    if not (ca and cert and key):
        raise ValueError(
            "For TLS, set all of HARMONIA_TLS_CA_PATH, HARMONIA_TLS_SERVER_CERT, HARMONIA_TLS_SERVER_KEY "
            "(or omit all for plaintext gRPC)."
        )
    return load_server_certificates_from_paths(ca, cert, key)


