#!/usr/bin/env python3
"""
Generate a dev-only CA + server certificate + key for Harmonia gRPC TLS smoke tests.

Uses `cryptography` (often already installed with Flower). Do **not** use these
artifacts in production; integrate your own PKI or ACME instead.

Usage:
  python -m harmonia_vision.scripts.generate_dev_tls --out-dir ./dev_tls
"""

from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("dev_tls"))
    p.add_argument(
        "--dns",
        action="append",
        default=["localhost"],
        help="DNS SAN (repeatable). Default: localhost",
    )
    p.add_argument(
        "--ip",
        action="append",
        default=["127.0.0.1"],
        help="IP SAN (repeatable). Default: 127.0.0.1",
    )
    args = p.parse_args()
    out = args.out_dir

    key_size = 2048
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    srv_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)

    ca_name = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Harmonia Dev CA"),
            x509.NameAttribute(NameOID.COMMON_NAME, "Harmonia Dev Root"),
        ]
    )
    now = dt.datetime.now(dt.timezone.utc)
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + dt.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    srv_name = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Harmonia Dev Server"),
            x509.NameAttribute(NameOID.COMMON_NAME, "harmonia-server"),
        ]
    )
    san: list[x509.GeneralName] = [x509.DNSName(d) for d in args.dns]
    san.extend(x509.IPAddress(ipaddress.ip_address(ip)) for ip in args.ip)

    srv_cert = (
        x509.CertificateBuilder()
        .subject_name(srv_name)
        .issuer_name(ca_cert.subject)
        .public_key(srv_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + dt.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    _write(
        out / "ca.pem",
        ca_cert.public_bytes(serialization.Encoding.PEM),
    )
    _write(
        out / "server.pem",
        srv_cert.public_bytes(serialization.Encoding.PEM),
    )
    _write(
        out / "server.key",
        srv_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ),
    )
    print(f"Wrote {out / 'ca.pem'}, {out / 'server.pem'}, {out / 'server.key'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
