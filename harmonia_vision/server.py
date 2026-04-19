"""
Flower server: FedAvg on Harmonia Vision U-Net with optional CKKS (TenSEAL) uploads.

TenSEAL implements CKKS/BFV-style encryption; Paillier is not implemented in TenSEAL.
When PHE is disabled, aggregation is standard FedAvg over plaintext weight tensors.
"""

from __future__ import annotations

import argparse
import base64
import os
import pickle
import sys
from typing import Any

import numpy as np

try:
    import tenseal as ts
except ImportError:
    ts = None  # type: ignore[assignment]

import flwr as fl
from flwr.common import FitRes, Metrics, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from .crypto_phe import SerializedEncryptedRound, add_encrypted_rounds, decrypt_weight_vector, make_ckks_context
from .tls_util import load_server_certificates_from_paths, resolve_server_tls_from_env


def _weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}
    total = float(sum(n for n, _ in metrics))
    if total <= 0:
        return {}
    acc: dict[str, float] = {}
    for num, m in metrics:
        w = float(num) / total
        for k, v in m.items():
            if isinstance(v, (int, float)):
                acc[k] = acc.get(k, 0.0) + float(v) * w
    return acc


class HarmoniaFedAvg(FedAvg):
    """FedAvg with optional encrypted uploads (pickled CKKS payloads in a single uint8 tensor)."""

    def __init__(
        self,
        *args: Any,
        use_phe: bool = False,
        phe_scale: float = 1e5,
        secret_context: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_phe = use_phe
        self.phe_scale = phe_scale
        if secret_context is not None:
            self._secret_ctx = secret_context
        else:
            self._secret_ctx = make_ckks_context() if use_phe and ts is not None else None
        if self._secret_ctx is not None:
            _ = self._secret_ctx.serialize(save_secret_key=False)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Any],
    ) -> tuple[fl.common.Parameters | None, Metrics]:
        if not results:
            return None, {}
        if not self.use_phe or self._secret_ctx is None:
            aggregated = super().aggregate_fit(server_round, results, failures)
            if aggregated[0] is None:
                return aggregated
            params, met = aggregated
            comm = sum(
                sum(t.nbytes for t in parameters_to_ndarrays(r[1].parameters)) for r in results
            )
            met = dict(met or {})
            met["comm_bytes_total"] = float(comm)
            return params, met

        acc_enc: SerializedEncryptedRound | None = None
        metrics: list[tuple[int, Metrics]] = []
        comm = 0

        for _, fit in results:
            arr_list = parameters_to_ndarrays(fit.parameters)
            if not arr_list:
                continue
            blob = pickle.loads(np.frombuffer(arr_list[0], dtype=np.uint8))
            comm += int(arr_list[0].nbytes)
            if not isinstance(blob, SerializedEncryptedRound):
                raise TypeError("Expected SerializedEncryptedRound when PHE is enabled")
            acc_enc = blob if acc_enc is None else add_encrypted_rounds(acc_enc, blob, self._secret_ctx)
            metrics.append((fit.num_examples, dict(fit.metrics or {})))

        if acc_enc is None:
            return None, {}

        total_examples = float(sum(float(r[1].num_examples) for r in results))
        weights = decrypt_weight_vector(acc_enc, self._secret_ctx, total_examples=total_examples)
        met = _weighted_average(metrics)
        met["comm_bytes_total"] = float(comm)
        return ndarrays_to_parameters(weights), met


def build_on_fit_config(public_b64: str | None, scale: float):
    def on_fit_config_fn(_rnd: int) -> dict[str, str]:
        cfg: dict[str, str] = {"local_epochs": "5"}
        if public_b64:
            cfg["phe_public_b64"] = public_b64
            cfg["phe_scale"] = str(scale)
        return cfg

    return on_fit_config_fn


def _initial_ndarrays() -> list[np.ndarray]:
    from .model import UNet

    m = UNet(in_channels=1, base_ch=32)
    return [v.detach().cpu().numpy() for v in m.state_dict().values()]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("HARMONIA_SERVER_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("HARMONIA_SERVER_PORT", "8080")))
    p.add_argument("--rounds", type=int, default=int(os.environ.get("HARMONIA_ROUNDS", "5")))
    p.add_argument("--min-clients", type=int, default=int(os.environ.get("HARMONIA_MIN_CLIENTS", "2")))
    p.add_argument("--phe", action="store_true", help="Enable TenSEAL CKKS encrypted uploads (requires `tenseal`).")
    p.add_argument("--phe-scale", type=float, default=float(os.environ.get("HARMONIA_PHE_SCALE", "1e5")))
    p.add_argument("--tls-ca", default=os.environ.get("HARMONIA_TLS_CA_PATH", "").strip(), help="PEM path: CA (same as ssl-ca-certfile order for Flower).")
    p.add_argument("--tls-cert", default=os.environ.get("HARMONIA_TLS_SERVER_CERT", "").strip(), help="PEM path: server certificate.")
    p.add_argument("--tls-key", default=os.environ.get("HARMONIA_TLS_SERVER_KEY", "").strip(), help="PEM path: server private key.")
    args = p.parse_args()

    address = f"{args.host}:{args.port}"
    init = _initial_ndarrays()

    use_phe = bool(args.phe) or os.environ.get("HARMONIA_PHE", "").lower() in ("1", "true", "yes")
    if use_phe and ts is None:
        print("HARMONIA: `tenseal` not installed; disabling PHE and using plaintext FedAvg.", file=sys.stderr)
        use_phe = False

    phe_secret = make_ckks_context() if (use_phe and ts is not None) else None
    public_b64 = base64.b64encode(phe_secret.serialize(save_secret_key=False)).decode("ascii") if phe_secret else None

    strategy = HarmoniaFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_available_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        initial_parameters=ndarrays_to_parameters(init),
        on_fit_config_fn=build_on_fit_config(public_b64, args.phe_scale),
        on_evaluate_config_fn=lambda _rnd: {},
        fit_metrics_aggregation_fn=_weighted_average,
        evaluate_metrics_aggregation_fn=_weighted_average,
        use_phe=use_phe,
        phe_scale=args.phe_scale,
        secret_context=phe_secret,
    )

    certificates: tuple[bytes, bytes, bytes] | None = None
    if args.tls_ca and args.tls_cert and args.tls_key:
        certificates = load_server_certificates_from_paths(args.tls_ca, args.tls_cert, args.tls_key)
    elif args.tls_ca or args.tls_cert or args.tls_key:
        print("HARMONIA: provide all three of --tls-ca, --tls-cert, --tls-key (or none for plaintext).", file=sys.stderr)
        return 2
    else:
        try:
            certificates = resolve_server_tls_from_env()
        except ValueError as e:
            print(f"HARMONIA TLS: {e}", file=sys.stderr)
            return 2

    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        grpc_max_message_length=512 * 1024 * 1024,
        certificates=certificates,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
