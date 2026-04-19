"""
Flower NumPy client: local U-Net training on `.npy` patches and optional CKKS uploads.
"""

from __future__ import annotations

import argparse
import base64
import os
import pickle
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import tenseal as ts
except ImportError:
    ts = None  # type: ignore[assignment]

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from .crypto_phe import encrypt_weight_vector
from .dataset_npy import default_loader
from .model import UNet, dice_iou_from_logits
from .model import HybridSegmentationLoss
from .tls_util import read_pem_file


def set_model_weights(model: torch.nn.Module, parameters: list[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    for key, arr in zip(keys, parameters):
        model.state_dict()[key].data.copy_(torch.tensor(arr, device=next(model.parameters()).device))


def get_model_weights(model: torch.nn.Module) -> list[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def _cfg_str(config: dict[str, Any], key: str, default: str) -> str:
    v = config.get(key, default)
    return str(v)


def _cfg_int(config: dict[str, Any], key: str, default: int) -> int:
    v = config.get(key, default)
    return int(v) if not isinstance(v, int) else v


def _cfg_float(config: dict[str, Any], key: str, default: float) -> float:
    v = config.get(key, default)
    if isinstance(v, float):
        return v
    return float(str(v))


class HarmoniaClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optim: torch.optim.Optimizer,
        train_loader: DataLoader,
        device: torch.device,
        use_phe: bool,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.train_loader = train_loader
        self.device = device
        self.use_phe = use_phe and ts is not None

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return get_model_weights(self.model)

    def fit(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        set_model_weights(self.model, parameters)

        local_epochs = _cfg_int(config, "local_epochs", 5)
        self.model.train()
        running = 0.0
        n = 0
        for _ in range(local_epochs):
            for xb, yb in self.train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                self.optim.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
                loss.backward()
                self.optim.step()
                running += float(loss.item()) * xb.size(0)
                n += xb.size(0)
        avg_loss = running / max(n, 1)

        if self.use_phe:
            pub = _cfg_str(config, "phe_public_b64", "")
            if not pub:
                raise RuntimeError("PHE enabled but server did not provide phe_public_b64 in fit config")
            scale = _cfg_float(config, "phe_scale", 1e5)
            ctx = ts.context_from(base64.b64decode(pub.encode("ascii")))
            # FedAvg numerator: encrypt (w_k * n_k) so the server can sum homomorphically.
            weights = [w.astype(np.float64) * float(n) for w in get_model_weights(self.model)]
            enc = encrypt_weight_vector(weights, scale=scale, ctx=ctx)
            blob = pickle.dumps(enc)
            arr = np.frombuffer(blob, dtype=np.uint8)
            metrics = {"train_loss": avg_loss, "comm_bytes": float(arr.nbytes)}
            return [arr], n, metrics

        metrics = {"train_loss": avg_loss, "comm_bytes": float(sum(w.nbytes for w in get_model_weights(self.model)))}
        return get_model_weights(self.model), n, metrics

    def evaluate(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple[float, int, dict[str, Any]]:
        set_model_weights(self.model, parameters)
        self.model.eval()
        loss_fn = HybridSegmentationLoss().to(self.device)
        tot_loss = 0.0
        tot_dice = 0.0
        tot_iou = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in self.train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                dices: list[float] = []
                ious: list[float] = []
                for i in range(xb.size(0)):
                    d, j = dice_iou_from_logits(logits[i : i + 1], yb[i : i + 1])
                    dices.append(d)
                    ious.append(j)
                bs = xb.size(0)
                tot_loss += float(loss.item()) * bs
                tot_dice += float(np.mean(dices)) * bs
                tot_iou += float(np.mean(ious)) * bs
                n += bs
        if n == 0:
            return 0.0, 0, {}
        return (
            tot_loss / n,
            n,
            {"dsc": tot_dice / n, "iou": tot_iou / n},
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default=os.environ.get("HARMONIA_SERVER", "127.0.0.1:8080"))
    p.add_argument("--data-root", default=os.environ.get("HARMONIA_DATA_ROOT", "/data/processed"))
    p.add_argument("--client-key", default=os.environ.get("HARMONIA_CLIENT_KEY", "client_a"), choices=("client_a", "client_b"))
    p.add_argument("--phe", action="store_true", help="Encrypt weight uploads with TenSEAL CKKS when server expects PHE.")
    p.add_argument(
        "--tls-ca",
        default=os.environ.get("HARMONIA_TLS_ROOT_CA", "").strip(),
        help="PEM path to CA that signed the server cert (enables TLS; matches server HARMONIA_TLS_CA_PATH file).",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, base_ch=32).to(device)
    loss_fn = HybridSegmentationLoss().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    ds = default_loader(args.data_root, args.client_key, train=True)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

    env_phe = os.environ.get("HARMONIA_PHE", "").lower() in ("1", "true", "yes")
    use_phe = (bool(args.phe) or env_phe) and ts is not None
    if (bool(args.phe) or env_phe) and ts is None:
        print("HARMONIA: `tenseal` not installed; disabling PHE uploads.", file=sys.stderr)

    client = HarmoniaClient(model, loss_fn, optim, loader, device, use_phe=use_phe)
    root_pem: bytes | None = None
    ca_path = (args.tls_ca or "").strip()
    if ca_path:
        try:
            root_pem = read_pem_file(ca_path)
        except OSError as e:
            print(f"HARMONIA TLS: {e}", file=sys.stderr)
            return 2

    kwargs: dict[str, Any] = {
        "server_address": args.server,
        "client": client,
        "grpc_max_message_length": 512 * 1024 * 1024,
    }
    if root_pem is not None:
        kwargs["root_certificates"] = root_pem
        # insecure defaults False when root_certificates is set (Flower compat client)

    if hasattr(fl.client, "start_numpy_client"):
        fl.client.start_numpy_client(**kwargs)
    else:
        kwargs["client"] = client.to_client()
        fl.client.start_client(**kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
