"""
Benchmark Harmonia Vision training modes on local `.npy` data:

- **centralized**: train on the union of Client A + Client B folders.
- **fl_iid** / **fl_non_iid**: simulate two-client FedAvg (same protocol as Docker FL) in one process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm.auto import tqdm

from .dataset_npy import MammogramNpyDataset
from .model import HybridSegmentationLoss, UNet, dice_iou_from_logits


def _select_device(kind: str) -> torch.device:
    """
    kind: 'auto' | 'cuda' | 'cpu'
    - auto: CUDA varsa GPU, yoksa CPU (+ uyarı)
    - cuda: zorunlu GPU; yoksa RuntimeError
    - cpu: zorunlu CPU
    """
    k = kind.lower().strip()
    if k == "cpu":
        return torch.device("cpu")
    if k == "cuda":
        if not torch.cuda.is_available():
            tc = getattr(torch.version, "cuda", None)
            raise RuntimeError(
                f"--device cuda istendi ama torch.cuda kullanılamıyor. "
                f"torch={torch.__version__} torch.version.cuda={tc!r}. "
                "NVIDIA GPU sürücüsü + PyTorch'un CUDA'lı wheel'i gerekir: https://pytorch.org "
                "(ör. pip: torch torchvision --index-url https://download.pytorch.org/whl/cu124)"
            )
        return torch.device("cuda")
    if k != "auto":
        raise ValueError(f"Unknown --device {kind!r} (use auto, cuda, cpu)")
    if torch.cuda.is_available():
        return torch.device("cuda")
    tc = getattr(torch.version, "cuda", None)
    print(
        "[benchmark] UYARI: CUDA yok — CPU ile eğitim yavaşlar. "
        f"torch={torch.__version__} torch.version.cuda={tc!r}. "
        "Çözüm: GPU sürücüsü güncel olsun; pip ile CPU-only torch yerine CUDA build kurun (pytorch.org).",
        flush=True,
    )
    return torch.device("cpu")


def _patient_id_from_stem(stem: str) -> str:
    """CBIS-style stem e.g. P_00001_LEFT_CC -> patient_id P_00001."""
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "P":
        return f"{parts[0]}_{parts[1]}"
    return stem


def train_val_indices_by_patient(
    ds: ConcatDataset,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], dict[str, object]]:
    """
    Split global indices by patient (no patient appears in both train and val).
    Returns (train_indices, val_indices, stats).
    """
    stems_and_idx: list[tuple[str, int]] = []
    offset = 0
    for sub in ds.datasets:
        for i in range(len(sub)):
            stem = sub.samples[i][0].stem
            stems_and_idx.append((_patient_id_from_stem(stem), offset + i))
        offset += len(sub)

    patients = sorted({p for p, _ in stems_and_idx})
    stats: dict[str, object] = {"n_patients": len(patients), "n_samples": len(stems_and_idx)}
    if len(patients) < 2 or val_fraction <= 0:
        train_idx = [i for _, i in stems_and_idx]
        return train_idx, [], stats

    rng = np.random.default_rng(seed)
    shuffled = list(patients)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    n_val = min(n_val, len(shuffled) - 1)  # keep at least one patient for train
    val_patients = set(shuffled[:n_val])
    train_patients = set(shuffled[n_val:])
    train_idx = [i for p, i in stems_and_idx if p in train_patients]
    val_idx = [i for p, i in stems_and_idx if p in val_patients]
    stats["n_train_patients"] = len(train_patients)
    stats["n_val_patients"] = len(val_patients)
    stats["n_train_samples"] = len(train_idx)
    stats["n_val_samples"] = len(val_idx)
    return train_idx, val_idx, stats


def _get_weights(m: torch.nn.Module) -> list[np.ndarray]:
    return [v.detach().cpu().numpy().copy() for v in m.state_dict().values()]


def _set_weights(m: torch.nn.Module, weights: list[np.ndarray]) -> None:
    keys = list(m.state_dict().keys())
    for k, arr in zip(keys, weights):
        m.state_dict()[k].data.copy_(torch.tensor(arr, device=next(m.parameters()).device))


def _train_local(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optim: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    *,
    show_progress: bool = True,
    desc: str = "train",
) -> float:
    model.train()
    running = 0.0
    n = 0
    n_batches = len(loader)
    if n_batches == 0 or epochs == 0:
        return 0.0
    total_steps = epochs * n_batches
    pbar: tqdm | None = None
    if show_progress:
        pbar = tqdm(total=total_steps, desc=desc, unit="step", dynamic_ncols=True, mininterval=0.3)
    try:
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optim.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optim.step()
                running += float(loss.item()) * xb.size(0)
                n += xb.size(0)
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{float(loss.item()):.4f}", avg=f"{running / max(n, 1):.4f}")
    finally:
        if pbar is not None:
            pbar.close()
    return running / max(n, 1)


@torch.no_grad()
def _eval_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    show_progress: bool = True,
    desc: str = "eval",
) -> dict[str, float]:
    loss_fn = HybridSegmentationLoss().to(device)
    model.eval()
    tot_loss = 0.0
    tot_dice = 0.0
    tot_iou = 0.0
    n = 0
    it = loader
    if show_progress and len(loader) > 0:
        it = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True, mininterval=0.3)
    for xb, yb in it:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        bs = xb.size(0)
        dices = []
        ious = []
        for i in range(bs):
            d, j = dice_iou_from_logits(logits[i : i + 1], yb[i : i + 1])
            dices.append(d)
            ious.append(j)
        tot_loss += float(loss.item()) * bs
        tot_dice += float(np.mean(dices)) * bs
        tot_iou += float(np.mean(ious)) * bs
        n += bs
        if show_progress and hasattr(it, "set_postfix"):
            it.set_postfix(
                loss=f"{float(loss.item()):.4f}",
                dsc=f"{tot_dice / max(n, 1):.3f}",
                iou=f"{tot_iou / max(n, 1):.3f}",
            )
    if n == 0:
        return {"loss": 0.0, "dsc": 0.0, "iou": 0.0}
    return {"loss": tot_loss / n, "dsc": tot_dice / n, "iou": tot_iou / n}


def estimate_comm_mb_per_round(num_params_elements: int, n_clients: int, rounds: int) -> float:
    """Rough gRPC payload estimate: upload + download of all parameters each round per client."""
    bytes_per = num_params_elements * 4
    per_round = n_clients * bytes_per * 2
    return per_round * rounds / (1024.0**2)


PRESETS: dict[str, tuple[int, int]] = {
    "quick": (2, 2),
    "standard": (5, 5),
    "long": (20, 10),
}


def resolve_rounds_epochs(preset: str, rounds: int | None, local_epochs: int | None) -> tuple[int, int]:
    pr, pe = PRESETS[preset]
    return (rounds if rounds is not None else pr, local_epochs if local_epochs is not None else pe)


def save_training_artifact(
    model: torch.nn.Module,
    checkpoint_path: str | None,
    metrics_path: str | None,
    meta: dict[str, object],
) -> tuple[str | None, str | None]:
    """Writes `checkpoint_path` (.pth with state_dict + meta) and optional JSON metrics sidecar."""
    ckpt_out: str | None = None
    json_out: str | None = None
    if checkpoint_path:
        p = Path(checkpoint_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": model.state_dict(),
            "meta": {
                **meta,
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        }
        torch.save(payload, str(p))
        ckpt_out = str(p.resolve())

    if metrics_path:
        mp = Path(metrics_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        with mp.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        json_out = str(mp.resolve())
    elif checkpoint_path:
        mp = Path(checkpoint_path).with_suffix(".metrics.json")
        mp.parent.mkdir(parents=True, exist_ok=True)
        with mp.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        json_out = str(mp.resolve())

    return ckpt_out, json_out


def run_centralized(
    data_root: str,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    *,
    checkpoint_path: str | None = None,
    metrics_path: str | None = None,
    extra_meta: dict[str, object] | None = None,
    val_patient_fraction: float = 0.15,
    split_seed: int = 42,
    use_val_split: bool = True,
    device_kind: str = "auto",
    show_progress: bool = True,
) -> dict[str, float | str | None]:
    device = _select_device(device_kind)
    ds_a = MammogramNpyDataset(os.path.join(data_root, "client_a", "images"), os.path.join(data_root, "client_a", "masks"))
    ds_b = MammogramNpyDataset(os.path.join(data_root, "client_b", "images"), os.path.join(data_root, "client_b", "masks"))
    ds = ConcatDataset([ds_a, ds_b])

    vf = val_patient_fraction if use_val_split else 0.0
    train_idx, val_idx, split_stats = train_val_indices_by_patient(ds, vf, split_seed)

    train_loader = DataLoader(
        Subset(ds, train_idx) if val_idx else ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    if val_idx:
        eval_loader = DataLoader(
            Subset(ds, val_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        eval_split = "patient_holdout"
    else:
        eval_loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        eval_split = "full_overlap_no_holdout"

    model = UNet(in_channels=1, base_ch=32).to(device)
    loss_fn = HybridSegmentationLoss().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    n_train = len(train_loader.dataset)
    n_eval = len(eval_loader.dataset)
    print(
        f"[benchmark] python={sys.executable} | ver={sys.version.split()[0]} | cwd={os.getcwd()}",
        flush=True,
    )
    print(
        f"[benchmark] centralized start: device={device} rounds={rounds} local_epochs={local_epochs} "
        f"train_samples={n_train} eval_samples={n_eval} eval_split={eval_split}",
        flush=True,
    )

    t0 = time.perf_counter()
    last_loss = 0.0
    for r in range(rounds):
        last_loss = _train_local(
            model,
            loss_fn,
            optim,
            train_loader,
            device,
            local_epochs,
            show_progress=show_progress,
            desc=f"train r{r + 1}/{rounds}",
        )
        print(
            f"[benchmark] round {r + 1}/{rounds} train_loss={last_loss:.6f}",
            flush=True,
        )
    train_time = time.perf_counter() - t0

    print("[benchmark] running eval …", flush=True)
    metrics = _eval_model(model, eval_loader, device, show_progress=show_progress, desc="eval val")
    n_params = sum(p.numel() for p in model.parameters())
    comm_mb = estimate_comm_mb_per_round(n_params, n_clients=1, rounds=rounds)  # single process baseline

    meta: dict[str, object] = {
        "mode": "centralized",
        "data_root": os.path.abspath(data_root),
        "rounds": rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "train_loss": float(last_loss),
        "dsc": metrics["dsc"],
        "iou": metrics["iou"],
        "eval_loss": float(metrics["loss"]),
        "eval_split": eval_split,
        "split_seed": int(split_seed),
        "val_patient_fraction": float(vf) if val_idx else None,
        "split_stats": split_stats,
        "comm_mb_est": float(comm_mb),
        "train_time_s": float(train_time),
        "n_parameters": int(n_params),
        "device": str(device),
    }
    if extra_meta:
        meta.update(extra_meta)

    ck, js = save_training_artifact(model, checkpoint_path, metrics_path, meta)
    meta["checkpoint_path"] = ck
    meta["metrics_json_path"] = js

    return {
        "train_loss": float(last_loss),
        "dsc": metrics["dsc"],
        "iou": metrics["iou"],
        "eval_loss": float(metrics["loss"]),
        "comm_mb": float(comm_mb),
        "train_time_s": float(train_time),
        "checkpoint_path": ck,
        "metrics_json_path": js,
    }


def run_federated_simulation(
    data_root: str,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    *,
    checkpoint_path: str | None = None,
    metrics_path: str | None = None,
    extra_meta: dict[str, object] | None = None,
    val_patient_fraction: float = 0.15,
    split_seed: int = 42,
    use_val_split: bool = True,
    device_kind: str = "auto",
    show_progress: bool = True,
) -> dict[str, float | str | None]:
    device = _select_device(device_kind)
    ds_a = MammogramNpyDataset(
        os.path.join(data_root, "client_a", "images"),
        os.path.join(data_root, "client_a", "masks"),
        train=True,
    )
    ds_b = MammogramNpyDataset(
        os.path.join(data_root, "client_b", "images"),
        os.path.join(data_root, "client_b", "masks"),
        train=True,
    )
    ds = ConcatDataset([ds_a, ds_b])
    len_a = len(ds_a)

    vf = val_patient_fraction if use_val_split else 0.0
    train_idx, val_idx, split_stats = train_val_indices_by_patient(ds, vf, split_seed)

    idx_a = [i for i in train_idx if i < len_a]
    idx_b = [i - len_a for i in train_idx if i >= len_a]

    loaders = [
        DataLoader(
            Subset(ds_a, idx_a) if val_idx else ds_a,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        ),
        DataLoader(
            Subset(ds_b, idx_b) if val_idx else ds_b,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        ),
    ]

    global_model = UNet(in_channels=1, base_ch=32).to(device)
    global_w = _get_weights(global_model)

    print(
        f"[benchmark] federated_sim start: device={device} rounds={rounds} local_epochs={local_epochs} "
        f"split_stats={split_stats}",
        flush=True,
    )

    t0 = time.perf_counter()
    for _r in range(rounds):
        accum = [np.zeros_like(w) for w in global_w]
        denom = 0
        for ci, loader in enumerate(loaders):
            if len(loader.dataset) == 0:
                continue
            local = UNet(in_channels=1, base_ch=32).to(device)
            _set_weights(local, global_w)
            loss_fn = HybridSegmentationLoss().to(device)
            optim = torch.optim.Adam(local.parameters(), lr=1e-4)
            tag = "A" if ci == 0 else "B"
            _train_local(
                local,
                loss_fn,
                optim,
                loader,
                device,
                local_epochs,
                show_progress=show_progress,
                desc=f"fed r{_r + 1}/{rounds} client-{tag}",
            )
            w = _get_weights(local)
            n_k = len(loader.dataset)
            denom += n_k
            for i in range(len(accum)):
                accum[i] = accum[i] + w[i].astype(np.float64) * float(n_k)
        global_w = [(a / max(float(denom), 1e-8)).astype(np.float32) for a in accum]
        _set_weights(global_model, global_w)
        print(f"[benchmark] federated round {_r + 1}/{rounds} done", flush=True)
    train_time = time.perf_counter() - t0

    ds_a_eval = MammogramNpyDataset(
        os.path.join(data_root, "client_a", "images"),
        os.path.join(data_root, "client_a", "masks"),
        train=False,
    )
    ds_b_eval = MammogramNpyDataset(
        os.path.join(data_root, "client_b", "images"),
        os.path.join(data_root, "client_b", "masks"),
        train=False,
    )
    ds_eval = ConcatDataset([ds_a_eval, ds_b_eval])
    if val_idx:
        eval_loader = DataLoader(
            Subset(ds_eval, val_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        eval_split = "patient_holdout"
    else:
        eval_loader = DataLoader(
            ds_eval,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        eval_split = "full_overlap_no_holdout"

    print("[benchmark] running eval …", flush=True)
    metrics = _eval_model(global_model, eval_loader, device, show_progress=show_progress, desc="eval val (fed)")
    n_params = sum(p.numel() for p in global_model.parameters())
    comm_mb = estimate_comm_mb_per_round(n_params, n_clients=2, rounds=rounds)

    meta: dict[str, object] = {
        "mode": "federated_simulation",
        "data_root": os.path.abspath(data_root),
        "rounds": rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "dsc": metrics["dsc"],
        "iou": metrics["iou"],
        "eval_loss": float(metrics["loss"]),
        "eval_split": eval_split,
        "split_seed": int(split_seed),
        "val_patient_fraction": float(vf) if val_idx else None,
        "split_stats": split_stats,
        "comm_mb_est": float(comm_mb),
        "train_time_s": float(train_time),
        "n_parameters": int(n_params),
        "device": str(device),
    }
    if extra_meta:
        meta.update(extra_meta)

    ck, js = save_training_artifact(global_model, checkpoint_path, metrics_path, meta)
    meta["checkpoint_path"] = ck
    meta["metrics_json_path"] = js

    return {
        "train_loss": None,
        "dsc": metrics["dsc"],
        "iou": metrics["iou"],
        "eval_loss": float(metrics["loss"]),
        "comm_mb": float(comm_mb),
        "train_time_s": float(train_time),
        "checkpoint_path": ck,
        "metrics_json_path": js,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Train/evaluate Harmonia U-Net on processed client_a/client_b .npy data.",
    )
    p.add_argument("--mode", choices=("centralized", "fl_iid", "fl_non_iid"), required=True)
    p.add_argument("--data-root", required=True, help="Processed root containing client_a/ and client_b/")
    p.add_argument(
        "--preset",
        choices=tuple(PRESETS.keys()),
        default="standard",
        help="Default rounds/epochs unless overridden: quick=2×2, standard=5×5, long=20×10.",
    )
    p.add_argument("--rounds", type=int, default=None, help="Outer training loops (overrides preset).")
    p.add_argument("--local-epochs", type=int, default=None, help="Epochs per loop / per client round (overrides preset).")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Save model weights + metadata to this .pth path (e.g. harmonia_checkpoints/centralized.pth).",
    )
    p.add_argument(
        "--metrics-json",
        default=None,
        metavar="PATH",
        help="Write metrics JSON here; default: same path as checkpoint with .metrics.json suffix.",
    )
    p.add_argument(
        "--val-patient-fraction",
        type=float,
        default=0.15,
        help="Fraction of patients held out for eval (patient-level split). Ignored with --no-val-split.",
    )
    p.add_argument("--split-seed", type=int, default=42, help="RNG seed for patient hold-out split.")
    p.add_argument(
        "--no-val-split",
        action="store_true",
        help="Train and report metrics on the full dataset (eval overlaps training; for debugging only).",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto: use GPU if PyTorch sees CUDA; cuda: fail if no GPU; cpu: force CPU.",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (cleaner logs when redirecting to a file).",
    )
    args = p.parse_args()

    rounds, local_epochs = resolve_rounds_epochs(args.preset, args.rounds, args.local_epochs)
    extra = {"preset": args.preset, "cli_mode": args.mode, "device_requested": args.device}
    split_kw = {
        "val_patient_fraction": args.val_patient_fraction,
        "split_seed": args.split_seed,
        "use_val_split": not args.no_val_split,
        "device_kind": args.device,
        "show_progress": not args.no_progress,
    }

    if args.mode == "centralized":
        out = run_centralized(
            args.data_root,
            rounds,
            local_epochs,
            args.batch_size,
            checkpoint_path=args.checkpoint,
            metrics_path=args.metrics_json,
            extra_meta=extra,
            **split_kw,
        )
    else:
        extra["note"] = "IID vs non-IID depends on preprocessing; simulator is identical for fl_iid and fl_non_iid."
        out = run_federated_simulation(
            args.data_root,
            rounds,
            local_epochs,
            args.batch_size,
            checkpoint_path=args.checkpoint,
            metrics_path=args.metrics_json,
            extra_meta=extra,
            **split_kw,
        )

    row: dict[str, object] = {
        "mode": args.mode,
        "preset": args.preset,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "dsc": out["dsc"],
        "iou": out["iou"],
        "eval_loss": out["eval_loss"],
        "comm_mb_est": out["comm_mb"],
        "train_time_s": out["train_time_s"],
        "checkpoint_path": out.get("checkpoint_path"),
        "metrics_json_path": out.get("metrics_json_path"),
    }
    if out.get("train_loss") is not None:
        row["train_loss"] = out["train_loss"]
    print(row, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
