"""Held-out MINE-style DV and InfoNCE experiments; models export as safe arrays.

Population variational bounds do not certify finite-sample estimates. Selection
uses validation data; true Gaussian MI is reserved for final evaluation.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .estimators import gaussian_pairs, gaussian_truth


class Critic(nn.Module):
    def __init__(self, dimension, kind, width=128):
        super().__init__()
        self.kind = kind
        if kind == "mine_dv":
            self.network = nn.Sequential(
                nn.Linear(2 * dimension, width),
                nn.ELU(),
                nn.Linear(width, width),
                nn.ELU(),
                nn.Linear(width, 1),
            )
        elif kind == "infonce":
            self.x_network = nn.Sequential(
                nn.Linear(dimension, width), nn.ELU(), nn.Linear(width, 64)
            )
            self.y_network = nn.Sequential(
                nn.Linear(dimension, width), nn.ELU(), nn.Linear(width, 64)
            )
        else:
            raise ValueError("Unknown critic objective")

    def pair_score(self, x, y):
        return 15 * torch.tanh(self.network(torch.cat([x, y], dim=1)).squeeze(1) / 15)

    def score_matrix(self, x, y):
        return self.x_network(x) @ self.y_network(y).T / 8


def dv_value(joint_score, product_score):
    return joint_score.mean() - torch.logsumexp(product_score, dim=0) + np.log(len(product_score))


def infonce_value(scores):
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or len(scores) < 2:
        raise ValueError("At least two equally sized positive/negative batches required")
    return np.log(len(scores)) + (scores.diag() - torch.logsumexp(scores, dim=1)).mean()


@torch.no_grad()
def evaluate(critic, x, y, batch_size):
    critic.eval()
    if critic.kind == "mine_dv":
        half = len(x) // 2
        if half < 1:
            raise ValueError("Independent product evaluation needs two observations")
        positive = critic.pair_score(x, y)
        # Disjoint observations ensure each product pair is from p(x)p(y).
        negative = critic.pair_score(x[:half], y[half : 2 * half])
        value = float(dv_value(positive, negative).cpu()) / np.log(2)
        return {
            "estimate_bits": value,
            "joint_scores": positive.cpu().numpy(),
            "product_scores": negative.cpu().numpy(),
        }
    values = []
    for start in range(0, len(x) - batch_size + 1, batch_size):
        score = critic.score_matrix(x[start : start + batch_size], y[start : start + batch_size])
        values.append(float(infonce_value(score).cpu()) / np.log(2))
    if not values:
        raise ValueError("Evaluation sample count must cover a complete contrastive batch")
    return {
        "estimate_bits": float(np.mean(values)),
        "batch_estimates_bits": np.array(values),
        "batch_ceiling_bits": float(np.log2(batch_size)),
        "batch_standard_error": float(np.std(values, ddof=1) / np.sqrt(len(values)))
        if len(values) > 1
        else None,
    }


def export(path, critic, metadata):
    state = {name: tensor.detach().cpu().numpy() for name, tensor in critic.state_dict().items()}
    state["__metadata__"] = np.array(json.dumps(metadata))
    np.savez_compressed(path, **state)


def load(path, device="cpu"):
    with np.load(path, allow_pickle=False) as values:
        metadata = json.loads(str(values["__metadata__"]))
        critic = Critic(metadata["dimension"], metadata["kind"], metadata["width"])
        state = {
            key: torch.from_numpy(values[key].copy())
            for key in values.files
            if key != "__metadata__"
        }
    critic.load_state_dict(state, strict=True)
    return critic.to(device).eval(), metadata


def train(settings, seed, model_path, device="cpu"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dimension, correlation = settings["dimension"], settings["correlation"]
    kind = settings["method"]
    batch, steps = settings["batch_size"], settings["steps"]
    if min(batch, steps) < 2:
        raise ValueError("Nontrivial batch and training budget required")
    raw = {}
    tensors = {}
    for split, n in [
        ("train", settings["train_size"]),
        ("validation", settings["validation_size"]),
        ("test", settings["test_size"]),
    ]:
        offset = {"train": 100000, "validation": 200000, "test": 300000}[split]
        x, y = gaussian_pairs(n, dimension, correlation, seed + offset)
        raw[split] = (x, y)
        tensors[split] = (
            torch.tensor(x, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device),
        )
    if min(settings["train_size"], settings["validation_size"], settings["test_size"]) < batch:
        raise ValueError("Every data split must support the selected batch")
    critic = Critic(dimension, kind, settings.get("width", 128)).to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=settings["learning_rate"])
    rng = np.random.default_rng(seed + 400000)
    ema = torch.tensor(1.0, device=device)
    history = []
    best, selected_step, selected_state = -np.inf, 0, None
    started = time.perf_counter()
    for step in range(1, steps + 1):
        critic.train()
        index = rng.choice(settings["train_size"], batch, replace=False)
        x, y = (t[index] for t in tensors["train"])
        if kind == "mine_dv":
            permutation = np.roll(np.arange(batch), 1)
            positive, negative = critic.pair_score(x, y), critic.pair_score(x, y[permutation])
            mean_exp = negative.exp().mean()
            ema = 0.99 * ema + 0.01 * mean_exp.detach()
            loss = -positive.mean() + mean_exp / ema.detach()
            displayed_objective = float(dv_value(positive, negative).detach().cpu()) / np.log(2)
        else:
            value = infonce_value(critic.score_matrix(x, y))
            loss = -value
            displayed_objective = float(value.detach().cpu()) / np.log(2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
        optimizer.step()
        if step % settings.get("validate_every", 100) == 0 or step == steps:
            validation = evaluate(critic, *tensors["validation"], batch)["estimate_bits"]
            history.append([step, displayed_objective, validation])
            if validation > best:
                best, selected_step = validation, step
                selected_state = {
                    k: v.detach().cpu().clone() for k, v in critic.state_dict().items()
                }
    critic.load_state_dict(selected_state)
    test = evaluate(critic, *tensors["test"], batch)
    metadata = {
        "dimension": dimension,
        "kind": kind,
        "width": settings.get("width", 128),
        "selected_step": selected_step,
        "seed": seed,
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    export(model_path, critic, metadata)
    # Test inference from the portable model, independently of the training object.
    restored, _ = load(model_path, device)
    restored_test = evaluate(restored, *tensors["test"], batch)["estimate_bits"]
    if not np.isclose(restored_test, test["estimate_bits"], rtol=0, atol=1e-7):
        raise RuntimeError("Portable critic inference mismatch")
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    metrics = {
        "estimate_bits": test["estimate_bits"],
        "true_bits": gaussian_truth(dimension, correlation),
        "validation_bits": float(best),
        "selected_step": selected_step,
        "fit_seconds": seconds,
        "device": str(device),
        "batch_ceiling_bits": test.get("batch_ceiling_bits"),
        "scope": "MINE-style bounded DV critic with EMA gradient normalization, or separable InfoNCE critic. Independent train/validation/test draws; validation-only checkpoint selection. Finite empirical estimates are not certified bounds and are not clipped to the known truth.",
    }
    arrays = {"history": np.array(history), "test_x": raw["test"][0], "test_y": raw["test"][1]}
    arrays.update({key: value for key, value in test.items() if isinstance(value, np.ndarray)})
    return metrics, arrays
