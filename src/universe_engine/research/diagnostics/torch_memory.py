"""Supervised memory benchmark (PyTorch reference).

Sprint 14 Pass 1 course correction: establish that the learning machinery
itself works before attempting delayed-cue control.

Task: a binary cue is presented at timestep 0, then disappears. After a
configurable delay the model must predict the original cue.

Models compared:
  * Feedforward: sees only the final (post-delay) observation. Cannot recover
    a randomized earlier cue (chance ~50%). Negative control.
  * Frozen GRU: random recurrent weights, trained readout head. Tests whether a
    fixed random recurrent projection retains information.
  * Trained GRU: torch.nn.GRU trained end-to-end with autograd; recurrent
    weights receive gradients and should learn to preserve the cue.

This is a representation-learning test, not yet proof of a learned control
policy. It verifies that a recurrent network can be trained to retain
information across a delay.

No claim of consciousness; measures representational memory only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class CueMemoryDataset:
    """Generate sequences: binary cue at t=0, neutral input during delay,
    then a query. Target is the original cue."""

    def __init__(self, seed: int = 0, n_features: int = 1, neutral: float = 0.5):
        self.rng = np.random.default_rng(seed)
        self.n_features = int(n_features)
        self.neutral = float(neutral)

    def sample(self, delay: int) -> Tuple[np.ndarray, float]:
        """Return (sequence, cue). Sequence shape (delay+1, n_features)."""
        cue = float(self.rng.integers(0, 2))
        seq = np.full((delay + 1, self.n_features), self.neutral, dtype=np.float32)
        seq[0, 0] = cue
        return seq, cue

    def sample_batch(self, batch_size: int, delay: int):
        seqs = []; cues = []
        for _ in range(batch_size):
            seq, cue = self.sample(delay)
            seqs.append(seq); cues.append(cue)
        return (np.stack(seqs), np.array(cues, dtype=np.float32))


class FeedForwardCueModel(nn.Module):
    """Sees only the final observation (post-delay, neutral). Cannot know the cue."""

    def __init__(self, n_features: int = 1, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (batch, delay+1, n_features); use only the final timestep.
        final = seq[:, -1, :]
        return self.net(final).squeeze(-1)


class GRUCueModel(nn.Module):
    """Processes the full sequence with a GRU; reads out the cue from the
    final hidden state."""

    def __init__(self, n_features: int = 1, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(seq)  # h_n: (1, batch, hidden)
        h = h_n[-1]  # (batch, hidden)
        return self.readout(h).squeeze(-1)


@dataclass
class MemoryResult:
    model: str
    delay: int
    train_seed: int
    accuracy: float
    n_train: int
    n_test: int
    n_epochs: int
    recurrent_weight_changed: bool


def _accuracy(model: nn.Module, seqs: np.ndarray, cues: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(seqs))
        pred = (torch.sigmoid(logits) > 0.5).float().reshape(-1, 1).numpy()
    return float((pred == cues.reshape(-1, 1)).mean())


def _recurrent_params_changed(model: nn.Module, before: dict) -> bool:
    for name, p in model.named_parameters():
        if "gru" in name.lower() and p.data is not None:
            if not torch.allclose(p.data, before[name], atol=1e-6):
                return True
    return False


def train_supervised(
    delay: int,
    train_seed: int = 0,
    n_train: int = 2000,
    n_test: int = 1000,
    n_epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden: int = 32,
    grad_clip: float = 1.0,
    device: str = "cpu",
) -> List[dict]:
    """Train and evaluate feedforward, frozen-GRU, and trained GRU models.

    Returns per-model accuracies plus diagnostics (gradient nonzero, parameter
    change) for the trained GRU's recurrent weights.
    """
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    dataset = CueMemoryDataset(seed=train_seed + 1)

    train_seqs, train_cues = dataset.sample_batch(n_train, delay)
    test_seqs, test_cues = dataset.sample_batch(n_test, delay)

    train_seqs_t = torch.from_numpy(train_seqs)
    train_cues_t = torch.from_numpy(train_cues).reshape(-1)

    records: List[dict] = []

    def _record(model: str, accuracy: float, **extra) -> dict:
        r = {
            "model": model,
            "delay": delay,
            "train_seed": train_seed,
            "accuracy": float(accuracy),
            "n_train": n_train,
            "n_test": n_test,
            "n_epochs": n_epochs,
        }
        r.update(extra)
        return r

    # --- Feedforward baseline (final observation only) ---
    ff = FeedForwardCueModel(n_features=dataset.n_features, hidden=hidden).to(device)
    opt = optim.Adam(ff.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    n_batches = max(1, n_train // batch_size)
    for _ in range(n_epochs):
        perm = torch.randperm(n_train)
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            xb = train_seqs_t[idx].to(device)
            yb = train_cues_t[idx].to(device)
            opt.zero_grad()
            logits = ff(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(ff.parameters(), grad_clip)
            opt.step()
    records.append(_record("feedforward", _accuracy(ff, test_seqs, test_cues)))

    # --- Frozen GRU: random recurrent weights, trained readout head ---
    frz = GRUCueModel(n_features=dataset.n_features, hidden=hidden).to(device)
    for p in frz.gru.parameters():
        p.requires_grad = False
    opt = optim.Adam([p for p in frz.parameters() if p.requires_grad], lr=lr)
    for _ in range(n_epochs):
        perm = torch.randperm(n_train)
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            xb = train_seqs_t[idx].to(device)
            yb = train_cues_t[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(frz(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(frz.parameters(), grad_clip)
            opt.step()
    records.append(_record("frozen_gru", _accuracy(frz, test_seqs, test_cues)))

    # --- Trained GRU: end-to-end with autograd ---
    gru = GRUCueModel(n_features=dataset.n_features, hidden=hidden).to(device)
    gru_before = {n: p.data.clone() for n, p in gru.named_parameters() if "gru" in n.lower()}
    opt = optim.Adam(gru.parameters(), lr=lr)
    grad_nonzero = False
    for _ in range(n_epochs):
        perm = torch.randperm(n_train)
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            xb = train_seqs_t[idx].to(device)
            yb = train_cues_t[idx].to(device)
            opt.zero_grad()
            logits = gru(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            # Verify recurrent weights receive gradients.
            for p in gru.gru.parameters():
                if p.grad is not None and p.grad.abs().max().item() > 0:
                    grad_nonzero = True
            nn.utils.clip_grad_norm_(gru.parameters(), grad_clip)
            opt.step()
    w_changed = _recurrent_params_changed(gru, gru_before)
    records.append(_record("trained_gru", _accuracy(gru, test_seqs, test_cues),
                           recurrent_grad_nonzero=bool(grad_nonzero),
                           recurrent_weight_changed=bool(w_changed)))
    return records


def run_delays(delays: List[int] | None = None,
               n_seeds: int = 3,
               **kwargs) -> List[dict]:
    """Run the supervised memory benchmark across delays and seeds."""
    if delays is None:
        delays = [1, 3, 5, 10, 20]
    out: List[dict] = []
    for delay in delays:
        for seed in range(n_seeds):
            out.extend(train_supervised(delay=delay, train_seed=seed, **kwargs))
    return out


if __name__ == "__main__":
    results = run_delays()
    print(f"{'delay':>5} {'model':>14} {'accuracy':>9} {'grad_nz':>7} {'w_changed':>10}")
    for r in results:
        print(f"{r['delay']:>5} {r['model']:>14} {r['accuracy']:>9.3f} "
              f"{str(r.get('recurrent_grad_nonzero','')):>7} "
              f"{str(r.get('recurrent_weight_changed','')):>10}")
