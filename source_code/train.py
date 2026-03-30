import math
#import os
import time
import copy
from dataclasses import dataclass # asdict
from typing import List, Optional, Tuple # Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Initialization for Token and Position vectors
# ----------------------------

def make_random_positions(L: int, seed: Optional[int] = None) -> torch.Tensor:
    if seed is not None:
        g = torch.Generator().manual_seed(int(seed))
        return torch.randn(L, 2, dtype=torch.get_default_dtype(), generator=g)
    return torch.randn(L, 2, dtype=torch.get_default_dtype())


# ----------------------------
# Model
# ----------------------------

class PosAttn_TokenOut(nn.Module):
    """
    Minimal version for this notebook:
    - token embeddings E in R^{V x dE}
    - positional vectors Ppos in R^{(L-1) x 2}, always initialized randomly in R^2
    """
    def __init__(
        self,
        V: int,
        L: int,
        dE: int = 3,
        use_batch_invariant_alpha: bool = False,
        use_identity_Mp=False,
        pos_seed: Optional[int] = None,
    ):
        super().__init__()
        self.V = int(V)
        self.L = int(L)
        self.L_pos = int(L - 1)
        self.dE = int(dE)
        self.use_batch_invariant_alpha = bool(use_batch_invariant_alpha)

        self.E = nn.Embedding(self.V, self.dE) ## initialize E tokens

        # Positions: always random R^2 (no pos_mode options)
        p = make_random_positions(self.L_pos, seed=pos_seed)
        self.Ppos = nn.Parameter(p)

        # parameters (keep shapes as in your safe_backup.py)
        self.M = nn.Parameter(0.001 * torch.randn(5, 5, dtype=torch.get_default_dtype()))
        self.use_identity_Mp = use_identity_Mp
        self.k_pos = nn.Parameter(torch.tensor(0.0, dtype=torch.get_default_dtype()))

        self.raw_beta_attn = nn.Parameter(torch.tensor(1.5, dtype=torch.get_default_dtype()))
        self.raw_beta_class = nn.Parameter(torch.tensor(0.0, dtype=torch.get_default_dtype()))

        self.W = nn.Parameter(0.01 * torch.randn(5, 5, dtype=torch.get_default_dtype()))

        mask = torch.tril(torch.ones(self.L_pos, self.L_pos, dtype=torch.bool), diagonal=-1)
        self.register_buffer("causal_mask", mask)

    # --- Convenience properties ---

    @property
    def beta_attn(self) -> torch.Tensor:
        return F.softplus(self.raw_beta_attn) + 1e-6

    @property
    def beta_class(self) -> torch.Tensor:
        return F.softplus(self.raw_beta_class) + 1e-6

    @property
    def WE(self) -> torch.Tensor:
        return self.W[: self.dE, : self.dE]

    @property
    def Mp(self) -> torch.Tensor:
        if not self.use_identity_Mp:
            return self.M[-2:, -2:]
        I = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=self.M.dtype, device=self.M.device)
        return I  # keep current behavior

    def _parts(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = x.shape
        Epart = self.E(x)
        Ppart = self.Ppos[:L].unsqueeze(0).expand(B, L, 2)
        return Epart, Ppart

    def attn_weights(self, s: torch.Tensor) -> torch.Tensor:
        B, L, D = s.shape
        causal = self.causal_mask

        if self.use_batch_invariant_alpha:
            p = self.Ppos[:L]
            scores = (p @ self.Mp) @ p.t()
            scores = self.beta_attn * scores
            scores = scores.masked_fill(~causal[:L, :L], float("-inf"))
            alpha = torch.softmax(scores, dim=-1)
            alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
            alpha[0, :] = 0.0
            return alpha.unsqueeze(0).expand(B, L, L)

        M_block = torch.zeros(5, 5, dtype=s.dtype, device=s.device)
        M_block[-2:, -2:] = self.Mp
        sM = torch.einsum("bld,dk->blk", s, M_block)
        scores = torch.einsum("bld,bmd->blm", sM, s)
        scores = self.beta_attn * scores
        scores = scores.masked_fill(~causal[:L, :L].unsqueeze(0), float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        alpha[:, 0, :] = 0.0
        return alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Epart, Ppart = self._parts(x)
        s = torch.cat([Epart, Ppart], dim=-1)
        alpha = self.attn_weights(s)
        c_s = torch.einsum("bij,bjd->bid", alpha, s)
        c_E = c_s[..., : self.dE]
        h_E = torch.einsum("bij,jk->bik", c_E, self.WE)
        logits_full = self.beta_class * (h_E @ self.E.weight.t())
        return logits_full





# ----------------------------
# Training
# ----------------------------

@dataclass
class TrainConfig:
    n_epochs: int = 40
    lr: float = 3e-3
    batch_size: int = 100

    # Snapshot / animation cadence:
    # Save one snapshot every `epochs_per_frame` epochs (approx; implemented via batch_counter modulus).
    epochs_per_frame: float = 0.5

    # Regularization knobs
    l2_lambda: float = 1e-5

    # Logging
    log_every_epochs: int = 50

    save_path: Optional[str] = None   # keep field if you want, but train_model won't use it


@dataclass
class TrainSnapshots:
    # All are stored on CPU as numpy arrays for easy animation
    pos_snapshots: List[np.ndarray]
    E_snapshots: List[np.ndarray]
    alpha_snapshots: List[np.ndarray]
    D_abs_snapshots: List[np.ndarray]          # |empirical_logits - target_logits|, if target provided
    P_abs_snapshots: List[np.ndarray]          # |P_model - P_target|, if target provided (row-softmax)
    snapshot_steps: List[int]                 # batch_counter at snapshot time


def _full_ce_loss(model: PosAttn_TokenOut, seqs: torch.Tensor) -> float:
    """Full-dataset next-token CE (no L2), computed in one pass."""
    V = model.V
    x_input = seqs[:, :-1]
    targets = seqs[:, :-1]
    logits = model(x_input)
    return float(
        F.cross_entropy(
            logits[:, 1:, :].reshape(-1, V),
            targets[:, 1:].reshape(-1),
        ).item()
    )


def train_model(
    model,
    train_seqs,
    *,
    test_seqs=None,
    theoretical_entropy_floor=None,
    target_logits=None,
    config: TrainConfig = TrainConfig(),
):
    """
    Train on next-token prediction using your model forward().
    """
    device = train_seqs.device

    model = model.to(device)
    train_seqs = train_seqs.to(device)
    if test_seqs is not None:
        test_seqs = test_seqs.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    n_seqs_train = int(train_seqs.shape[0])
    steps_per_epoch = int(math.ceil(n_seqs_train / config.batch_size))
    batches_per_snapshot = max(1, int(math.ceil(max(config.epochs_per_frame, 0.0) * steps_per_epoch)))

    snaps = TrainSnapshots(
        pos_snapshots=[],
        E_snapshots=[],
        alpha_snapshots=[],
        D_abs_snapshots=[],
        P_abs_snapshots=[],
        snapshot_steps=[],
    )

    V = model.V
    batch_counter = 0
    t0 = time.time()

    # keep track of latest training loss (with reg); also compute a pure CE if needed later

    for epoch in range(config.n_epochs):
        perm = torch.randperm(n_seqs_train, device=device)

        for i in range(0, n_seqs_train, config.batch_size):
            idx = perm[i : i + config.batch_size]
            x = train_seqs[idx]  # (B, seq_len)

            x_input = x[:, :-1]      # (B, L-1)
            targets = x[:, :-1]      # same as original setup

            logits = model(x_input)  # (B, L-1, V)

            ce_loss = F.cross_entropy(
                logits[:, 1:, :].reshape(-1, V),
                targets[:, 1:].reshape(-1),
            )

            loss = ce_loss
            if config.l2_lambda > 0:
                reg = (model.W.pow(2).sum()) #+ model.k_pos.pow(2)
                reg = reg + model.Ppos.pow(2).sum() / model.Ppos.numel()
                reg = reg + model.E.weight.pow(2).sum() / model.E.weight.numel()
                loss = loss + config.l2_lambda * reg

            #last_loss = loss.detach()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # snapshots (store CPU numpy)
            if (batch_counter % batches_per_snapshot) == 0:
                with torch.no_grad():
                    # alpha snapshot uses a single batch (fine for viz)
                    Epart, Ppart = model._parts(x_input)
                    s = torch.cat([Epart, Ppart], dim=-1)
                    alpha0 = model.attn_weights(s)[0]

                    snaps.pos_snapshots.append(model.Ppos.detach().cpu().numpy().copy())
                    snaps.E_snapshots.append(model.E.weight.detach().cpu().numpy().copy())
                    snaps.alpha_snapshots.append(alpha0.detach().cpu().numpy().copy())
                    snaps.snapshot_steps.append(int(batch_counter))

                    if target_logits is not None:
                        E_snapshot = model.E.weight.detach()
                        beta_class_snapshot = float(model.beta_class.item())
                        WE_snapshot = model.WE.detach()
                        empirical_logits = beta_class_snapshot * (E_snapshot @ WE_snapshot @ E_snapshot.t())

                        tgt = target_logits.to(empirical_logits.device)

                        # existing: logit-space absolute error
                        D_abs = (empirical_logits - tgt).abs()
                        snaps.D_abs_snapshots.append(D_abs.detach().cpu().numpy().copy())

                        # new: probability-space absolute error (row-wise distributions)
                        eps = 1e-6
                        P_model = torch.softmax(empirical_logits, dim=1)
                        P_tgt = torch.softmax(tgt, dim=1)
                        P_rel = (P_model - P_tgt).abs() / (P_tgt + eps)
                        snaps.P_abs_snapshots.append(P_rel.detach().cpu().numpy().copy())

            batch_counter += 1

        if config.log_every_epochs and ((epoch % config.log_every_epochs) == 0):
            # Print CE (more interpretable), and optionally total loss
            with torch.no_grad():
                ce_val = float(ce_loss.item())
            elapsed = time.time() - t0
            print(
                f"epoch {epoch:4d}/{config.n_epochs-1} | "
                f"ce {ce_val:.6f} | "
                f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                f"elapsed {elapsed:.1f}s"
            )
    
    print(
        f"[train] n_epochs={config.n_epochs} n_seqs_train={n_seqs_train} "
        f"batch_size={config.batch_size} steps_per_epoch={steps_per_epoch} \n"
    )


    ## Computes loss without penalty terms 
    model.eval()
    with torch.no_grad():
        final_train_ce = _full_ce_loss(model, train_seqs)
        final_test_ce = _full_ce_loss(model, test_seqs) if test_seqs is not None else None
    model.train()

    meta = dict(
        steps_per_epoch=steps_per_epoch,
        batches_per_snapshot=batches_per_snapshot,
        epochs_per_frame=config.epochs_per_frame,
        final_train_loss=final_train_ce,
        final_test_loss=final_test_ce,
        theoretical_entropy_floor=float(theoretical_entropy_floor) if theoretical_entropy_floor is not None else None,
    )

    # --- compute transformation rec + transformed snapshots at end of training ---
    # Avoid circular import at module import time; local import is cheap here.
    from source_code.analysis import whiten_embeddings_and_recover_W

    with torch.no_grad():
        rec = whiten_embeddings_and_recover_W(
            model.E.weight.detach(),
            float(model.beta_class.detach().cpu().item()),
            float(model.beta_attn.detach().cpu().item()),
            model.WE.detach(),
            model.Mp.detach(),
            model.Ppos.detach(),
        )
    
    T_E = rec["A"] @ rec["U"]  # numpy (dE,dE)

    # Transform ONLY the embedding snapshots; keep everything else identical
    # snaps.E_snapshots is a list of numpy arrays (V,dE) in your current dashboard code.
    snaps_tilde = copy.copy(snaps)
    E_snapshots_tilde = []
    for E in snaps.E_snapshots:
        E_snapshots_tilde.append(E @ T_E)
    snaps_tilde.E_snapshots = E_snapshots_tilde

    # store in meta too (optional)
    meta = dict(meta)
    meta["has_transformation_data"] = True

    return model, snaps, meta, rec, snaps_tilde


def _snap_to_cpu(obj):
    """Move nested snapshot structures to CPU-friendly python/torch objects."""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, (list, tuple)):
        return type(obj)(_snap_to_cpu(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _snap_to_cpu(v) for k, v in obj.items()}
    return obj


def save_training_artifacts(
    save_path: str,
    *,
    model,
    snaps,
    meta: dict,
):
    """
    Save checkpoint in the schema expected by load_training_artifacts():
      - model_hparams
      - model_state_dict
      - snaps (dict with required fields)
    You can still include extra keys, but these must exist.
    """
    model_hparams = dict(
        V=int(model.V),
        L=int(model.L),
        dE=int(model.dE),
        use_batch_invariant_alpha=bool(getattr(model, "use_batch_invariant_alpha", True)),
        pos_seed=None,  # record if you used one
    )

    snaps_payload = dict(
        pos_snapshots=snaps.pos_snapshots,
        E_snapshots=snaps.E_snapshots,
        alpha_snapshots=snaps.alpha_snapshots,
        D_abs_snapshots=snaps.D_abs_snapshots,
        P_abs_snapshots=getattr(snaps, "P_abs_snapshots", None),
        snapshot_steps=snaps.snapshot_steps,
    )

    ckpt = dict(
        model_hparams=_snap_to_cpu(model_hparams),
        model_state_dict=_snap_to_cpu(model.state_dict()),
        snaps=_snap_to_cpu(snaps_payload),
        meta=_snap_to_cpu(meta),
    )
    torch.save(ckpt, save_path)
    return save_path


def save_transformation_data(
    out_path: str,
    *,
    rec: dict,
    snaps_tilde,
):
    """
    Save whitening/rotation transform + transformed embedding snapshots for downstream plotting.
    """
    payload = dict(
        rec=_snap_to_cpu(rec),
        snaps_tilde=_snap_to_cpu(snaps_tilde),
    )
    torch.save(payload, out_path)
    return out_path


def load_training_artifacts(path: str, *, map_location: str = "cpu"):
    """
    Load a saved checkpoint and return (model, snaps, meta). # , config_dict, extra
    """
    ckpt = torch.load(path, map_location=map_location)

    h = ckpt["model_hparams"]
    model = PosAttn_TokenOut(
        V=h["V"],
        L=h["L"],
        dE=h["dE"],
        use_batch_invariant_alpha=h.get("use_batch_invariant_alpha", True),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    s = ckpt["snaps"]
    snaps = TrainSnapshots(
        pos_snapshots=s["pos_snapshots"],
        E_snapshots=s["E_snapshots"],
        alpha_snapshots=s["alpha_snapshots"],
        D_abs_snapshots=s.get("D_abs_snapshots", []),
        P_abs_snapshots=s.get("P_abs_snapshots", []),
        snapshot_steps=s["snapshot_steps"],
    )
    return model, snaps, ckpt.get("meta", {})