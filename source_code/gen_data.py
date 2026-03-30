import numpy as np
import torch
import os
from typing import Dict, Optional, Tuple


def fibonacci_sphere(V: int) -> torch.Tensor:
    """
    Generate approximately uniform points on the unit sphere.

    Returns:
        torch.Tensor: Tensor of shape (V, 3), where each row is a 3D token embedding.
    """
    indices = torch.arange(0, V, dtype=torch.get_default_dtype()) + 0.5
    phi = torch.acos(1 - 2 * indices / V)
    theta = torch.pi * (1 + 5**0.5) * indices
    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=1)

def make_true_P_from_embeddings(V=20, dim=3, beta=1.0, W_true: Optional[torch.Tensor] = None, 
    blocked_tokens: Optional[torch.Tensor] = None):
    """
    Construct true token embeddings and the induced Markov transition matrix.

    Returns:
        tuple:
            - E_true: tensor of shape (V, dim) containing token embeddings
            - P: tensor of shape (V, V) with row-stochastic transition probabilities
            - W_true: tensor of shape (dim, dim) giving the token metric (Euclidean if unspecified)
    """
    E_true = fibonacci_sphere(V)

    
    ## Default measure on the sphere is Euclidean
    if W_true is None:
        W_true = torch.eye(dim, dtype=torch.get_default_dtype())

    ## Defines true probability distribution
    logits = beta * (E_true @ W_true @ E_true.t())
    logits = logits - logits.max(dim=1, keepdim=True)[0]  # gauge-fix
    P = torch.exp(logits)
    P = P / (P.sum(dim=1, keepdim=True) + 1e-12)

    # --- zero out transitions into selected tokens ---
    if blocked_tokens is not None and len(blocked_tokens) > 0:
        # ensure long tensor on same device
        blocked_tokens = blocked_tokens.to(P.device).long()
        P[:, blocked_tokens] = 0.0

    # renormalize rows to keep P row-stochastic
    row_sums = P.sum(dim=1, keepdim=True) + 1e-12
    P = P / row_sums

    # OPTIONAL: remove any token-id ordering by relabeling states
    #P, E_true, _ = permute_token_ids(P, E_true)

    return E_true, P, W_true

def sample_markov_sequences(P: torch.Tensor, n_seqs=200, seq_len=500) -> torch.Tensor:
    """
    Generates n_seqs sequences of length seq_len by performing a token-to-token random walk, with transition
    probability given by P. 

    return has shape (n_seqs, seq_len), a tensor of sequeences, each sequence being a list of token ID's. 
    """
    V = P.shape[0]
    seqs = torch.zeros(n_seqs, seq_len, dtype=torch.long)
    for s in range(n_seqs):
        cur = torch.randint(0, V, ()).item()
        seqs[s, 0] = cur
        for t in range(1, seq_len):
            cur = torch.multinomial(P[cur], 1).item()
            seqs[s, t] = cur
    return seqs

def markov_entropy_rate(P: np.ndarray, *, tol: float = 1e-12, max_iter: int = 200_000) -> float:
    """
    Theoretical entropy rate for a (row-stochastic) Markov chain:
        H = sum_i pi_i * H(P[i,:])
    where pi is the stationary distribution (left eigenvector).

    P: shape (V, V), rows sum to 1.

    Returns:
        float: Entropy rate in nats.
    """
    P = np.asarray(P, dtype=np.float64)
    V = P.shape[0]
    assert P.shape == (V, V)

    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-10):
        raise ValueError("P must be row-stochastic (rows sum to 1).")

    # Stationary distribution via power iteration on the left: pi_{t+1} = pi_t P
    pi = np.full(V, 1.0 / V, dtype=np.float64)
    for _ in range(max_iter):
        pi_next = pi @ P
        if np.max(np.abs(pi_next - pi)) < tol:
            pi = pi_next
            break
        pi = pi_next
    pi = pi / pi.sum()

    # Row entropies: H(P[i]) = -sum_j P[i,j] log P[i,j]
    eps = 1e-300
    row_H = -(P * np.log(P + eps)).sum(axis=1)

    return float((pi * row_H).sum())

def save_dataset(
    path: str,
    *,
    train_seqs: torch.Tensor,
    test_seqs: torch.Tensor,
    E_true: Optional[torch.Tensor] = None,
    W_true: Optional[torch.Tensor] = None,
    P_true: Optional[torch.Tensor] = None,
    meta: Optional[Dict] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "format_version": 1,
        "train_seqs": train_seqs.detach().cpu(),
        "test_seqs": test_seqs.detach().cpu(),
        "E_true": E_true.detach().cpu() if E_true is not None else None,
        "W_true": W_true.detach().cpu() if W_true is not None else None,
        "P_true": P_true.detach().cpu() if P_true is not None else None,
        "meta": meta or {},
    }
    torch.save(ckpt, path)


def load_dataset(
    path: str,
    *,
    map_location: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Dict]:
    ckpt = torch.load(path, map_location=map_location)
    return (
        ckpt["train_seqs"],
        ckpt["test_seqs"],
        ckpt.get("E_true", None),
        ckpt.get("W_true", None),
        ckpt.get("P_true", None),
        ckpt.get("meta", {}),
    )

def permute_token_ids(P: torch.Tensor, E: Optional[torch.Tensor] = None, *, generator=None):
    """
    Randomly relabel token indices by a permutation. Prevents any bias found in the token initialization.
    Returns (P_perm, E_perm, perm), where P_perm = P[perm][:,perm] and E_perm = E[perm].
    """
    V = P.shape[0]
    perm = torch.randperm(V, generator=generator)
    P_perm = P[perm][:, perm]
    E_perm = E[perm] if E is not None else None
    return P_perm, E_perm, perm