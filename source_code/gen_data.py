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

def sample_memory_sequences(
    E_true: torch.Tensor,
    W_true: torch.Tensor,
    *,
    beta: float = 1.0,
    mu: float = 0.1,
    n_seqs: int = 200,
    seq_len: int = 500,
) -> torch.Tensor:
    """
    Sample sequences from a latent-memory process:

        P(E^{t+1} | m^t) ∝ exp(m^t W_true E^{t+1})
        m^{t+1} = (1 - mu) m^t + mu E^t

    where E^t is the embedding of the current token at time t.
    """
    V, dim = E_true.shape
    assert W_true.shape == (dim, dim)

    seqs = torch.zeros(n_seqs, seq_len, dtype=torch.long)

    for s in range(n_seqs):
        cur = torch.randint(0, V, ()).item()
        seqs[s, 0] = cur
        m = E_true[cur].clone()

        for t in range(1, seq_len):
            tmp = (m.unsqueeze(0) @ W_true)
            logits = beta * (tmp @ E_true.t()).squeeze(0)
            probs = torch.softmax(logits, dim=0)
            nxt = torch.multinomial(probs, 1).item()
            seqs[s, t] = nxt

            # FIX: update memory toward the *new* token
            E_next = E_true[nxt]
            m = (1.0 - mu) * m + mu * E_next

            cur = nxt

    return seqs


def make_true_memory_sequences(
    V: int = 20,
    dim: int = 3,
    *,
    beta: float = 1.0,
    mu: float = 0.1,
    W_true: Optional[torch.Tensor] = None,
    n_seqs: int = 200,
    seq_len: int = 500,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience wrapper for the latent-memory process:

        P(E^{t+1} | m^t) ∝ exp(m^t W_true E^{t+1})
        m^{t+1} = (1 - mu) m^t + mu E^t
    """
    E_true = fibonacci_sphere(V)  # (V, dim) if adapted for dim; 3D sphere in your case

    if W_true is None:
        W_true = torch.eye(dim, dtype=torch.get_default_dtype())

    seqs = sample_memory_sequences(
        E_true,
        W_true,
        beta=beta,
        mu=mu,
        n_seqs=n_seqs,
        seq_len=seq_len,
    )
    return E_true, W_true, seqs


    ## more complicated data_gen

def sample_conv_memory_sequences(
    E_true: torch.Tensor,
    W_true: torch.Tensor,
    alpha: torch.Tensor,
    *,
    beta: float = 1.0,
    n_seqs: int = 200,
    seq_len: int = 500,
) -> torch.Tensor:
    """
    Sample sequences from a convolutional-memory process:

        m_t = sum_{k >= 0} alpha_k E_{t-k}   (discrete convolution, truncated)
        P(E_{t+1} | m_t) ∝ exp( m_t^T W_true E_{t+1} )

    Here alpha is a 1D tensor of length K: alpha[k] = weight at lag k.
    We use only lags k <= t at time t, and renormalize alpha over the
    available window.

    Args:
        E_true: (V, dim) token embedding matrix.
        W_true: (dim, dim) bilinear form matrix.
        alpha:  (K,) convolution kernel over lags k = 0..K-1.
        beta:   inverse temperature.
        n_seqs: number of sequences to sample.
        seq_len: length of each sequence.

    Returns:
        seqs: LongTensor of shape (n_seqs, seq_len) with token indices.
    """
    V, dim = E_true.shape
    assert W_true.shape == (dim, dim)
    alpha = alpha.to(E_true.device).to(E_true.dtype)
    K = alpha.shape[0]

    # ensure alpha is nonnegative and normalized (in case caller forgot)
    alpha = torch.clamp(alpha, min=0)
    if alpha.sum() <= 0:
        raise ValueError("alpha must have some positive mass.")
    alpha = alpha / alpha.sum()

    seqs = torch.zeros(n_seqs, seq_len, dtype=torch.long)

    for s in range(n_seqs):
        # initial token
        cur = torch.randint(0, V, ()).item()
        seqs[s, 0] = cur

        # history of token indices; history[t] = index at time t
        history = [cur]

        for t in range(seq_len - 1):
            # compute m_t from available history: E_t, E_{t-1}, ...
            # valid lags: k = 0..min(K-1, t)
            max_k = min(K - 1, t)
            # slice the tail of history: [E_t, E_{t-1}, ..., E_{t-max_k}]
            # and corresponding alpha[0..max_k], renormalized
            alpha_used = alpha[: max_k + 1]
            alpha_used = alpha_used / alpha_used.sum()

            # accumulate m_t
            m = torch.zeros(dim, dtype=E_true.dtype, device=E_true.device)
            for k in range(max_k + 1):
                token_idx = history[t - k]          # x_{t-k}
                m = m + alpha_used[k] * E_true[token_idx]

            # logits_j = beta * m^T W_true E_j
            tmp = (m.unsqueeze(0) @ W_true)              # (1, dim)
            logits = beta * (tmp @ E_true.t()).squeeze(0)  # (V,)

            probs = torch.softmax(logits, dim=0)
            nxt = torch.multinomial(probs, 1).item()

            seqs[s, t + 1] = nxt
            history.append(nxt)
            cur = nxt

    return seqs


def make_true_conv_memory_sequences(
    V: int = 20,
    dim: int = 3,
    *,
    beta: float = 1.0,
    W_true: Optional[torch.Tensor] = None,
    n_seqs: int = 200,
    seq_len: int = 500,
    # kernel controls
    K: int = 12,
    peak_lag: int = 5,
    sigma: float = 1.0,
    alpha: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience wrapper for a convolutional-memory process:

        m_t = sum_k alpha_k E_{t-k},
        P(E_{t+1} | m_t) ∝ exp( m_t^T W_true E_{t+1} ).

    If `alpha` is not provided, we build a more complex kernel over lags
    k = 0..K-1: a mixture of two Gaussians plus a mild sinusoidal term.
    """
    E_true = fibonacci_sphere(V)  # (V, 3) for dim=3

    if W_true is None:
        W_true = torch.eye(dim, dtype=torch.get_default_dtype())

    if alpha is None:
        ks = torch.arange(K, dtype=torch.get_default_dtype())

        # main bump near `peak_lag`
        gauss_main = torch.exp(- (ks - float(peak_lag)) ** 2 / (2.0 * sigma ** 2))

        # secondary broader bump farther back in time
        peak_far = float(peak_lag + 4)
        sigma_far = sigma * 2.5
        gauss_far = 0.4 * torch.exp(- (ks - peak_far) ** 2 / (2.0 * sigma_far ** 2))

        # mild sinusoidal modulation over lags
        sinus = 0.15 * (1.0 + torch.sin(0.7 * ks))

        alpha = gauss_main + gauss_far + sinus
        alpha = torch.clamp(alpha, min=0)
        alpha = alpha / alpha.sum()

    seqs = sample_conv_memory_sequences(
        E_true=E_true,
        W_true=W_true,
        alpha=alpha,
        beta=beta,
        n_seqs=n_seqs,
        seq_len=seq_len,
    )
    return E_true, W_true, alpha, seqs