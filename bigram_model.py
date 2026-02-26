import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting backend

torch.manual_seed(0)
np.random.seed(0)
torch.set_default_dtype(torch.float64)

# -----------------------------
# Utilities: true generator + sampling
# -----------------------------
def make_true_P_from_embeddings(V=20, dim=3, beta=1.0):
    E_true = torch.randn(V, dim)
    E_true = E_true / (E_true.norm(dim=1, keepdim=True) + 1e-12)
    W_true = torch.tensor([[1.0, 0., 0.0],
                  [0.0,  1.0, 0.0],
                  [0.0, 0.0, 1.0]], dtype=torch.float64)
    logits = beta * (E_true @ W_true @ E_true.t())
    logits = logits - logits.max(dim=1, keepdim=True)[0]
    P = torch.exp(logits)
    P = P / (P.sum(dim=1, keepdim=True) + 1e-12)
    return E_true, P, W_true  # <--- Add W_true here

def sample_markov_sequences(P, n_seqs=200, seq_len=500):
    V = P.shape[0]
    seqs = torch.zeros(n_seqs, seq_len, dtype=torch.long)
    for s in range(n_seqs):
        cur = torch.randint(0, V, ()).item()
        seqs[s, 0] = cur
        for t in range(1, seq_len):
            cur = torch.multinomial(P[cur], 1).item()
            seqs[s, t] = cur
    return seqs

def disorder_sequences(seqs, mode="none"):
    if mode == "none":
        return seqs
    n_seqs, L = seqs.shape
    if mode == "permute_within_each_sequence":
        out = seqs.clone()
        for s in range(n_seqs):
            perm = torch.randperm(L)
            out[s] = out[s, perm]
        return out
    if mode == "shuffle_all_tokens":
        flat = seqs.reshape(-1)
        perm = torch.randperm(flat.numel())
        return flat[perm].reshape(n_seqs, L)
    raise ValueError(f"Unknown disorder mode: {mode}")

def bigram_counts_to_Q(seqs, V):
    counts = torch.zeros(V, V, dtype=torch.float64)
    for s in range(seqs.shape[0]):
        a = seqs[s, :-1]
        b = seqs[s, 1:]
        for i in range(a.numel()):
            counts[a[i], b[i]] += 1.0
    row_sums = counts.sum(dim=1, keepdim=True).clamp_min(1e-12)
    Q = counts / row_sums
    row_freq = counts.sum(dim=1)
    row_freq = row_freq / row_freq.sum().clamp_min(1e-12)
    return Q, row_freq

def rowfreq_cond_entropy(Q, row_freq):
    Qs = Q.clamp_min(1e-16)
    return float(-(row_freq[:, None] * Qs * torch.log(Qs)).sum().item())

def rowfreq_cross_entropy(Q, P_hat, row_freq):
    Qs = Q.clamp_min(1e-16)
    Ph = P_hat.clamp_min(1e-16)
    return float(-(row_freq[:, None] * Qs * torch.log(Ph)).sum().item())

# -----------------------------
# Position: 2D circle (fixed)
# -----------------------------
def make_circle_positions(L):
    i = torch.arange(L, dtype=torch.get_default_dtype())
    theta = 2.0 * np.pi * i / float(L)
    p = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)  # (L,2)
    return p

# -----------------------------
# Position: 2D circle (fixed)
# -----------------------------
def make_random_positions(L, seed=None):
    "Return (L,2) random 2D vectors. Seed optional for reproducibility."
    if seed is not None:
        g = torch.Generator().manual_seed(int(seed))
        return torch.randn(L, 2, dtype=torch.get_default_dtype(), generator=g)
    return torch.randn(L, 2, dtype=torch.get_default_dtype())

# -----------------------------
# Model: position-only attention, token-only output
# -----------------------------
class PosAttn_TokenOut(nn.Module):
    """
    s_i = concat(E(x_i), p_i). Optionally L2-normalize s_i.
    Attention scores depend only on p_i, output logits only on E(x).
    """
    def __init__(self, V, L, dE=3, normalize_s=True):
        super().__init__()
        self.V = V
        self.L = L
        self.dE = dE
        self.normalize_s = normalize_s

        self.E = nn.Embedding(V, dE)

        p = make_random_positions(L)          # (L,2)
        
        #p = make_circle_positions(L)          # (L,2)
        self.Ppos = nn.Parameter(p)

        # position metric (2x2)
        self.Mp = nn.Parameter(torch.eye(2, dtype=torch.get_default_dtype()))

        # positive scales (avoid NaNs / sign flips)
        self.raw_beta_attn = nn.Parameter(torch.tensor(1.5, dtype=torch.get_default_dtype()))
        self.raw_tau = nn.Parameter(torch.tensor(0.0, dtype=torch.get_default_dtype()))

        # token metric for output
        self.WE = nn.Parameter(torch.eye(dE, dtype=torch.get_default_dtype()))

        # causal mask j < i, BUT we will also allow (0,0) to avoid all -inf row
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool), diagonal=-1)
        mask[0, 0] = True
        self.register_buffer("causal_mask", mask)

    @property
    def beta_attn(self):
        return F.softplus(self.raw_beta_attn) + 1e-6

    @property
    def tau(self):
        return F.softplus(self.raw_tau) + 1e-6

    def _parts(self, x):
        B, L = x.shape
        Epart = self.E(x)  # (B,L,dE)
        Ppart = self.Ppos.unsqueeze(0).expand(B, L, 2)  # (B,L,2)
        # normalize position vectors per-position so ||p_i|| == 1 before concat
        Ppart = Ppart #/ (Ppart.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        Epart = Epart #/ (Epart.norm(dim=-1, keepdim=True).clamp_min(1e-12))


        if not self.normalize_s:
            return Epart, Ppart

        s = torch.cat([Epart, Ppart], dim=-1)  # (B,L,dE+2)
        s = s #/ s.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return s[..., : self.dE], s[..., self.dE :]

    def attn_weights(self):
        L = self.L
        dE = self.dE
        # Build s_i for each position index (no batch)
        # For attention, we use dummy tokens (e.g., zeros) for E(x_i)
        E_dummy = torch.zeros(L, dE, dtype=self.Ppos.dtype, device=self.Ppos.device)
        Ppart = self.Ppos  # (L,2)
        s = torch.cat([E_dummy, Ppart], dim=1)  # (L, dE+2)

        #s = s / (s.norm(dim=-1, keepdim=True).clamp_min(1e-12))

        # Build block metric: zeros except for lower-right 2x2 block
        M_block = torch.zeros(dE + 2, dE + 2, dtype=self.Mp.dtype, device=self.Mp.device)
        M_block[dE:, dE:] = self.Mp  # lower-right 2x2 block is Mp

        scores = (s @ M_block) @ s.t()  # (L,L)
        scores = self.beta_attn * scores

        # ensure row 0 has a finite entry so softmax is defined
        scores = scores.clone()
        scores[0, 0] = 0.0

        scores = scores.masked_fill(~self.causal_mask, float("-inf"))
        alpha = torch.softmax(scores, dim=1)    # (L,L)
        # numerically: replace any residual NaNs (shouldn't happen now)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)

        attn_fixed = False
        if attn_fixed:
            alpha = torch.zeros(L, L, dtype=torch.get_default_dtype(), device=self.Ppos.device)
            alpha[0, 0] = 1.0
            alpha[1:, :-1] = torch.eye(L - 1, dtype=alpha.dtype, device=alpha.device)

        return alpha
    


    def forward(self, x):
        B, L = x.shape
        assert L == self.L

        Epart, _ = self._parts(x)

        alpha = self.attn_weights()  # (L,L)
        c = torch.einsum("ij,bjd->bid", alpha, Epart)  # (B,L,dE)

        logits = self.tau * ((c @ self.WE) @ self.E.weight.t())  # (B,L,V)
        return logits[:, 1:, :]  # predict t=1..L-1



@torch.no_grad()
def implied_bigram_Phat_forwardlike(model):
    E = model.E.weight  # (V,dE) raw
    denom = (E.pow(2).sum(dim=1, keepdim=True) + 1.0).sqrt().clamp_min(1e-12)
    Eeff = E / denom  # matches normalize_s=True with ||p||=1

    logits = model.tau * (Eeff @ model.WE @ E.t())  # (V,V)
    logits = logits - logits.max(dim=1, keepdim=True)[0]
    P_hat = torch.softmax(logits, dim=1)
    return P_hat



@torch.no_grad()
def full_data_loss(model, seqs):
    V = model.V
    logits = model(seqs)                 # (B,L-1,V)
    targets = seqs[:, 1:]                # (B,L-1)
    return float(F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1)).item())

@torch.no_grad()
def alpha_diagnostics(model):
    alpha = model.attn_weights()
    L = alpha.shape[0]
    mass_prev = []
    max_comp = []
    for i in range(1, L):
        mass_prev.append(alpha[i, i-1])
        # competitor among allowed j<i excluding i-1
        if i >= 2:
            competitors = alpha[i, :i-1]  # j=0..i-2
            max_comp.append(competitors.max())
        else:
            max_comp.append(torch.tensor(0.0, dtype=alpha.dtype))
    return float(torch.stack(mass_prev).mean().item()), float(torch.stack(max_comp).mean().item())

# -----------------------------
# Main experiment
# -----------------------------
def run(
    disorder_mode="none",
    V=100,
    dE=3,
    n_seqs=20000,
    seq_len=8,
    steps=501,
    lr=3e-3,
    batch_size=100,
    true_beta=1.0,
    whiten_positions=True,  # <--- Add this flag
):
    E_true, P_true, W_true = make_true_P_from_embeddings(V=V, dim=dE, beta=true_beta)
    seqs = sample_markov_sequences(P_true, n_seqs=n_seqs, seq_len=seq_len)
    seqs_data = disorder_sequences(seqs, mode=disorder_mode)

    Q, row_freq = bigram_counts_to_Q(seqs_data, V)
    H_row = rowfreq_cond_entropy(Q, row_freq)

    model = PosAttn_TokenOut(V=V, L=seq_len, dE=dE, normalize_s=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # snapshots recorded for animation
    pos_snapshots = []
    E_snapshots = []
    alpha_snapshots = []
    # Mp_snapshots = []  # <-- Add this line
    # WE_snapshots = []
    snapshot_steps = []

    l2_lambda = 1e-4  # You can adjust this value

    for step in range(steps):
        idx = torch.randint(0, n_seqs, (batch_size,))
        x = seqs_data[idx]

        logits = model(x)        # (bs,L-1,V)
        targets = x[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))

        p_norms = model.Ppos.norm(dim=1)
        E_norms = model.E.weight.norm(dim=1)
        radius_penalty = ((p_norms - 1.0) ** 2).mean() + ((E_norms - 1.0) ** 2).mean()
        
        #loss = loss + l2_lambda * radius_penalty

        opt.zero_grad()
        loss.backward()
        # optional: clip to be extra safe
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        if step % 300 == 0 or step == steps - 1:
            avg_prev, avg_comp = alpha_diagnostics(model)
            print(
                f"Step {step:4d} | loss {loss.item():.6f} | "
                f"alpha[i,i-1] mean {avg_prev:.3f} | max competitor mean {avg_comp:.3f} | "
                f"beta_attn {float(model.beta_attn.item()):.3f} | tau {float(model.tau.item()):.3f}"
            )
            # Print average embedding and position norms
            avg_E = model.E.weight.norm(dim=1).mean().item()
            avg_p = model.Ppos.data.norm(dim=1).mean().item()

            # Compute mean |s| (concatenated and normalized vector)
            with torch.no_grad():
                Epart, Ppart = model._parts(x)
                s = torch.cat([Epart, Ppart], dim=-1)
                mean_s = s.norm(dim=-1).mean().item()

            print(f"Mean |E|: {avg_E:.4f} | Mean |p|: {avg_p:.4f} | Mean |s|: {mean_s:.4f}")
        # record snapshots for animation (positions and entire embedding table)
        try:
            pos_snapshots.append(model.Ppos.detach().cpu().numpy().copy())  # (L,2)
            E_snapshots.append(model.E.weight.detach().cpu().numpy().copy())  # (V,dE)
            alpha_snapshots.append(model.attn_weights().detach().cpu().numpy().copy())  # (L,L)
            # Mp_snapshots.append(model.Mp.detach().cpu().numpy().copy())  # <-- Add this line
            # WE_snapshots.append(model.Mp.detach().cpu().numpy().copy())  # <-- Add this line
            snapshot_steps.append(step)
        except Exception:
            pass

    L_full = full_data_loss(model, seqs_data)

    #P_hat_bigram = implied_bigram_Phat(model)
    P_hat_bigram = implied_bigram_Phat_forwardlike(model)
    CE_row = rowfreq_cross_entropy(Q, P_hat_bigram, row_freq)
    
    # P_hat_bigram = implied_bigram_Phat_consistent(model)
    # CE_row = rowfreq_cross_entropy(Q, P_hat_bigram, row_freq)

    avg_prev, avg_comp = alpha_diagnostics(model)

    print("\n=== Summary ===")
    print(f"Disorder mode: {disorder_mode}")
    print(f"Full-data loss (model forward): {L_full:.6f}")
    print(f"Entropy floor H(Q) (rowfreq-weighted): {H_row:.6f}")
    print(f"Rowfreq CE(Q, P_hat_bigram): {CE_row:.6f}   gap: {CE_row - H_row:.6f}")
    print(f"Alpha diagnostic: mean alpha[i,i-1]={avg_prev:.3f} | mean max competitor={avg_comp:.3f}")
    print("||P_hat_bigram - Q||_F =", float(torch.norm(P_hat_bigram - Q).item()))
    print("||P_hat_bigram - P_true||_F =", float(torch.norm(P_hat_bigram - P_true).item()))

    # print the learned token metric WE for inspection
    try:
        WE_val = model.WE.detach().cpu().numpy()
        print("\nLearned token metric WE:")
        print(np.round(WE_val, 4))

        # Compute symmetric and antisymmetric parts for token metric
        S_E = 0.5 * (WE_val + WE_val.T)
        K_E = 0.5 * (WE_val - WE_val.T)
        eigvals_E, eigvecs_E = np.linalg.eigh(S_E)
        if whiten_positions:
            eigvals_E_clipped = np.clip(eigvals_E, 1e-8, None)
            U_E = eigvecs_E @ np.diag(1.0 / np.sqrt(eigvals_E_clipped))
        else:
            U_E = eigvecs_E  # Just diagonalize, no whitening

        W_S = U_E.T @ S_E @ U_E
        W_K = U_E.T @ K_E @ U_E

        print("\nW_S = 1/2 U_E^T(WE + WE^T)U_E:")
        print(np.round(W_S, 4))
        print("\nW_K = 1/2 U_E^T(WE - WE^T)U_E:")
        print(np.round(W_K, 4))
    except Exception as e:
        print("Could not print model.WE or transformed token metrics:", e)
        # print the learned position metric Mp for inspection
    try:
        Mp_val = model.Mp.detach().cpu().numpy()
        print("\nLearned position metric Mp:")
        print(np.round(Mp_val, 4))

        # Compute symmetric and antisymmetric parts
        S = 0.5 * (Mp_val + Mp_val.T)
        K = 0.5 * (Mp_val - Mp_val.T)
        eigvals, eigvecs = np.linalg.eigh(S)
        if whiten_positions:
            eigvals_clipped = np.clip(eigvals, 1e-8, None)
            U = eigvecs @ np.diag(1.0 / np.sqrt(eigvals_clipped))
        else:
            U = eigvecs  # Just diagonalize, no whitening

        # Transform symmetric and antisymmetric parts
        M_S = U.T @ S @ U
        M_K = U.T @ K @ U

        print("\nM_S = 1/2 U^T(M + M^T)U:")
        print(np.round(M_S, 4))
        print("\nM_K = 1/2 U^T(M - M^T)U:")
        print(np.round(M_K, 4))

    except Exception as e:
        print("Could not print model.Mp or transformed metrics:", e)
    # ---------------------------
    # Build and save animations for Ppos and embeddings over snapshots
    # ---------------------------
    if len(pos_snapshots) == 0:
        # no snapshots recorded (very short run) — use final state
        pos_snapshots = [model.Ppos.detach().cpu().numpy().copy()]
        E_snapshots = [model.E.weight.detach().cpu().numpy().copy()]
        snapshot_steps = [steps - 1]

    # Subsample snapshots to every 50th frame for faster animation
    stride = 50
    pos_snapshots = pos_snapshots[::stride]
    E_snapshots = E_snapshots[::stride]
    alpha_snapshots = alpha_snapshots[::stride]
    # Mp_snapshots = Mp_snapshots[::stride]
    # WE_snapshots = WE_snapshots[::stride]
    snapshot_steps = snapshot_steps[::stride]
    Mp_final = model.Mp.detach().cpu().numpy()

    # Position animation (2D) -- show transformed positions in metric geometry
    try:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_aspect("equal", "box")
        seq_len = pos_snapshots[0].shape[0]
        reds = np.linspace(0.3, 1.0, seq_len)
        colors = np.array([[r, 0, 0] for r in reds])

        # Prepare transformed positions for all snapshots
        ptilde_snapshots = []
        for i, p in enumerate(pos_snapshots):
            Mp = Mp_final
            S = 0.5 * (Mp + Mp.T)
            eigvals, eigvecs = np.linalg.eigh(S)
            if whiten_positions:
                eigvals_clipped = np.clip(eigvals, 1e-8, None)
                U = eigvecs @ np.diag(1.0 / np.sqrt(eigvals_clipped))
            else:
                U = eigvecs  # Just diagonalize, no whitening
            ptilde = p @ U  # (L,2)
            ptilde_snapshots.append(ptilde)

        all_ptilde = np.vstack(ptilde_snapshots)
        ax.set_xlim(all_ptilde[:,0].min() - 0.2, all_ptilde[:,0].max() + 0.2)
        ax.set_ylim(all_ptilde[:,1].min() - 0.2, all_ptilde[:,1].max() + 0.2)

        scatter = ax.scatter(ptilde_snapshots[0][:,0], ptilde_snapshots[0][:,1], s=40, c=colors)

        def init_ptilde():
            scatter.set_offsets(ptilde_snapshots[0])
            scatter.set_color(colors)
            return (scatter,)

        def update_ptilde(i):
            scatter.set_offsets(ptilde_snapshots[i])
            scatter.set_color(colors)
            ax.set_title(f"Transformed Ppos snapshot step {snapshot_steps[i]}")
            return (scatter,)

        anim_ptilde = animation.FuncAnimation(fig, update_ptilde, init_func=init_ptilde,
                                              frames=len(ptilde_snapshots), interval=80, blit=True)
        plt.show()
        plt.close(fig)
    except Exception as e:
        print("Could not build/show transformed position animation:", e)

    # Embedding animation (3D only) -- color by E_true, transformed by final token metric
    try:
        dE_now = E_snapshots[0].shape[1]
        fig = plt.figure(figsize=(6,5))
        ax3 = fig.add_subplot(111, projection='3d')
        V = E_snapshots[0].shape[0]
        emb_colors = (E_true[:, :3].detach().cpu().numpy() + 1) / 2  # RGB in [0,1]

        # Use final WE for all frames
        WE_final = model.WE.detach().cpu().numpy()
        S_E = 0.5 * (WE_final + WE_final.T)
        eigvals_E, eigvecs_E = np.linalg.eigh(S_E)
        if whiten_positions:
            eigvals_E_clipped = np.clip(eigvals_E, 1e-8, None)
            U_E = eigvecs_E @ np.diag(1.0 / np.sqrt(eigvals_E_clipped))
        else:
            U_E = eigvecs_E

        Etilde_snapshots = [E @ U_E for E in E_snapshots]

        all_Etilde = np.vstack(Etilde_snapshots)
        mins = all_Etilde.min(axis=0)[:3]
        maxs = all_Etilde.max(axis=0)[:3]
        for dim in range(3):
            rng = maxs[dim] - mins[dim]
            ax3.set_xlim(mins[0]-0.1*rng, maxs[0]+0.1*rng)
            ax3.set_ylim(mins[1]-0.1*rng, maxs[1]+0.1*rng)
            ax3.set_zlim(mins[2]-0.1*rng, maxs[2]+0.1*rng)
        ax3.set_xlim(-3., 3.)
        ax3.set_ylim(-3., 3.)
        ax3.set_zlim(-3., 3.)

        sc = ax3.scatter(Etilde_snapshots[0][:,0], Etilde_snapshots[0][:,1], Etilde_snapshots[0][:,2], s=30, c=emb_colors)
        ax3.set_title("")

        def update_emb3(i):
            Ecur = Etilde_snapshots[i]
            xs = Ecur[:,0]
            ys = Ecur[:,1]
            zs = Ecur[:,2]
            sc._offsets3d = (xs, ys, zs)
            sc.set_color(emb_colors)
            ax3.set_title(f"Transformed Embeddings snapshot step {snapshot_steps[i]}")
            return (sc,)

        anim_emb = animation.FuncAnimation(fig, update_emb3, frames=len(Etilde_snapshots),
                                           interval=80, blit=False)
        plt.show()
        plt.close(fig)
    except Exception as e:
        print("Could not build/show transformed embedding animation:", e)

    # Alpha heatmap animation
    try:
        # Ensure all snapshot lists are the same length
        n_frames = min(len(alpha_snapshots), len(snapshot_steps))
        alpha_snapshots = alpha_snapshots[:n_frames]
        snapshot_steps = snapshot_steps[:n_frames]

        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(alpha_snapshots[0], vmin=0, vmax=1, cmap='hot')
        cbar = plt.colorbar(im, ax=ax)
        ax.set_title(f"Alpha heatmap step {snapshot_steps[0]}")

        def update_alpha(i):
            im.set_data(alpha_snapshots[i])
            ax.set_title(f"Alpha heatmap step {snapshot_steps[i]}")
            return (im,)

        anim_alpha = animation.FuncAnimation(
            fig, update_alpha, frames=n_frames, interval=80, blit=False
        )
        plt.show()
        plt.close(fig)
    except Exception as e:
        print("Could not build/show alpha heatmap animation:", e)
    
    WE_learned = model.WE.detach().cpu().numpy()
    tau = float(model.tau.item())
    WE_learned_rescaled = tau * WE_learned

    S_learned = 0.5 * (WE_learned_rescaled + WE_learned_rescaled.T)
    S_true = 0.5 * (W_true + W_true.T)

    print()
    print(f"S_learned : {S_learned}")
    print(f"S_true : {S_true}")
    print()
    print("Eigenvalues S_learned:", np.sort(np.linalg.eigvalsh(S_learned)))
    print("Eigenvalues S_true:", np.sort(np.linalg.eigvalsh(S_true)))
    print()
    print("Eigenvalues S_learned:", np.sort(np.linalg.eigvalsh(WE_learned_rescaled)))
    print("Eigenvalues S_true:", np.sort(np.linalg.eigvalsh(W_true)))

if __name__ == "__main__":
    run()  # Set to False to just diagonalize
