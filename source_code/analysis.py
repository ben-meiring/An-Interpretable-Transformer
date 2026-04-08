import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation


@dataclass
class DashboardConfig:
    out_dir: str = "model_and_data"
    filename: str = "training_animation.mp4"
    interval_ms: int = 60
    dpi: int = 100

    # Embedding plot controls
    elev: float = 16 
    azim: float = 35
    emb_lim: float = 1.4

    # Heatmap controls
    q_vmin: float = 0.0
    q_vmax: float = 0.2 #0.2

    # Text placement (match safe_backup dashboard)
    epoch_xy: Tuple[float, float] = (0.5, 0.85)
    stats_xy: Tuple[float, float] = (0.5, 0.82)


def whiten_embeddings_and_recover_W(
    E_hat: torch.Tensor,
    beta_class_hat: float,
    beta_attn_hat: float,
    WE_hat: torch.Tensor,
    M_hat: torch.Tensor,
    P_hat: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """
    Recovers whitened token and positional representations from learned model parameters.

    Our embedding tokens and weights are only defined up to several gauge choices.
    We fix the gauge for the tokens and W-weights by:

        - choosing the score to have zero mean (fixes softmax normalization). To recover 
            W in this gauge, we must extract it from the correctly gauged score.

        - Whitening tokens E by fixing sum_i E_i^T E_i = I/3. Enforces Isotropy and
            normalization conditions on average.
            
        - Set W = (W + W^T)/2 + (W - W^T) = U^T D U + K,
            for diagonal D, anti-symmetric K, and orthogonal U. We then absorb the 
            U transformation into the tokens E.

        - Rotate any remaining subspaces spanned by K, so as to move the remaining 
            components into K_12 and K_21.

    Similar transformations are performed for the position vectors p, and weight matrix M,
    but the geometric meaning of the resulting space is still obscure. Because we don't 
    necessarily expect isotropic p_i, we:

        - Extract M from the correctly normalized attn_score. 

    """
    E = E_hat.detach()
    V, _ = E.shape
    WE = WE_hat.detach()

    # S_emp = classification score
    S_emp = beta_class_hat * (E @ WE @ E.t())
    S_tilde = S_emp - S_emp.mean(dim=1, keepdim=True)

    E_centered = E - E.mean(dim=0, keepdim=True)

    E_pinv = torch.linalg.pinv(E)
    E_centered_pinv = torch.linalg.pinv(E_centered)

    W_rec = E_pinv @ S_tilde @ E_centered_pinv.t()
    # effective beta_class_hat should be rescaled too

    E_dimension = 3
    Sigma = E_dimension * (E.t() @ E) / V  ## sets average E_i to be normalized and isotropic.
    lam, U0 = torch.linalg.eigh(Sigma)
    A = (U0 * (1.0 / torch.sqrt(lam + 1e-12))) @ U0.t()

    Ainv = torch.linalg.inv(A)

    W_whiten = Ainv.t() @ W_rec @ Ainv
    S = 0.5 * (W_whiten + W_whiten.t())
    K = 0.5 * (W_whiten - W_whiten.t())

    evals, U = torch.linalg.eigh(S)
    D = torch.diag(evals)

    E_whiten = E @ A
    Etilde = E_whiten @ U
    T_E = A @ U
    Ktilde = (U.t() @ K @ U)
    Wtilde = D + Ktilde

    ## Rotating K into a single plane
    omega = torch.tensor(
        [Ktilde[2, 1], Ktilde[0, 2], Ktilde[1, 0]],
        dtype=Wtilde.dtype,
        device=Wtilde.device,
    )

    if torch.linalg.norm(omega) < 1e-12:
        R = torch.eye(3, dtype=Wtilde.dtype, device=Wtilde.device)

    else:

        e3 = omega / torch.linalg.norm(omega)
        u = torch.tensor([1.0, 0.0, 0.0], dtype=Wtilde.dtype, device=Wtilde.device)
        ## makes sure we dont choose a ref vector too aligned with e3
        if torch.abs(torch.dot(u, e3)) > 0.9: 
            u = torch.tensor([0.0, 1.0, 0.0], dtype=Wtilde.dtype, device=Wtilde.device)

        e1 = u - torch.dot(u, e3) * e3
        e1 = e1 / torch.linalg.norm(e1)
        e2 = torch.cross(e3, e1, dim=0)

        R = torch.stack([e1, e2, e3], dim=1)
        Wtilde = R.T @ Wtilde @ R
        Etilde = Etilde @ R
        T_E = T_E @ R

        if Wtilde[1, 0] < 0:
            F = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=Wtilde.dtype, device=Wtilde.device))
            Wtilde = F.T @ Wtilde @ F
            Etilde = Etilde @ F
            T_E = T_E @ F
            R = R @ F

    Wtilde_np = Wtilde.detach().cpu().numpy() / D[0, 0].abs().item()


    # Position vector analysis
    P = P_hat.detach()
    M = M_hat.detach()
    L_pos, d_pos = P.shape
    # 1) Centered design matrix X = p_j - p_bar
    P_bar = P.mean(dim=0)      # (1, d_pos)
    X = P - P_bar

    # 2) SVD: X = U S V^T  ⇒ columns of V are orthonormal directions
    #    e2 = direction of largest variance, e1 = next one.
    U_x, S_x, Vh_x = torch.linalg.svd(X, full_matrices=False)
    V_x = Vh_x.t()                           # (d_pos, d_pos)
    e2 = V_x[:, 0]                           # dominant direction
    e1 = V_x[:, 1]                           # secondary direction

    # 3) Fix sign so e2^T (p_0 - p_L) > 0
    #p0 = P[0]                                # (d_pos,)
    pL = P[-1]                               # (d_pos,)
    diff = P_bar - pL
    if torch.dot(e2, diff) < 0:
        e2 = -e2

    # 4) Build R = [e1 e2], and define scale_q, scale_p
    #    R has shape (d_pos, 2); columns are orthonormal.
    R = torch.stack([e1, e2], dim=1)        # (d_pos, 2)

    scale_p = torch.dot(e2, diff).clamp_min(1e-12)   # > 0 by construction
    scale_q = torch.dot(pL, e2)

    # 5) New coordinates: q_j = (1/scale_q) * R^T (p_j - p_L)
    #    Implemented as (P - pL) @ R, which gives rows q_j^T.
    p_shift = P - pL.unsqueeze(0)           # (L_pos, d_pos)
    ptilde = (p_shift @ R) / scale_p

    # 6) New metric: M_rec = scale_p * scale_q * R^T M R  (2x2)
    M_tilde = scale_p * scale_q * (R.t() @ M @ R)     # (2, 2)
    # queries:
    q = (P @ R) / scale_p                           # (L_pos, 2)
    q_L = q[-1]   

    # Symmetric / skew parts of the 2x2 metric, for diagnostics
    # M_sym = 0.5 * (M_tilde + M_tilde.t())
    # M_skew = 0.5 * (M_tilde - M_tilde.t())

    # Optional diagnostics (do NOT rescale ptilde further)
    ptilde_quad = torch.einsum("bi,ij,bj->b", ptilde, M_tilde, ptilde)
    ptilde_M_mean = torch.sqrt(torch.abs(ptilde_quad)).mean().clamp_min(1e-12)
    ptilde_norm_mean = torch.linalg.norm(ptilde, dim=1).mean().clamp_min(1e-12)
    
    
    
    ## original approach
    # # SM_emp = attention score
    # SM_emp = beta_attn_hat * (P @ M @ P.t())
    # SM_tilde = SM_emp - SM_emp.mean(dim=1, keepdim=True)

    # P_centered = P - P.mean(dim=0, keepdim=True)

    # P_pinv = torch.linalg.pinv(P)
    # P_centered_pinv = torch.linalg.pinv(P_centered)

    # M_rec = P_pinv @ SM_tilde @ P_centered_pinv.t()
    # M_sym = 0.5 * (M_rec + M_rec.t())
    # M_skew = 0.5 * (M_rec - M_rec.t())

    # evals_M, U_M = torch.linalg.eigh(M_sym)
    # D_M = torch.diag(evals_M)

    # sqrt_abs_D_M = torch.diag(torch.sqrt(torch.abs(evals_M)))
    # sign_diag_M = torch.sign(evals_M)
    # Sign_M = torch.diag(sign_diag_M)

    # T_pos_raw = U_M # @ sqrt_abs_D_M
    # T_pos_pinv = torch.linalg.pinv(T_pos_raw)
    # M_tilde = T_pos_pinv @ M_rec @ T_pos_pinv.t()
    # ptilde = P @ T_pos_raw
    ## end of section

    ## just uses the whitening anyway
    # M_tilde = M_rec
    # ptilde = P

    # ptilde_quad = torch.einsum("bi,ij,bj->b", ptilde, M_tilde, ptilde)
    # ptilde_norm_mean = torch.linalg.norm(ptilde, dim=1).mean().clamp_min(1e-12)
    # ptilde = ptilde / ptilde_norm_mean
    #tau_pos_rescaled = float(tau_hat * (float(ptilde_norm_mean.item()) ** 2)) #SUSPICIOUS

    ## recalculates so that it now should be 1.
    ptilde_quad = torch.einsum("bi,ij,bj->b", ptilde, M_tilde, ptilde)
    ptilde_M_mean = torch.sqrt(torch.abs(ptilde_quad)).mean().clamp_min(1e-12)
    ptilde_norm_mean = torch.linalg.norm(ptilde, dim=1).mean().clamp_min(1e-12)

    out = dict(
        W_rec=W_rec.detach().cpu().numpy(),
        W_whiten=W_whiten.detach().cpu().numpy(),
        A=A.detach().cpu().numpy(),
        Sigma=Sigma.detach().cpu().numpy(),
        S=S.detach().cpu().numpy(),
        K=K.detach().cpu().numpy(),
        U=U.detach().cpu().numpy(),
        D=D.detach().cpu().numpy(),
        Etilde=Etilde.detach().cpu().numpy(),
        ptilde=ptilde.detach().cpu().numpy(),
        T_E=T_E.detach().cpu().numpy(),
        Wtilde=Wtilde_np,
        # M_sym=M_sym.detach().cpu().numpy(),
        # M_skew=M_skew.detach().cpu().numpy(),
        M_tilde=M_tilde.detach().cpu().numpy(),
        ptilde_norm_mean=float(ptilde_norm_mean.item()),
        ptilde_M_mean=float(ptilde_M_mean.item()),
        # --- positional gauge parameters for transforming snapshots ---
        R_pos=R.detach().cpu().numpy(),          # (d_pos, 2)
        pL=pL.detach().cpu().numpy(),            # (d_pos,)
        scale_q=float(scale_q.item()),
        scale_p=float(scale_p.item()),
        q_L=q_L.detach().cpu().numpy(),
    )
    return out




def save_dashboard_mp4(
    model,
    snaps,
    *,
    out_path: Optional[str] = None,
    E_true: torch.Tensor,
    W_true: Optional[torch.Tensor] = None,
    meta: Optional[Dict] = None,
    cfg: DashboardConfig = DashboardConfig(),
    emb_transform: Optional[np.ndarray] = None,
    pos_transform: Optional[np.ndarray] = None,
) -> str:
    """
    Save the training dashboard animation as an MP4.

    The dashboard shows the learned token embedding cloud, the positional embedding,
    the learned attention matrix, and the prediction-error heatmap across saved
    training snapshots.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)

    if out_path is None:
        out_path = os.path.join(cfg.out_dir, "training_dashboard.mp4")
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # --- unpack snapshots ---
    pos_dash = snaps.pos_snapshots
    E_dash = snaps.E_snapshots
    alpha_dash = snaps.alpha_snapshots

    # Prefer probability-space error if available; fall back to logit-space error
    P_dash = getattr(snaps, "P_abs_snapshots", None)
    if P_dash is not None and len(P_dash) > 0:
        Q_dash = P_dash
        q_title = r"Prediction Error  $\frac{|P^{(target)}_{ab} - P^{(model)}_{ab}|}{P^{(target)}_{ab}}$"
        #q_title = r"Prediction Error  $|P^{(target)}_{ab} - P^{(model)}_{ab}|$"
    else:
        D_dash = snaps.D_abs_snapshots if (snaps.D_abs_snapshots is not None) else []
        Q_dash = D_dash
        q_title = r"Prediction Error  $|S^{(target)}_{ab} - S^{(model)}_{ab}|$"  # logits/scores

    steps_dash = snaps.snapshot_steps

    n_frames = min(len(pos_dash), len(E_dash), len(alpha_dash), len(steps_dash))
    if Q_dash is not None and len(Q_dash) > 0:
        n_frames = min(n_frames, len(Q_dash))

    if n_frames <= 0:
        raise ValueError("No snapshots available to animate.")

    pos_dash = pos_dash[:n_frames]
    E_dash = E_dash[:n_frames]
    alpha_dash = alpha_dash[:n_frames]
    steps_dash = steps_dash[:n_frames]
    if Q_dash is not None and len(Q_dash) > 0:
        Q_dash = Q_dash[:n_frames]


    # --- positions transform ---
    if pos_transform is None:
        T_pos = np.eye(pos_dash[0].shape[1], dtype=pos_dash[0].dtype)
    else:
        T_pos = np.asarray(pos_transform)
        d_pos = pos_dash[0].shape[1]
        if T_pos.shape != (d_pos, d_pos):
            raise ValueError(f"pos_transform must have shape {(d_pos, d_pos)}, got {T_pos.shape}")

    ptilde_dash = [p @ T_pos for p in pos_dash]
    all_ptilde = np.vstack(ptilde_dash)

    if meta is not None and "rec" in meta and meta["rec"] is not None:
        rec = meta["rec"]
        if "T_pos_raw" in rec:
            T_ref = np.asarray(rec["T_pos_raw"])
            if T_ref.shape == T_pos.shape:
                max_diff = float(np.max(np.abs(T_pos - T_ref)))
                print(f"[dashboard] max |pos_transform - rec['T_pos_raw']| = {max_diff:.3e}")

    # --- embeddings transform ---
    # If emb_transform is provided, use it directly (final A@U from `rec`).
    # Otherwise fall back to the old behavior (eigbasis of sym(WE_final)).
    if emb_transform is None:
        WE_final = model.WE.detach().cpu().numpy()
        S_E = 0.5 * (WE_final + WE_final.T)
        _, eigvecs_E = np.linalg.eigh(S_E)
        T_E = eigvecs_E
    else:
        T_E = np.asarray(emb_transform)
        dE = E_dash[0].shape[1]
        if T_E.shape != (dE, dE):
            raise ValueError(f"emb_transform must have shape {(dE, dE)}, got {T_E.shape}")

    # Subsample tokens for dashboard (match safe_backup feel)
    V_now = E_dash[0].shape[0]
    max_tokens_dashboard = 1000  # 200
    token_idx = np.linspace(0, V_now - 1, min(V_now, max_tokens_dashboard), dtype=int)

    emb_colors = (E_true[:, :3].detach().cpu().numpy() + 1) / 2
    emb_colors_sub = emb_colors[token_idx]

    # Apply the SAME final-basis transform to every snapshot (no per-frame recompute)
    Etilde_dash_sub = [(E @ T_E)[token_idx] for E in E_dash]

    # --- figure layout (tweak left panel: bigger + shifted down) ---
    fig = plt.figure(figsize=(16, 9))  # slightly bigger overall

    epoch_text = fig.text(cfg.epoch_xy[0], cfg.epoch_xy[1], "", ha="left", va="top", fontsize=12)
    stats_text = fig.text(
        cfg.stats_xy[0], cfg.stats_xy[1], "",
        ha="left", va="top",
        fontsize=11,
        family="monospace",
    )

    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[2.75, 1.0],          # make left column a bit wider
        height_ratios=[1.0, 1.0, 1.0],
        left=0.07, right=0.93,
        top=0.87, bottom=0.02,            # a touch more bottom margin
        wspace=0.22, hspace=0.55,
    )

    ax_emb = fig.add_subplot(gs[:, 0], projection="3d")
    gs_pos = gs[0, 1].subgridspec(1, 2, width_ratios=[20, 1], wspace=0.05)
    gs_alpha = gs[1, 1].subgridspec(1, 2, width_ratios=[20, 1], wspace=0.05)
    gs_Q = gs[2, 1].subgridspec(1, 2, width_ratios=[20, 1], wspace=0.05)

    ax_pos = fig.add_subplot(gs_pos[0, 0])
    cax_pos = fig.add_subplot(gs_pos[0, 1])

    ax_alpha = fig.add_subplot(gs_alpha[0, 0])
    cax_alpha = fig.add_subplot(gs_alpha[0, 1])

    ax_Q = fig.add_subplot(gs_Q[0, 0])
    cax_Q = fig.add_subplot(gs_Q[0, 1])

    # --- Nudge ONLY the right-hand column up ---
    # Top (pos) moved most, middle (alpha) a bit less, bottom (Q) least
    dy_pos   = 0.035
    dy_alpha = 0.045 #0.022
    dy_Q     = 0.04 #0.016

    # position helpers
    def _shift_up(ax, dy):
        bb = ax.get_position()
        ax.set_position([bb.x0, bb.y0 + dy, bb.width, bb.height])

    _shift_up(ax_pos,   dy_pos)
    _shift_up(cax_pos,  dy_pos)
    _shift_up(ax_alpha, dy_alpha)
    _shift_up(cax_alpha, dy_alpha)
    _shift_up(ax_Q,     dy_Q)
    _shift_up(cax_Q,    dy_Q)

    # --- titles/labels exactly like safe_backup ---
    title_fs = 16
    ax_emb.set_title("Transformer learning to predict random walks on a Sphere", pad=2, fontsize=20)

    ax_emb.text2D(
        0.5, 0.93,
        "Token Embedding",
        transform=ax_emb.transAxes,
        ha="center", va="bottom",
        fontsize=title_fs,
    )
    ax_pos.set_title("Positional Embedding", pad=8, fontsize=title_fs)
    ax_alpha.set_title("Attention Matrix", pad=8, fontsize=title_fs)
    ax_Q.set_title(q_title, pad=8, fontsize=title_fs)

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
    })
    label_fs = 13
    tick_fs = 10

    ax_pos.set_xlabel(r"$p_x$", fontsize=label_fs, fontweight="normal")
    ax_pos.set_ylabel(r"$p_y$", fontsize=label_fs, fontweight="normal")
    ax_alpha.set_xlabel("index j", fontsize=label_fs, fontweight="normal")
    ax_alpha.set_ylabel("index i", fontsize=label_fs, fontweight="normal")
    ax_Q.set_xlabel("token b", fontsize=label_fs, fontweight="normal")
    ax_Q.set_ylabel("token a", fontsize=label_fs, fontweight="normal")

    for _ax in (ax_pos, ax_alpha, ax_Q):
        _ax.tick_params(axis="both", which="major", labelsize=tick_fs)

    if ptilde_dash is not None and len(ptilde_dash) > 0:
        p_final = np.asarray(ptilde_dash[-1])
        if p_final.size > 0:
            p_lim = 1.05 * float(np.max(np.abs(p_final)))
            p_lim = max(p_lim, 1e-3)
        else:
            p_lim = 1.0
    else:
        p_lim = 1.0

    ax_pos.set_xlim(-p_lim-0.3, p_lim+0.3)
    ax_pos.set_ylim(-p_lim-0.3, p_lim+0.3)

    # Right panel axis labels
    ax_pos.set_aspect("equal", "box")
    ax_pos.set_xlabel(r"$p_x$")
    ax_pos.set_ylabel(r"$p_y$")

    # Position scatter + index colorbar (like safe_backup)
    seq_len_pos = ptilde_dash[0].shape[0]
    reds = np.linspace(0.3, 1.0, seq_len_pos)
    colors = np.array([[r, 0, 0] for r in reds])
    sc_pos = ax_pos.scatter(ptilde_dash[0][:, 0], ptilde_dash[0][:, 1], s=35, c=colors)

    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap_idx = ListedColormap(colors, name="p_index_reds_exact")
    boundaries = np.arange(-0.5, seq_len_pos + 0.5, 1.0)
    norm_idx = BoundaryNorm(boundaries, cmap_idx.N)
    sm_idx = plt.cm.ScalarMappable(cmap=cmap_idx, norm=norm_idx)
    sm_idx.set_array([])
    cb_pos = fig.colorbar(sm_idx, cax=cax_pos)
    cb_pos.set_label("Position index i  (early → late)")
    cb_pos.set_ticks(list(range(seq_len_pos)))
    cb_pos.set_ticklabels([fr"$p_{{{t}}}$" for t in range(seq_len_pos)])
    cb_pos.ax.tick_params(labelsize=7, pad=1)
    cb_pos.ax.set_ylim(-0.5, seq_len_pos - 0.5)

    # Light arrows
    arrow_patches = []
    pts0 = ptilde_dash[0]
    for k in range(seq_len_pos - 1):
        a = ax_pos.annotate(
            "",
            xy=(pts0[k + 1, 0], pts0[k + 1, 1]),
            xytext=(pts0[k, 0], pts0[k, 1]),
            arrowprops=dict(
                arrowstyle="->",
                color=(0, 0, 0, 0.18),
                lw=0.8,
                mutation_scale=8,
                shrinkA=0.0,
                shrinkB=0.0,
            ),
            zorder=1,
        )
        arrow_patches.append(a)

    # Embeddings scatter
    E0 = Etilde_dash_sub[0]
    sc_emb = ax_emb.scatter(E0[:, 0], E0[:, 1], E0[:, 2], s=34, c=emb_colors_sub, depthshade=True)

    # (REMOVE these lines if present; ax_rollout/emb_label_fs do not belong in training dashboard)
    # emb_label_fs = 15
    # ax_rollout.set_xlabel("x", fontsize=emb_label_fs, labelpad=8)
    # ax_rollout.set_ylabel("y", fontsize=emb_label_fs, labelpad=8)
    # ax_rollout.set_zlabel("z", fontsize=emb_label_fs, labelpad=8)

    ## Fixes Camera
    ax_emb.view_init(elev=cfg.elev, azim=cfg.azim + 0.6)

    ax_emb.set_xlim(-cfg.emb_lim, cfg.emb_lim)
    ax_emb.set_ylim(-cfg.emb_lim, cfg.emb_lim)
    ax_emb.set_zlim(-cfg.emb_lim, cfg.emb_lim)
    ax_emb.set_box_aspect((1, 1, 1))

    # Shift the 3D axis down a bit within its grid cell (relative positioning)
    bbox = ax_emb.get_position()
    ax_emb.set_position([bbox.x0, bbox.y0 - 0.015, bbox.width, bbox.height + 0.015])

    # Alpha heatmap
    im_alpha = ax_alpha.imshow(alpha_dash[0], vmin=0, vmax=1, cmap="hot", interpolation="nearest")
    cb_alpha = fig.colorbar(im_alpha, cax=cax_alpha)
    cb_alpha.ax.tick_params(labelsize=8)

    # Q-error heatmap (if available)
    if Q_dash is not None and len(Q_dash) > 0:
        im_Q = ax_Q.imshow(Q_dash[0], cmap="hot", vmin=cfg.q_vmin, vmax=cfg.q_vmax, interpolation="nearest")
        cb_Q = fig.colorbar(im_Q, cax=cax_Q)
        cb_Q.ax.tick_params(labelsize=8)
    else:
        im_Q = ax_Q.imshow(np.zeros((model.V, model.V)), cmap="hot", vmin=cfg.q_vmin, vmax=cfg.q_vmax)
        cb_Q = fig.colorbar(im_Q, cax=cax_Q)
        cb_Q.ax.tick_params(labelsize=8)

    # Text content sources
    train_loss = meta.get("final_train_loss", None) if meta else None
    test_loss = meta.get("final_test_loss", None) if meta else None
    floor = meta.get("theoretical_entropy_floor", None) if meta else None
    steps_per_epoch = meta.get("steps_per_epoch", None) if meta else None

    def update(frame: int):
        # embeddings
        Ecur = Etilde_dash_sub[frame]
        sc_emb._offsets3d = (Ecur[:, 0], Ecur[:, 1], Ecur[:, 2])
        ax_emb.view_init(elev=cfg.elev, azim=cfg.azim + 0.6 * frame)

        # positions
        pts = ptilde_dash[frame]
        sc_pos.set_offsets(pts)
        for k, a in enumerate(arrow_patches):
            a.xy = (pts[k + 1, 0], pts[k + 1, 1])
            a.set_position((pts[k, 0], pts[k, 1]))

        # alpha
        im_alpha.set_data(alpha_dash[frame])

        # Q error
        if Q_dash is not None and len(Q_dash) > 0:
            im_Q.set_data(Q_dash[frame])
            im_Q.set_clim(cfg.q_vmin, cfg.q_vmax)

        # epoch text (use actual snapshot time)
        if steps_per_epoch is not None and steps_per_epoch > 0:
            epoch_est = steps_dash[frame] / float(steps_per_epoch)
            epoch_text.set_text(f"epoch {epoch_est:.2f}")
        else:
            epoch_text.set_text(f"step {int(steps_dash[frame])}")

        # stats text (labels left, numbers right)
        label_w = 18
        lines = []
        if train_loss is not None:
            lines.append(f"{'Training loss':<{label_w}} {float(train_loss):>10.3f}")
        if test_loss is not None:
            lines.append(f"{'Test loss':<{label_w}} {float(test_loss):>10.3f}")
        if floor is not None:
            lines.append(f"{'Theoretical floor':<{label_w}} {float(floor):>10.3f}")
        stats_text.set_text("\n".join(lines))

        return None

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=cfg.interval_ms, blit=False)

    fps = int(round(1000.0 / cfg.interval_ms))
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=1800,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    anim.save(out_path, writer=writer, dpi=cfg.dpi)
    plt.close(fig)
    return out_path


def _front_mask(xyz: np.ndarray, elev_deg: float, azim_deg: float) -> np.ndarray:
    """
    Return a boolean mask indicating which 3D points lie on the front-facing side
    of the current camera view.
    """
    elev_r = np.deg2rad(elev_deg)
    azim_r = np.deg2rad(azim_deg)
    view_dir = np.array([
        np.cos(elev_r) * np.cos(azim_r),
        np.cos(elev_r) * np.sin(azim_r),
        np.sin(elev_r),
    ], dtype=np.float32)
    depth = xyz @ view_dir
    return depth >= 0.0


def normalization_constant(a, beta=1.0, num_points=10000):
    """
    Numerically compute the normalization constant for the theoretical z-density
    used in the rollout density comparison.
    """
    z = np.linspace(-1.0, 1.0, num_points)
    radius = np.sqrt(1 + a**2 * (1 - z**2))
    integrand = np.sinh(beta * radius) / (beta * radius)
    return np.trapz(integrand, z)

def p_z(z, a, beta=1.0, norm_const=None):
    """
    Evaluate the theoretical stationary density p(z) for the rollout comparison.

    If no normalization constant is provided, one is computed numerically.
    """
    z = np.asarray(z)
    radius = np.sqrt(1 + a**2 * (1 - z**2))
    numerator = np.sinh(beta * radius) / (beta * radius)

    if norm_const is None:
        norm_const = normalization_constant(a, beta=beta)

    return numerator / norm_const

# def plot_stationary_pz(
#     a: float,
#     *,
#     beta: float = 1.0,
#     out_path: Optional[str] = None,
#     num_points: int = 1000,
# ) -> str:
#     """
#     Plot the theoretical stationary z-density and save it as a PNG.
#     """
#     z = np.linspace(-1.0, 1.0, num_points)
#     p = p_z(z, a, beta=beta)

#     fig, ax = plt.subplots(figsize=(6.2, 4.2))
#     ax.plot(z, p, color="navy", lw=2.2)
#     ax.set_xlabel("z = cos(theta)")
#     ax.set_ylabel("p(z)")
#     ax.set_title(f"Stationary density p(z), a={a:.3f}, beta={beta:.3f}")
#     ax.grid(alpha=0.25)

#     if out_path is None:
#         out_path = f"pz_a_{a:.3f}_beta_{beta:.3f}.png"

#     fig.tight_layout()
#     fig.savefig(out_path, dpi=160, bbox_inches="tight")
#     plt.close(fig)
#     return out_path


@torch.no_grad()
def save_rollout_dashboard_mp4(
    model,
    seed_batch_rollout: torch.Tensor,
    seed_batch_density: torch.Tensor,
    *,
    out_path: str,
    a: float,
    beta: float,
    n_steps: int = 500,
    temperature: float = 1.0,
    sample: bool = True,
    use_soft_point: bool = False,
    rollout_batch_size: int = 20,
    density_batch_size: int = 500,
    elev: float = 16,
    azim: float = 35,
    lim: float = 1.6,
    interval_ms: int = 120,
    dpi: int = 120,
    density_bins: int = 16,
    emb_transform: Optional[np.ndarray] = None,
    n_z_bins: int = 40,
    n_phi_bins: int = 40,
) -> str:
    """
    Save a rollout dashboard animation as an MP4.

    The layout combines rollout trajectories on the left with cumulative empirical
    density comparisons on the right: z-density on the top and azimuthal density on
    the bottom, all updated from synchronized model rollouts.
    """
    model.eval()
    device = next(model.parameters()).device

    E = model.E.weight.detach().to(device)
    dE = int(E.shape[1])
    if dE < 3:
        raise ValueError(f"Need dE>=3 for 3D plot. Got dE={dE}")

    if emb_transform is not None:
        T = torch.as_tensor(emb_transform, dtype=E.dtype, device=device)
        if T.shape != (dE, dE):
            raise ValueError(f"emb_transform must have shape {(dE, dE)}, got {tuple(T.shape)}")
        E_plot = E @ T
    else:
        E_plot = E

    E3 = E_plot[:, :3]
    E3_cpu = E3.detach().cpu().numpy()

    z_tokens = E3[:, 2].detach().cpu().numpy()
    phi_tokens = torch.atan2(E3[:, 1], E3[:, 0]).detach().cpu().numpy()

    if seed_batch_rollout.dim() != 2:
        raise ValueError("seed_batch_rollout must have shape (B,T)")
    if seed_batch_density.dim() != 2:
        raise ValueError("seed_batch_density must have shape (B,T)")

    ctx_rollout = seed_batch_rollout.long().to(device).clone()
    ctx_density = seed_batch_density.long().to(device).clone()

    rollout_batch_size = int(ctx_rollout.shape[0])
    density_batch_size = int(ctx_density.shape[0])

    L_pos = int(getattr(model, "L_pos"))
    if ctx_rollout.shape[1] < L_pos:
        raise ValueError(f"Rollout seed length {ctx_rollout.shape[1]} < model.L_pos={L_pos}")
    if ctx_density.shape[1] < L_pos:
        raise ValueError(f"Density seed length {ctx_density.shape[1]} < model.L_pos={L_pos}")

    ctx_tail = 4
    blank_tok = 0

    palette_name = "blue"
    palettes = {
        "classic": {
            "cloud_color": "gray",
            "tail_color": "red",
            "pred_lo": np.array([0.55, 0.00, 0.00]),
            "pred_hi": np.array([1.00, 0.00, 0.00]),
        },
        "blue": {
            "cloud_color": "gray",
            "tail_color": "#4C72B0",
            "pred_lo": np.array([0.18, 0.35, 0.62]),
            "pred_hi": np.array([0.42, 0.60, 0.90]),
        },
    }
    palette = palettes[palette_name]

    init_xyz_cpu = E3[ctx_rollout[:, -1]].detach().cpu().numpy()

    max_abs = float(np.max(np.abs(E3_cpu)))
    lim_use = max(lim, 1.05 * max_abs)

    z_edges = np.linspace(-1.0, 1.0, n_z_bins + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    dz = z_edges[1] - z_edges[0]
    p_theory_z = p_z(z_centers, a=a, beta=beta)

    token_hist_z, _ = np.histogram(z_tokens, bins=z_edges)
    token_density_z = token_hist_z.astype(np.float64) / max(len(z_tokens) * dz, 1e-12)
    token_density_z_safe = np.maximum(token_density_z, 1e-12)

    phi_edges = np.linspace(-np.pi, np.pi, n_phi_bins + 1)
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    dphi = phi_edges[1] - phi_edges[0]
    p_theory_phi = np.full_like(phi_centers, 1.0 / (2.0 * np.pi), dtype=np.float64)

    token_hist_phi, _ = np.histogram(phi_tokens, bins=phi_edges)
    token_density_phi = token_hist_phi.astype(np.float64) / max(len(phi_tokens) * dphi, 1e-12)
    token_density_phi_safe = np.maximum(token_density_phi, 1e-12)

    cum_counts_z = np.zeros(n_z_bins, dtype=np.float64)
    total_count_z = 0
    cum_counts_phi = np.zeros(n_phi_bins, dtype=np.float64)
    total_count_phi = 0

    # --- Layout: match training dashboard feel (smaller panels, more margins) ---
    fig = plt.figure(figsize=(16, 9))

    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[2.75, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.07,
        right=0.93,
        top=0.87,
        bottom=0.06,
        wspace=0.22,
        hspace=0.55,
    )

    ax_rollout = fig.add_subplot(gs[:, 0], projection="3d")
    ax_density = fig.add_subplot(gs[0, 1])
    #ax_phi = fig.add_subplot(gs[1, 1])
    ax_phi = fig.add_subplot(gs[1, 1])

    # Move the bottom-right (azimuthal) panel slightly upward
    bbox_phi = ax_phi.get_position()
    ax_phi.set_position([bbox_phi.x0, bbox_phi.y0 + 0.018, bbox_phi.width, bbox_phi.height])

    # Keep RHS reasonably square-ish without forcing too small
    try:
        ax_density.set_box_aspect(1.0)
        ax_phi.set_box_aspect(1.0)
    except Exception:
        pass

    # --- Typography: align more closely with training dashboard ---
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
    })

    label_fs = 13
    tick_fs = 10
    title_fs = 16

    ax_rollout.set_title("Transformer predicting random walks on a Sphere", pad=2, fontsize=20)
    rollout_title = ax_rollout.text2D(   # <--- add the assignment
        0.55, 0.96,   # a bit to the right, near the top of the 3D panel
        "",
        transform=ax_rollout.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        family="monospace",
    )

    # ax_density.set_title("Cumulative axial density", pad=8, fontsize=title_fs)
    # ax_phi.set_title("Cumulative azimuthal density", pad=8, fontsize=title_fs)
    ax_density.set_title("Axial density vs equilibrium", pad=8, fontsize=title_fs)
    ax_phi.set_title("Azimuthal density vs equilibrium", pad=8, fontsize=title_fs)

    ax_rollout.set_xlabel("x", fontsize=label_fs, labelpad=8)
    ax_rollout.set_ylabel("y", fontsize=label_fs, labelpad=8)
    ax_rollout.set_zlabel("z", fontsize=label_fs, labelpad=8)

    ax_density.set_xlabel("z = cos(theta)", fontsize=label_fs)
    ax_density.set_ylabel("Cumulative density", fontsize=label_fs)
    ax_phi.set_xlabel(r"$\phi$", fontsize=label_fs)
    ax_phi.set_ylabel("Cumulative density", fontsize=label_fs)

    for _ax in (ax_density, ax_phi):
        _ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    ax_rollout.tick_params(axis="both", which="major", labelsize=tick_fs)

    ax_rollout.set_xlim(-lim_use, lim_use)
    ax_rollout.set_ylim(-lim_use, lim_use)
    ax_rollout.set_zlim(-lim_use, lim_use)
    ax_rollout.set_box_aspect((1, 1, 1))
    ax_rollout.view_init(elev=elev, azim=azim)

    # Shift the 3D axis down a bit within its grid cell (relative positioning)
    bbox = ax_rollout.get_position()
    ax_rollout.set_position([bbox.x0, bbox.y0 - 0.015, bbox.width, bbox.height + 0.015])

    token_size = 10
    init_front = np.ones(init_xyz_cpu.shape[0], dtype=bool)

    pred_sc_back = ax_rollout.scatter(
        init_xyz_cpu[~init_front, 0], init_xyz_cpu[~init_front, 1], init_xyz_cpu[~init_front, 2],
        s=token_size + 4,
        c="red",
        alpha=0.12,
        depthshade=False,
        edgecolors="none",
        linewidths=0.0,
    )

    all_sc = ax_rollout.scatter(
        E3[:, 0], E3[:, 1], E3[:, 2],
        s=token_size,
        c=palette["cloud_color"],
        alpha=0.10,
        depthshade=False,
    )

    pred_sc_front = ax_rollout.scatter(
        init_xyz_cpu[init_front, 0], init_xyz_cpu[init_front, 1], init_xyz_cpu[init_front, 2],
        s=token_size + 6,
        c="red",
        alpha=0.95,
        depthshade=False,
        edgecolors="black",
        linewidths=0.4,
    )

    mix = np.linspace(0.0, 1.0, ctx_rollout.shape[0])[:, None]
    pred_rgb = (1.0 - mix) * palette["pred_lo"][None, :] + mix * palette["pred_hi"][None, :]
    pred_colors = np.concatenate(
        [pred_rgb, np.full((ctx_rollout.shape[0], 1), 0.78)],
        axis=1,
    )

    ctx_glow_back = [None] * rollout_batch_size
    ctx_glow_mid = [None] * rollout_batch_size
    ctx_glow_head = [None] * rollout_batch_size
    ctx_line = [None] * rollout_batch_size

    # -----------------------
    # Z density (top-right)
    # -----------------------
    # bar_density = ax_density.bar(
    #     z_centers,
    #     np.zeros_like(z_centers),
    #     width=0.92 * dz,
    #     alpha=0.35,
    #     color="tab:blue",
    #     label="Cumulative density",
    # )
    bar_density = ax_density.bar(
    z_edges[:-1],                     # left edges of bins
    np.zeros_like(z_centers),
    width=0.92 * dz,                         # full bin width
    align="edge",                     # bars span [edge, edge+dz]
    alpha=0.35,
    color="tab:blue",
    label="Cumulative density",
    )
    line_density_theory, = ax_density.plot(
        z_centers,
        p_theory_z,
        color="crimson",
        lw=2.0,
        alpha=0.85,
        label="Theoretical equilibrium",
    )

    ax_density.set_xlim(-1.0, 1.0)
    ax_density.set_ylim(0.0, max(1.05 * float(np.max(p_theory_z)), 1.0))
    ax_density.set_xlabel("z = cos(theta)")
    ax_density.set_ylabel("density")
    ax_density.grid(alpha=0.25)

    # Legend inside axes
    leg_density = ax_density.legend(
        loc="upper right",
        frameon=True,
        fontsize=9,
        borderaxespad=0.6,
        handlelength=1.8,
        labelspacing=0.35,
    )

    # "Area Difference" text just underneath the legend, still inside the axes
    area_text_z = ax_density.text(
        0.98,
        0.78,  # positioned below the legend region; adjust later if needed
        "Summed Difference: --",
        transform=ax_density.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    # -----------------------
    # Phi density (bottom-right)
    # -----------------------
    bar_phi = ax_phi.bar(
        phi_centers,
        np.zeros_like(phi_centers),
        width=0.92 * dphi,
        alpha=0.35,
        color="tab:purple",
        label="Cumulative density",
    )
    line_phi_theory, = ax_phi.plot(
        phi_centers,
        p_theory_phi,
        color="crimson",
        lw=2.0,
        alpha=0.85,
        label="Theoretical equilibrium",
    )

    ax_phi.set_xlim(-np.pi, np.pi)
    ax_phi.set_ylim(0.0, max(1.05 * float(np.max(p_theory_phi)), 1.0))
    #ax_phi.set_xlabel(r"$\phi$")
    ax_phi.set_xlabel(r"$\phi=\mathrm{atan2}(y,x)$", fontsize=label_fs)
    ax_phi.set_ylabel("density")
    ax_phi.grid(alpha=0.25)

    leg_phi = ax_phi.legend(
        loc="upper right",
        frameon=True,
        fontsize=9,
        borderaxespad=0.6,
        handlelength=1.8,
        labelspacing=0.35,
    )

    area_text_phi = ax_phi.text(
        0.98,
        0.78,
        "Summed Difference: --",
        transform=ax_phi.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    def _densify_path(xyz: np.ndarray, points_per_seg: int = 8) -> np.ndarray:
        if xyz.shape[0] <= 1:
            return xyz
        pieces = []
        for i in range(xyz.shape[0] - 1):
            a0 = xyz[i]
            b0 = xyz[i + 1]
            ts = np.linspace(0.0, 1.0, points_per_seg, endpoint=False)[:, None]
            pieces.append((1.0 - ts) * a0 + ts * b0)
        pieces.append(xyz[-1:])
        return np.vstack(pieces)

    def update(t: int):
        nonlocal ctx_rollout, ctx_density
        nonlocal ctx_glow_back, ctx_glow_mid, ctx_glow_head, ctx_line
        nonlocal cum_counts_z, total_count_z, cum_counts_phi, total_count_phi

        ax_rollout.view_init(elev=elev, azim=azim + 0.6 * t)

        out_rollout = model(ctx_rollout)
        logits_rollout = out_rollout[:, -1, :]
        if temperature != 1.0:
            logits_rollout = logits_rollout / float(temperature)
        p_rollout = torch.softmax(logits_rollout, dim=-1)
        nxt_rollout = torch.multinomial(p_rollout, 1) if sample else torch.argmax(logits_rollout, dim=-1, keepdim=True)
        ctx_rollout[:, -1] = nxt_rollout.squeeze(1)

        if use_soft_point:
            e_pred = (p_rollout @ E3).detach().cpu().numpy()
        else:
            e_pred = E3[nxt_rollout.squeeze(1)].detach().cpu().numpy()

        cur_azim = azim + 0.6 * t
        front = _front_mask(e_pred, elev, cur_azim)
        back = ~front

        pred_sc_back._offsets3d = (e_pred[back, 0], e_pred[back, 1], e_pred[back, 2])
        pred_sc_front._offsets3d = (e_pred[front, 0], e_pred[front, 1], e_pred[front, 2])

        pred_sc_back.set_sizes(np.full(int(back.sum()), token_size + 4))
        pred_sc_front.set_sizes(np.full(int(front.sum()), token_size + 6))

        back_colors = pred_colors[back].copy()
        back_colors[:, 3] = 0.10
        pred_sc_back.set_facecolors(back_colors)
        pred_sc_back.set_edgecolors(back_colors)

        front_colors = pred_colors[front].copy()
        front_colors[:, 3] = 0.95
        pred_sc_front.set_facecolors(front_colors)

        front_edges = front_colors.copy()
        front_edges[:, :3] = 0.0
        front_edges[:, 3] = 0.45
        pred_sc_front.set_edgecolors(front_edges)


        for b in range(ctx_rollout.shape[0]):
            ctx_xyz = E3[ctx_rollout[b, -ctx_tail:]].detach().cpu().numpy()
            ctx_xyz_dense = _densify_path(ctx_xyz, points_per_seg=10)

            n = len(ctx_xyz_dense)
            i_mid = max(0, int(0.8 * n))
            i_head = max(0, int(0.9 * n))

            if ctx_glow_back[b] is not None:
                ctx_glow_back[b].remove()
            if ctx_glow_mid[b] is not None:
                ctx_glow_mid[b].remove()
            if ctx_glow_head[b] is not None:
                ctx_glow_head[b].remove()
            if ctx_line[b] is not None:
                ctx_line[b].remove()

            ctx_glow_back[b], = ax_rollout.plot(
                ctx_xyz_dense[:, 0], ctx_xyz_dense[:, 1], ctx_xyz_dense[:, 2],
                color=palette["tail_color"], alpha=0.04, lw=9,
                solid_capstyle="round", solid_joinstyle="round",
            )
            ctx_glow_mid[b], = ax_rollout.plot(
                ctx_xyz_dense[i_mid:, 0], ctx_xyz_dense[i_mid:, 1], ctx_xyz_dense[i_mid:, 2],
                color=palette["tail_color"], alpha=0.10, lw=5,
                solid_capstyle="round", solid_joinstyle="round",
            )
            ctx_glow_head[b], = ax_rollout.plot(
                ctx_xyz_dense[i_head:, 0], ctx_xyz_dense[i_head:, 1], ctx_xyz_dense[i_head:, 2],
                color=palette["tail_color"], alpha=0.30, lw=3.0,
                solid_capstyle="round", solid_joinstyle="round",
            )
            ctx_line[b], = ax_rollout.plot(
                ctx_xyz_dense[:, 0], ctx_xyz_dense[:, 1], ctx_xyz_dense[:, 2],
                color=palette["tail_color"], alpha=0.18, lw=1.6,
                solid_capstyle="round", solid_joinstyle="round",
            )

        out_density = model(ctx_density)
        logits_density = out_density[:, -1, :]
        if temperature != 1.0:
            logits_density = logits_density / float(temperature)
        probs_density = torch.softmax(logits_density, dim=-1)
        nxt_density = torch.multinomial(probs_density, 1) if sample else torch.argmax(logits_density, dim=-1, keepdim=True)
        tok_np = nxt_density.squeeze(1).detach().cpu().numpy()

        z_now = z_tokens[tok_np]
        hist_z, _ = np.histogram(z_now, bins=z_edges)
        cum_counts_z += hist_z
        total_count_z += z_now.shape[0]

        p_emp_z = cum_counts_z / max(total_count_z * dz, 1e-12)
        p_emp_z_debias = p_emp_z / token_density_z_safe
        p_emp_z_debias = p_emp_z_debias / max(np.sum(p_emp_z_debias) * dz, 1e-12)

        for rect, h in zip(bar_density, p_emp_z_debias):
            rect.set_height(h)

        l1_z = np.sum(np.abs(p_emp_z_debias - p_theory_z)) * dz
        area_text_z.set_text(f"Integrated Difference: {l1_z:.4f}")

        # --- phi density update (debiased cumulative density) ---
        phi_now = phi_tokens[tok_np]

        hist_phi, _ = np.histogram(phi_now, bins=phi_edges)
        cum_counts_phi += hist_phi
        total_count_phi += phi_now.shape[0]

        p_emp_phi = cum_counts_phi / max(total_count_phi * dphi, 1e-12)
        p_emp_phi_debias = p_emp_phi / token_density_phi_safe
        p_emp_phi_debias = p_emp_phi_debias / max(np.sum(p_emp_phi_debias) * dphi, 1e-12)

        for rect, h in zip(bar_phi, p_emp_phi_debias):
            rect.set_height(h)

        l1_phi = np.sum(np.abs(p_emp_phi_debias - p_theory_phi)) * dphi
        area_text_phi.set_text(f"Integrated Difference: {l1_phi:.4f}")

        blank_rollout = torch.full((ctx_rollout.shape[0], 1), blank_tok, dtype=ctx_rollout.dtype, device=ctx_rollout.device)
        ctx_rollout = torch.cat([ctx_rollout[:, 1:], blank_rollout], dim=1)

        blank_density = torch.full((ctx_density.shape[0], 1), blank_tok, dtype=ctx_density.dtype, device=ctx_density.device)
        ctx_density[:, -1] = nxt_density.squeeze(1)
        ctx_density = torch.cat([ctx_density[:, 1:], blank_density], dim=1)

        rollout_title.set_text(
            "Iteration step :  {t}\n"
            "Traj. displayed : {b_disp}\n"
            "Traj. total: {b_den}".format(
                t=int(t),
                b_disp=int(rollout_batch_size),
                b_den=int(density_batch_size),
            )
        )

        artists = [
            pred_sc_back, all_sc, pred_sc_front,
            line_density_theory, line_phi_theory,
            rollout_title,
            area_text_z, area_text_phi,
        ]
        artists.extend(list(bar_density))
        artists.extend(list(bar_phi))
        artists.extend([a for a in ctx_glow_back if a is not None])
        artists.extend([a for a in ctx_glow_mid if a is not None])
        artists.extend([a for a in ctx_glow_head if a is not None])
        artists.extend([a for a in ctx_line if a is not None])

        # legends are static; no need to return them for updates
        return tuple(artists)

    anim = animation.FuncAnimation(fig, update, frames=int(n_steps), interval=interval_ms, blit=False)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fps = max(1, int(round(1000.0 / interval_ms)))
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=1800,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path


@torch.no_grad()
def save_embedding_training_mp4(
    model,
    snaps,
    *,
    out_path: str,
    E_true: Optional[torch.Tensor] = None,
    emb_transform: Optional[np.ndarray] = None,
    blocked_tokens: Optional[np.ndarray] = None,
    elev: float = 16.0,
    azim: float = 35.0,
    lim: Optional[float] = None,
    interval_ms: int = 80,
    dpi: int = 120,
    max_tokens: int = 1000,
) -> str:
    """
    Save a simple training animation that only shows the token embeddings evolving.

    Uses snaps.E_snapshots (list of [V, dE] arrays) and an optional emb_transform
    (e.g. rec["T_E"]) to fix the embedding gauge. No positional / heatmaps, just
    a 3D scatter of tokens across training steps.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    E_dash = snaps.E_snapshots
    steps_dash = snaps.snapshot_steps
    if len(E_dash) == 0:
        raise ValueError("snaps.E_snapshots is empty; nothing to animate.")

    n_frames = len(E_dash)

    # --- choose embedding transform (same convention as dashboards) ---
    E0 = E_dash[0]
    V_now, dE = E0.shape
    if dE < 3:
        raise ValueError(f"Need dE>=3 for 3D embedding plot. Got dE={dE}")

    if emb_transform is None:
        WE_final = model.WE.detach().cpu().numpy()
        S_E = 0.5 * (WE_final + WE_final.T)
        _, eigvecs_E = np.linalg.eigh(S_E)
        T_E = eigvecs_E
    else:
        T_E = np.asarray(emb_transform)
        if T_E.shape != (dE, dE):
            raise ValueError(f"emb_transform must have shape {(dE, dE)}, got {T_E.shape}")

    # --- blocked token handling ---
    if blocked_tokens is not None:
        blocked_tokens = np.asarray(blocked_tokens, dtype=int).ravel()
        blocked_tokens = blocked_tokens[(blocked_tokens >= 0) & (blocked_tokens < V_now)]
        blocked_tokens = np.unique(blocked_tokens)
    else:
        blocked_tokens = np.array([], dtype=int)

    # --- choose which tokens to plot (always include blocked) ---
    if V_now <= max_tokens:
        token_idx = np.arange(V_now, dtype=int)
    else:
        # start with blocked, then fill the rest
        remaining = np.setdiff1d(np.arange(V_now, dtype=int), blocked_tokens, assume_unique=True)
        n_needed = max_tokens - blocked_tokens.size
        if n_needed > 0:
            # roughly uniform subsample of the remaining indices
            pick = np.linspace(0, remaining.size - 1, n_needed, dtype=int)
            extra = remaining[pick]
            token_idx = np.concatenate([blocked_tokens, extra])
        else:
            token_idx = blocked_tokens[:max_tokens]
    token_idx = np.unique(token_idx)

    # --- colors: dark gray → neutral blue gradient for active tokens, blocked in red (RGB only) ---
    if E_true is not None:
        E_true_np = E_true.detach().cpu().numpy()
        rgb_true = (E_true_np[:, :3] + 1.0) / 2.0

        lum = 0.299 * rgb_true[:, 0] + 0.587 * rgb_true[:, 1] + 0.114 * rgb_true[:, 2]
        lum_min, lum_max = np.min(lum), np.max(lum)
        if lum_max > lum_min:
            lum = (lum - lum_min) / (lum_max - lum_min)
        # lum in [0,1]; blend from dark gray (0) to neutral blue (1)
        dark_gray = np.array([0.25, 0.25, 0.25], dtype=float)
        blue = np.array([0.18, 0.35, 0.62], dtype=float)   # neutral blue
        base_rgb_all = (1.0 - lum)[:, None] * dark_gray[None, :] + lum[:, None] * blue[None, :]
        colors_sub = base_rgb_all[token_idx]
    else:
        # fallback: fixed neutral blue
        base_rgb = np.array([0.42, 0.60, 0.90], dtype=float)
        colors_sub = np.tile(base_rgb[None, :], (token_idx.size, 1))

    # blocked tokens mask in the subsampled set
    if blocked_tokens.size > 0:
        is_blocked_full = np.zeros(V_now, dtype=bool)
        is_blocked_full[blocked_tokens] = True
        is_blocked_sub = is_blocked_full[token_idx]
        # blocked: red in RGB; opacity will come from scatter(alpha=...)
        colors_sub[is_blocked_sub] = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        is_blocked_sub = np.zeros(token_idx.size, dtype=bool)

    # apply gauge transform per snapshot lazily in update
    def _embed_frame(i: int) -> np.ndarray:
        Ei = np.asarray(E_dash[i])
        E_plot = Ei @ T_E
        return E_plot[token_idx, :3]

    E_plot0 = _embed_frame(0)

    # determine limits: match training dashboard scaling
    if lim is None:
        # use the same default emb_lim as DashboardConfig
        lim_use = DashboardConfig().emb_lim   # currently 1.4
    else:
        lim_use = float(lim)

    # --- figure and initial scatters ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
    })

    ax.set_title("Token Embeddings During Training", fontsize=16, pad=6)

    ax.set_xlim(-lim_use, lim_use)
    ax.set_ylim(-lim_use, lim_use)
    ax.set_zlim(-lim_use, lim_use)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_zlabel("z", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=9)

    ax.view_init(elev=elev, azim=azim)

    # split into back / dead / front for frame 0
    from .analysis import _front_mask  # if needed, or just use directly since defined above

    front0 = _front_mask(E_plot0, elev, azim)
    active_mask_sub = ~is_blocked_sub

    back_mask0 = active_mask_sub & ~front0
    front_mask0 = active_mask_sub & front0
    dead_mask0 = is_blocked_sub

    # back half sphere (active)
    sc_back = ax.scatter(
        E_plot0[back_mask0, 0], E_plot0[back_mask0, 1], E_plot0[back_mask0, 2],
        s=22,
        c=colors_sub[back_mask0],
        depthshade=False,
        alpha=0.35,
        edgecolors="none",
    )

    # dead tokens (blocked) in opaque red
    sc_dead = ax.scatter(
        E_plot0[dead_mask0, 0], E_plot0[dead_mask0, 1], E_plot0[dead_mask0, 2],
        s=30,
        c=colors_sub[dead_mask0],
        depthshade=False,
        alpha=0.95,
        edgecolors="none",
    )

    # front half sphere (active)
    sc_front = ax.scatter(
        E_plot0[front_mask0, 0], E_plot0[front_mask0, 1], E_plot0[front_mask0, 2],
        s=22,
        c=colors_sub[front_mask0],
        depthshade=False,
        alpha=0.65,
        edgecolors="none",
    )

    # --- legend: gray-blue = active tokens, red = dead tokens ---
    legend_active = ax.scatter(
        [], [], [],
        s=30,
        c=np.array([[0.3, 0.4, 0.6]]),   # representative gray-blue
        alpha=0.7,
        label="active tokens (gray-blue)",
    )
    legend_dead = ax.scatter(
        [], [], [],
        s=30,
        c=np.array([[1.0, 0.0, 0.0]]),   # red
        alpha=0.95,
        label="dead tokens (red)",
    )
    ax.legend(
        handles=[legend_active, legend_dead],
        loc="upper right",
        fontsize=9,
        frameon=True,
    )

    step_text = fig.text(
        0.02, 0.96,
        "",
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    def update(frame: int):
        E_plot = _embed_frame(frame)

        # rotate camera
        cur_azim = azim + 1.8 * frame
        ax.view_init(elev=elev, azim=cur_azim)

        # recompute front/back for current view
        front = _front_mask(E_plot, elev, cur_azim)
        back = ~front
        active_mask = ~is_blocked_sub

        back_mask = active_mask & back
        front_mask = active_mask & front
        dead_mask = is_blocked_sub

        # update offsets and colors for each group, in desired draw order:
        # back half -> dead tokens -> front half
        sc_back._offsets3d = (
            E_plot[back_mask, 0],
            E_plot[back_mask, 1],
            E_plot[back_mask, 2],
        )
        sc_back.set_facecolors(colors_sub[back_mask])

        sc_dead._offsets3d = (
            E_plot[dead_mask, 0],
            E_plot[dead_mask, 1],
            E_plot[dead_mask, 2],
        )
        sc_dead.set_facecolors(colors_sub[dead_mask])

        sc_front._offsets3d = (
            E_plot[front_mask, 0],
            E_plot[front_mask, 1],
            E_plot[front_mask, 2],
        )
        sc_front.set_facecolors(colors_sub[front_mask])

        if steps_dash is not None and len(steps_dash) == n_frames:
            step = int(steps_dash[frame])
            step_text.set_text(f"step: {step}")
        else:
            step_text.set_text(f"frame: {frame}")

        return sc_back, sc_dead, sc_front, step_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval_ms,
        blit=False,
    )

    fps = max(1, int(round(1000.0 / interval_ms)))
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=1800,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path