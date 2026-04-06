import os
from dataclasses import dataclass, field

import numpy as np
import torch
import warnings
import matplotlib.pyplot as plt 

warnings.filterwarnings(
    "ignore",
    message=r".*urllib3 v2 only supports OpenSSL 1\.1\.1\+.*LibreSSL 2\.8\.3.*",
)

from source_code.analysis import (
    save_dashboard_mp4,
    save_rollout_dashboard_mp4,
    save_embedding_training_mp4,
)
from source_code.gen_data import (
    make_true_P_from_embeddings,
    make_true_memory_sequences,
    make_true_conv_memory_sequences,   # NEW
    sample_markov_sequences,
    markov_entropy_rate,
    save_dataset,
    load_dataset,
)
from source_code.train import (
    PosAttn_TokenOut,
    TrainConfig,
    train_model,
    load_training_artifacts,
    save_training_artifacts,
    save_transformation_data,  
)


@dataclass
class RunConfig:
    # mode
    train_new_run: bool = True
    generate_new_data: bool = True
    save_training_animation: bool = True
    save_rollout_animation: bool = True
    save_embedding_animation: bool = True

    # choose data type
    use_memory_data: bool = True   # False = old Markov, True = memory / conv-memory dynamics
    memory_mu: float = 0.3         # for EMA-style memory (if used)

    # conv-memory kernel params (used by make_true_conv_memory_sequences)
    use_conv_memory: bool = False   # if True, use conv-memory instead of EMA memory
    conv_K: int = 12
    conv_peak_lag: int = 5
    conv_sigma: float = 1.0

    # io
    folder: str = "model_and_data"
    ckpt_name: str = "run_latest.pt"
    dataset_name: str = "dataset_latest.pt"

    # experiment
    V: int = 500  # tokens
    dE: int = 3
    n_seqs: int = 50 * (1000)
    seq_len: int = 20
    true_beta: float = 5.0

    W_true: torch.Tensor = field(
        default_factory=lambda: torch.tensor(
            [[1.0, -0.2, 0.0],
             [0.2, 1.0, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=torch.get_default_dtype(),
        )
    )
    test_frac: float = .2

    # how many tokens to block: frac * V (used if blocked_tokens is empty)
    blocked_frac: float = 0.0          # e.g. 0.05*V tokens; set to 0.5 for 50%
    # explicit override: if non-empty, use these indices instead of random
    blocked_tokens: tuple = tuple(
        int(i) for i in np.random.permutation(V)[: int(0.05 * V)]
    )          # () -> sample randomly

    # training
    n_epochs: int = 800
    lr: float = 3e-3
    batch_size: int = 100
    epochs_per_frame: float = 0.4
    l2_lambda: float = 0 * 1e-5
    log_every_epochs: int = 50


def _get_or_make_dataset(*, cfg: RunConfig):
    data_path = os.path.join(cfg.folder, cfg.dataset_name)

    if (not cfg.generate_new_data) and os.path.exists(data_path):
        train_seqs, test_seqs, E_true, W_true, P_true, data_meta = load_dataset(data_path)
        if E_true is None or W_true is None:  # P_true may be None for memory data
            raise ValueError(f"{data_path} is missing E_true/W_true.")
        return train_seqs, test_seqs, E_true, W_true, P_true, data_meta

    # --- generate + save ---

    if cfg.use_memory_data:
        # NEW: latent memory dynamics, no global P_true
        if cfg.use_conv_memory:
            # convolutional memory: m_t = sum_k alpha_k E_{t-k}
            E_true, W_true, alpha, seqs = make_true_conv_memory_sequences(
                V=cfg.V,
                dim=cfg.dE,
                beta=cfg.true_beta,
                W_true=cfg.W_true,
                n_seqs=cfg.n_seqs,
                seq_len=cfg.seq_len,
                K=cfg.conv_K,
                peak_lag=cfg.conv_peak_lag,
                sigma=cfg.conv_sigma,
                alpha=None,  # let the helper build the Gaussian kernel
            )
        else:
            # original EMA-style memory
            E_true, W_true, seqs = make_true_memory_sequences(
                V=cfg.V,
                dim=cfg.dE,
                beta=cfg.true_beta,
                mu=cfg.memory_mu,
                W_true=cfg.W_true,
                n_seqs=cfg.n_seqs,
                seq_len=cfg.seq_len,
            )
            alpha = None

        P_true = None
        blocked = None
    else:
        # existing Markov data
        if cfg.blocked_tokens:
            blocked = torch.tensor(cfg.blocked_tokens, dtype=torch.long)
        else:
            blocked = None

        E_true, P_true, W_true = make_true_P_from_embeddings(
            V=cfg.V,
            dim=cfg.dE,
            beta=cfg.true_beta,
            W_true=cfg.W_true,
            blocked_tokens=blocked,
        )
        alpha = None
        seqs = sample_markov_sequences(P_true, n_seqs=cfg.n_seqs, seq_len=cfg.seq_len)

    # --- train/test split ---
    n_total = seqs.shape[0]
    n_test = int(round(cfg.test_frac * n_total))
    perm = torch.randperm(n_total)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    train_seqs = seqs[train_idx]
    test_seqs = seqs[test_idx]

    data_meta = dict(
        V=cfg.V,
        dE=cfg.dE,
        seq_len=cfg.seq_len,
        true_beta=cfg.true_beta,
        test_frac=cfg.test_frac,
        n_seqs=cfg.n_seqs,
        blocked_tokens=blocked.cpu().numpy() if (not cfg.use_memory_data and blocked is not None) else None,
        use_memory_data=cfg.use_memory_data,
        memory_mu=cfg.memory_mu if (cfg.use_memory_data and not cfg.use_conv_memory) else None,
        use_conv_memory=cfg.use_conv_memory if cfg.use_memory_data else None,
        conv_K=cfg.conv_K if (cfg.use_memory_data and cfg.use_conv_memory) else None,
        conv_peak_lag=cfg.conv_peak_lag if (cfg.use_memory_data and cfg.use_conv_memory) else None,
        conv_sigma=cfg.conv_sigma if (cfg.use_memory_data and cfg.use_conv_memory) else None,
        alpha=alpha.cpu().numpy() if (cfg.use_memory_data and cfg.use_conv_memory and alpha is not None) else None,
    )

    save_dataset(
        data_path,
        train_seqs=train_seqs,
        test_seqs=test_seqs,
        E_true=E_true,
        W_true=W_true,
        P_true=P_true,   # will be None for memory / conv-memory data
        meta=data_meta,
    )
    return train_seqs, test_seqs, E_true, W_true, P_true, data_meta


def main(cfg: RunConfig = RunConfig()):
    global np  
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_default_dtype(torch.float32)

    os.makedirs(cfg.folder, exist_ok=True)
    ckpt_path = os.path.join(cfg.folder, cfg.ckpt_name)

    train_seqs, test_seqs, E_true, W_true, P_true, data_meta = _get_or_make_dataset(cfg=cfg)

    # recover blocked_tokens for later use (may be None)
    blocked_tokens = data_meta.get("blocked_tokens", None)
    if blocked_tokens is not None:
        blocked_tokens = np.asarray(blocked_tokens, dtype=int)

    # --- Stationary axial density of the TRUE discrete chain (for comparison) ---
    if P_true is not None and torch.is_tensor(P_true):
        with torch.no_grad():
            # P_true is row-stochastic: shape (V, V)
            P = P_true.detach()  # (V, V)

            # Find stationary distribution π: left eigenvector of P with eigenvalue 1
            # i.e. eigenvector of P^T with eigenvalue ~1
            eigvals, eigvecs = torch.linalg.eig(P.t())
            eigvals = eigvals.real
            eigvecs = eigvecs.real  # (V, V) columns are eigenvectors

            idx = torch.argmin((eigvals - 1.0).abs())
            pi = eigvecs[:, idx]                # (V,)
            pi = pi / pi.sum()                  # normalize
            pi = pi.clamp(min=0)
            pi = pi / pi.sum()                  # re-normalize after clamp

            # z-coordinates from TRUE embedding (same gauge as theory)
            z_true = E_true[:, 2].detach().cpu().numpy()
            pi_np = pi.detach().cpu().numpy()

            # build a stationary axial density from the discrete chain
            n_bins = 40
            z_edges = np.linspace(-1.0, 1.0, n_bins + 1)
            z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
            dz = z_edges[1] - z_edges[0]

            stat_hist, _ = np.histogram(z_true, bins=z_edges, weights=pi_np)
            rho_stat = stat_hist / (stat_hist.sum() * dz)  # stationary ρ(z) from P_true

            # analytic continuum theory
            from source_code.analysis import p_z
            a = float(W_true[1, 0].detach().cpu().item())
            rho_theory = p_z(z_centers, a=a, beta=cfg.true_beta)

            l1_discrete_vs_theory = np.sum(np.abs(rho_stat - rho_theory)) * dz
            print("\n[true chain ρ(z)]")
            print("  mean/std rho_stat :", float(rho_stat.mean()), float(rho_stat.std()))
            print("  L1(ρ_stat, ρ_theory):", float(l1_discrete_vs_theory))

            # --- Plot discrete stationary ρ(z) vs analytic p_z(z) ---
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(
                z_centers,
                rho_theory,
                "r-",
                lw=2,
                label="Analytic $p_z(z)$",
            )
            ax.step(
                z_centers,
                rho_stat,
                where="mid",
                color="k",
                lw=1.5,
                label="Discrete stationary $\\rho_{\\text{stat}}(z)$",
            )
            ax.set_xlabel("$z$")
            ax.set_ylabel("density")
            ax.set_title("True chain stationary axial density vs theory")
            ax.legend(loc="best")
            ax.grid(alpha=0.2)

            out_png = os.path.join(cfg.folder, "true_chain_stationary_z.png")
            fig.tight_layout()
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            print("Saved true-chain ρ(z) plot to:", out_png)

            # --- Debias the discrete stationary ρ(z) in the same way as the rollout animation ---

            # Token "support" density from TRUE embedding (unweighted histogram)
            token_hist_true, _ = np.histogram(z_true, bins=z_edges)
            token_density_true = token_hist_true.astype(np.float64) / max(token_hist_true.sum() * dz, 1e-12)
            token_density_true_safe = np.maximum(token_density_true, 1e-12)

            # Debiased stationary density: ρ_stat / ρ_token, re-normalized
            rho_stat_debias = rho_stat / token_density_true_safe
            rho_stat_debias = rho_stat_debias / max(np.sum(rho_stat_debias) * dz, 1e-12)

            l1_debias_vs_theory = np.sum(np.abs(rho_stat_debias - rho_theory)) * dz
            print("\n[true chain ρ(z) debiased with token support]")
            print("  mean/std rho_stat_debias :", float(rho_stat_debias.mean()), float(rho_stat_debias.std()))
            print("  L1(ρ_stat_debias, ρ_theory):", float(l1_debias_vs_theory))

            # Plot: analytic vs discrete vs debiased
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(
                z_centers,
                rho_theory,
                "r-",
                lw=2,
                label="Analytic $p_z(z)$",
            )
            ax2.step(
                z_centers,
                rho_stat,
                where="mid",
                color="k",
                lw=1.5,
                alpha=0.7,
                label="Discrete $\\rho_{\\text{stat}}(z)$",
            )
            ax2.step(
                z_centers,
                rho_stat_debias,
                where="mid",
                color="b",
                lw=1.5,
                alpha=0.7,
                label="Debiased $\\rho_{\\text{stat}}(z)/\\rho_\\text{token}(z)$",
            )
            ax2.set_xlabel("$z$")
            ax2.set_ylabel("density")
            ax2.set_title("True chain ρ(z): raw vs debiased vs theory")
            ax2.legend(loc="best")
            ax2.grid(alpha=0.2)

            out_png2 = os.path.join(cfg.folder, "true_chain_stationary_z_debiased.png")
            fig2.tight_layout()
            fig2.savefig(out_png2, dpi=150)
            plt.close(fig2)
            print("Saved debiased true-chain ρ(z) plot to:", out_png2)

        # theoretical entropy floor from stored P_true (nats)
        P_np = P_true.detach().cpu().numpy()
        H_theory = markov_entropy_rate(P_np)
    else:
        H_theory = 0.0  # no Markov-theory floor for memory data

    # recompute target logits (do not store)
    target_logits = cfg.true_beta * (E_true @ W_true @ E_true.t())

    train_cfg = TrainConfig(
        n_epochs=cfg.n_epochs,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        epochs_per_frame=cfg.epochs_per_frame,
        l2_lambda=cfg.l2_lambda,
        log_every_epochs=cfg.log_every_epochs,
        save_path=None,  # keep I/O policy in main
    )

    transform_path = os.path.join(cfg.folder, "transformation_data.pt")

    if cfg.train_new_run:
        model = PosAttn_TokenOut(
            V=cfg.V,
            L=cfg.seq_len,
            dE=cfg.dE,
            use_batch_invariant_alpha=True,
            use_identity_Mp=False,
        )
        
        model, snaps, meta, rec, snaps_tilde = train_model(
            model,
            train_seqs,
            test_seqs=test_seqs,
            theoretical_entropy_floor=H_theory,
            target_logits=target_logits,
            config=train_cfg,
        )

        save_training_artifacts(
            ckpt_path,
            model=model,
            snaps=snaps,
            meta=meta,
        )

        save_transformation_data(
            transform_path,
            rec=rec,
            snaps_tilde=snaps_tilde,
        )
    else:
        model, snaps, meta = load_training_artifacts(ckpt_path)

        # load transformation data
        if os.path.exists(transform_path):
            td = torch.load(transform_path, map_location="cpu")
            rec = td.get("rec", None)
            snaps_tilde = td.get("snaps_tilde", None)
        else:
            rec, snaps_tilde = None, None


    # -------------------------------------------------------------
    # Learned embedding: support and stationary ρ(z), before/after debias
    # -------------------------------------------------------------
    if P_true is not None and torch.is_tensor(P_true):
        with torch.no_grad():
            # Recompute stationary π of the TRUE chain (same as above)
            P = P_true.detach()
            eigvals, eigvecs = torch.linalg.eig(P.t())
            eigvals = eigvals.real
            eigvecs = eigvecs.real

            idx = torch.argmin((eigvals - 1.0).abs())
            pi = eigvecs[:, idx]
            pi = pi / pi.sum()
            pi = pi.clamp(min=0)
            pi = pi / pi.sum()
            pi_np = pi.detach().cpu().numpy()

            # Learned embedding in rollout gauge: E_learn_plot = E_model @ T_E (if available)
            E_model = model.E.weight.detach().cpu().numpy()  # (V, dE)
            if rec is not None and "T_E" in rec:
                T_E = np.asarray(rec["T_E"])                 # (dE, dE)
                E_learn_plot = E_model @ T_E
            else:
                E_learn_plot = E_model

            z_learn = E_learn_plot[:, 2]

            # Same binning as before
            n_bins = 40
            z_edges = np.linspace(-1.0, 1.0, n_bins + 1)
            z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
            dz = z_edges[1] - z_edges[0]

            # (1) Learned token support density in z
            hist_learn_support, _ = np.histogram(z_learn, bins=z_edges)
            dens_learn_support = hist_learn_support.astype(np.float64) / max(hist_learn_support.sum() * dz, 1e-12)

            print("\n[z_learn support] mean/std density:",
                  float(dens_learn_support.mean()),
                  float(dens_learn_support.std()))
            print("[z_learn support] min/max density :",
                  float(dens_learn_support.min()),
                  float(dens_learn_support.max()))

            # (2) Stationary axial density measured with z_learn (no debias)
            stat_hist_learn, _ = np.histogram(z_learn, bins=z_edges, weights=pi_np)
            rho_stat_learn = stat_hist_learn.astype(np.float64) / max(stat_hist_learn.sum() * dz, 1e-12)

            from source_code.analysis import p_z
            a = float(W_true[1, 0].detach().cpu().item())
            rho_theory = p_z(z_centers, a=a, beta=cfg.true_beta)

            l1_learn_vs_theory = np.sum(np.abs(rho_stat_learn - rho_theory)) * dz
            print("\n[learned emb ρ(z) (no debias)]")
            print("  mean/std rho_stat_learn :", float(rho_stat_learn.mean()), float(rho_stat_learn.std()))
            print("  L1(ρ_stat_learn, ρ_theory):", float(l1_learn_vs_theory))

            # (3) Debias learned stationary ρ(z) with learned token support
            token_density_learn = dens_learn_support
            token_density_learn_safe = np.maximum(token_density_learn, 1e-12)

            rho_stat_learn_debias = rho_stat_learn / token_density_learn_safe
            rho_stat_learn_debias = rho_stat_learn_debias / max(np.sum(rho_stat_learn_debias) * dz, 1e-12)

            l1_learn_debias_vs_theory = np.sum(np.abs(rho_stat_learn_debias - rho_theory)) * dz
            print("\n[learned emb ρ(z) debiased with learned support]")
            print("  mean/std rho_stat_learn_debias :",
                  float(rho_stat_learn_debias.mean()),
                  float(rho_stat_learn_debias.std()))
            print("  L1(ρ_stat_learn_debias, ρ_theory):", float(l1_learn_debias_vs_theory))

    # Use transformed snapshots if available
    snaps_for_dashboard = snaps_tilde if (snaps_tilde is not None) else snaps

    if cfg.save_training_animation:
        out_path = os.path.join(cfg.folder, "training_animation.mp4")
        out_path = save_dashboard_mp4(
            model,
            snaps_for_dashboard,
            out_path=out_path,
            E_true=E_true,
            W_true=W_true,
            meta=meta,
            emb_transform=None,
        )
        print("Saved:", out_path, "\n")

    if cfg.save_embedding_animation:
        # embedding-only training animation
        out_path_embed = os.path.join(cfg.folder, "embedding_training.mp4")
        save_embedding_training_mp4(
            model,
            snaps_for_dashboard,
            out_path=out_path_embed,
            E_true=E_true,
            emb_transform= None, #np.asarray(rec["T_E"]) if rec is not None else None, ## here
            blocked_tokens=blocked_tokens,
        )
        print("Saved:", out_path_embed, "\n")
    # Rollout uses transform if available
    if cfg.save_rollout_animation:
        B_rollout_display = 5
        B_density = 200
        rollout_temperature = 0.1
        rollout_steps = 200

        assert B_density >= B_rollout_display, "B_density must be >= B_rollout_display"

        idx_all = torch.randperm(train_seqs.shape[0])[:B_density]
        seed_batch_all = train_seqs[idx_all, :-1]

        seed_batch_rollout = seed_batch_all[:B_rollout_display]
        seed_batch_density = seed_batch_all

        a = float(W_true[1, 0].detach().cpu().item())
        out_path = os.path.join(cfg.folder, "rollout_dashboard.mp4")

        save_rollout_dashboard_mp4(
            model,
            seed_batch_rollout=seed_batch_rollout,
            seed_batch_density=seed_batch_density,
            out_path=out_path,
            a=a,
            beta=cfg.true_beta / rollout_temperature,
            n_steps=rollout_steps,
            temperature=rollout_temperature,
            sample=True,
            use_soft_point=False,
            rollout_batch_size=B_rollout_display,
            density_batch_size=B_density,
            interval_ms=120, ## emb_transform needs to be adjusted to fix problem
            emb_transform=None, #np.asarray(rec["T_E"]) if rec is not None else None,
            n_z_bins=40,
            n_phi_bins=40,
        )
        print("Saved:", out_path, "\n")


    if rec is not None:
        Etilde = np.asarray(rec["Etilde"])
        Etilde_norms = np.linalg.norm(Etilde, axis=1)

        print("mean Etilde_i :", np.round(Etilde.mean(axis=0), 4))
        print("var  Etilde_i :", np.round(Etilde.var(axis=0), 4))
        print()

        print("mean |Etilde_i| :", np.round(Etilde_norms.mean(), 4))
        print("var  |Etilde_i| :", np.round(Etilde_norms.var(), 4))
        print()

        cov_Etilde = np.cov(Etilde.T)
        print("cov(Etilde):\n", np.round(cov_Etilde, 4))
        print()

        print("Recovered W (Wtilde):\n", np.round(np.asarray(rec["Wtilde"]), 2))
        print()
        print("True W:\n", np.round(W_true.detach().cpu().numpy(), 2))
        print()

    # --- Diagnose diagonal mismatch in |P_target - P_model| ---
    with torch.no_grad():
        device = next(model.parameters()).device
        E_model = model.E.weight.detach().to(device)
        WE_model = model.WE.detach().to(device)
        beta_class = float(model.beta_class.item()) if hasattr(model, "beta_class") else 1.0

        # model kernel -> probabilities
        S_model = beta_class * (E_model @ WE_model @ E_model.t())
        P_model = torch.softmax(S_model, dim=1)

        # target kernel -> probabilities (use the same tensors used for training target_logits)
        E_true_dev = E_true.to(device)
        W_true_dev = W_true.to(device)
        S_tgt = cfg.true_beta * (E_true_dev @ W_true_dev @ E_true_dev.t())
        P_tgt = torch.softmax(S_tgt, dim=1)

        diag_abs = (P_model.diag() - P_tgt.diag()).abs()
        topk = torch.topk(diag_abs, k=min(10, diag_abs.numel()))

        print("\n[diag check] mean |P_model(a,a)-P_tgt(a,a)| =", float(diag_abs.mean().cpu()))
        print("[diag check] max  |P_model(a,a)-P_tgt(a,a)| =", float(diag_abs.max().cpu()))
        print("[diag check] worst tokens (a):", topk.indices.detach().cpu().tolist())

        for a in topk.indices.detach().cpu().tolist():
            print(
                f"  a={a:4d}  P_tgt(a,a)={float(P_tgt[a,a].cpu()):.6f}  "
                f"P_model(a,a)={float(P_model[a,a].cpu()):.6f}  "
                f"|diff|={float(diag_abs[a].cpu()):.6f}"
            )

        # --- Dead tokens with largest incoming P_model from active tokens ---
        if blocked_tokens is not None and blocked_tokens.size > 0 and rec is not None:
            Pm_np = P_model.detach().cpu().numpy()
            V = Pm_np.shape[0]

            dead = np.asarray(blocked_tokens, dtype=int)
            dead = dead[(dead >= 0) & (dead < V)]
            dead = np.unique(dead)

            all_idx = np.arange(V, dtype=int)
            active = np.setdiff1d(all_idx, dead, assume_unique=True)

            if active.size > 0 and dead.size > 0:
                # P_model(active -> dead), shape (n_active, n_dead)
                P_active_to_dead = Pm_np[active[:, None], dead[None, :]]
                max_from_active = P_active_to_dead.max(axis=0)  # length n_dead

                # sort dead tokens by this max probability, descending
                order = np.argsort(-max_from_active)
                k = min(10, dead.size)
                top_idx = order[:k]
                top_dead = dead[top_idx]
                top_max = max_from_active[top_idx]

                Etilde = np.asarray(rec["Etilde"])
                Etilde_norms = np.linalg.norm(Etilde, axis=1)

                print("\n[dead tokens: largest max P_model(i->j) from active i]")
                for j, pmax in zip(top_dead, top_max):
                    print(
                        f"  j={int(j):4d}  max_active P_model(i->j)={pmax:.6f}  "
                        f"|Etilde_j|={Etilde_norms[int(j)]:.4f}"
                    )
                
                            # Global Etilde norm stats
                Etilde = np.asarray(rec["Etilde"])
                Etilde_norms = np.linalg.norm(Etilde, axis=1)
                print(f"mean |Etilde_i| : {Etilde_norms.mean():.4f}")
                print(f"var  |Etilde_i| : {Etilde_norms.var():.4f}")

                # Dead vs active Etilde norms
                if blocked_tokens is not None:
                    blk = np.asarray(blocked_tokens, dtype=int).ravel()
                    blk = blk[(blk >= 0) & (blk < Etilde_norms.shape[0])]
                    blk = np.unique(blk)
                    if blk.size > 0:
                        dead_norms = Etilde_norms[blk]
                        active_mask = np.ones_like(Etilde_norms, dtype=bool)
                        active_mask[blk] = False
                        active_norms = Etilde_norms[active_mask]

                        print(f"[Etilde] dead tokens:   mean |Etilde| = {dead_norms.mean():.4f}  (n={dead_norms.size})")
                        print(f"[Etilde] active tokens: mean |Etilde| = {active_norms.mean():.4f}  (n={active_norms.size})")
    # --- Stationary debiased ρ(z) for P_model in learned gauge (matches animation logic) ---
    with torch.no_grad():
        # P_model from the diag-check block (S_model -> softmax)
        Pm = P_model.detach()  # (V, V)

        eigvals_m, eigvecs_m = torch.linalg.eig(Pm.t())
        eigvals_m = eigvals_m.real
        eigvecs_m = eigvecs_m.real

        idx_m = torch.argmin((eigvals_m - 1.0).abs())
        pi_model = eigvecs_m[:, idx_m]
        pi_model = pi_model / pi_model.sum()
        pi_model = pi_model.clamp(min=0)
        pi_model = pi_model / pi_model.sum()
        pi_model_np = pi_model.detach().cpu().numpy()

        # learned z in rollout gauge (same as before)
        E_model_np = model.E.weight.detach().cpu().numpy()
        if rec is not None and "T_E" in rec:
            T_E = np.asarray(rec["T_E"])
            E_learn_plot = E_model_np @ T_E
        else:
            E_learn_plot = E_model_np
        z_learn = E_learn_plot[:, 2]

        # same bins as rollout
        n_bins = 40
        z_edges = np.linspace(-1.0, 1.0, n_bins + 1)
        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
        dz = z_edges[1] - z_edges[0]

        # token support in learned gauge (what rollout uses)
        hist_learn_support, _ = np.histogram(z_learn, bins=z_edges)
        dens_learn_support = hist_learn_support.astype(np.float64) / max(hist_learn_support.sum() * dz, 1e-12)
        token_density_learn_safe = np.maximum(dens_learn_support, 1e-12)

        # stationary ρ(z) for P_model in learned gauge
        stat_hist_model, _ = np.histogram(z_learn, bins=z_edges, weights=pi_model_np)
        rho_stat_model = stat_hist_model.astype(np.float64) / max(stat_hist_model.sum() * dz, 1e-12)

        # --- BEFORE debiasing: compare P_model stationary ρ(z) to theory ---
        from source_code.analysis import p_z
        a = float(W_true[1, 0].detach().cpu().item())
        rho_theory = p_z(z_centers, a=a, beta=cfg.true_beta)

        l1_model_vs_theory = np.sum(np.abs(rho_stat_model - rho_theory)) * dz
        print("\n[P_model stationary, learned gauge, NO debias]")
        print("  mean/std rho_stat_model :",
              float(rho_stat_model.mean()),
              float(rho_stat_model.std()))
        print("  L1(ρ_stat_model, ρ_theory):", float(l1_model_vs_theory))

        # debias exactly like animation: divide by support, renormalize
        rho_stat_model_debias = rho_stat_model / token_density_learn_safe
        rho_stat_model_debias = rho_stat_model_debias / max(np.sum(rho_stat_model_debias) * dz, 1e-12)

        l1_model_debias_vs_theory = np.sum(np.abs(rho_stat_model_debias - rho_theory)) * dz
        print("\n[P_model stationary, learned gauge, debiased]")
        print("  mean/std rho_stat_model_debias :",
              float(rho_stat_model_debias.mean()),
              float(rho_stat_model_debias.std()))
        print("  L1(ρ_stat_model_debias, ρ_theory):", float(l1_model_debias_vs_theory))

    # --- Inspect learned positional matrix Mp and positional vectors p_t ---
    with torch.no_grad():
        # recovered / scaled-out positional metric from rec
        try:
            Mp_tilde = np.asarray(rec["M_tilde"])
            print("\n[M_tilde] recovered positional metric (canonical gauge):")
            print(np.round(Mp_tilde, 3))
        except Exception:
            pass

        # raw model Mp
        Mp = model.Mp.detach().cpu().numpy()
        print("\n[Mp] learned positional matrix (raw model basis):")
        print(np.round(Mp, 3))

        # ptilde (positional vectors in canonical gauge, from analysis)
        try:
            ptilde = np.asarray(rec["ptilde"])
            print("\n[ptilde] positional vectors (canonical gauge):")
            print("  shape:", ptilde.shape)
            print("  ptilde_t (all rows):\n", np.round(ptilde, 3))

            t_idx = np.arange(ptilde.shape[0])
            for d in range(ptilde.shape[1]):
                corr = np.corrcoef(t_idx, ptilde[:, d])[0, 1]
                print(f"  corr(t, ptilde[:,{d}]): {corr:.3f}")
        except Exception:
            pass

        # Positional vectors from the model (raw basis)
        try:
            Ppos = model.Ppos.detach().cpu().numpy()
            print("\n[Ppos] positional vectors (raw model basis):")
            print("  shape:", Ppos.shape)
            print("  p_t (all rows):\n", np.round(Ppos, 3))

            t_idx = np.arange(Ppos.shape[0])
            for d in range(Ppos.shape[1]):
                corr = np.corrcoef(t_idx, Ppos[:, d])[0, 1]
                print(f"  corr(t, Ppos[:,{d}]): {corr:.3f}")
        except AttributeError:
            print("\n[Ppos] model has no Ppos (old checkpoint or different config).")

    assert cfg.W_true.shape == (cfg.dE, cfg.dE), f"W_true must be ({cfg.dE},{cfg.dE})"


if __name__ == "__main__":
    main(
        RunConfig(
            generate_new_data=False,
            train_new_run=False,
            save_training_animation=False,
            save_rollout_animation=False,
            save_embedding_animation=False,
        )
    )
