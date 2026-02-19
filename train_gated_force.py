"""
Train Gated Force-EGNN: ForceEGNN + gate×magnitude factoring + BCE gate supervision.
Reuses data pipeline from train_force_egnn and integrators from force_egnn.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import physics as P
from physics.engine import generate_dataset, generate_trajectory, WorldConfig
from models.gated_force import GatedForceNet
from models.force_egnn import symplectic_euler_step, kinetic_energy
from train_force_egnn import build_acceleration_dataset


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, A = build_acceleration_dataset(500, 100)
    dt = P.DT

    # Normalize acceleration targets (same as train_force_egnn)
    a_mean = np.mean(A, axis=0)
    a_std = np.std(A, axis=0) + 1e-8
    A_norm = (A - a_mean) / a_std

    X_t = torch.FloatTensor(X)
    A_t = torch.FloatTensor(A_norm)
    a_mean_t = torch.FloatTensor(a_mean).to(device)
    a_std_t = torch.FloatTensor(a_std).to(device)

    model = GatedForceNet(n_balls=P.N_BALLS, hidden_dim=64, n_layers=2, temperature=1.0).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = optim.Adam(model.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5, min_lr=1e-5)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0], device=device))

    collision_lambda, collision_eps = 50.0, 0.01
    best, patience_ctr = float('inf'), 0

    for epoch in range(150):
        perm = torch.randperm(X_t.size(0))
        eloss, nb = 0., 0
        model.train()
        for i in range(0, X_t.size(0), 256):
            bx = X_t[perm[i:i+256]].to(device)
            ba = A_t[perm[i:i+256]].to(device)

            pred_acc = model(bx)
            pred_norm = (pred_acc - a_mean_t) / a_std_t
            gate_logit = model._last_gate_logit  # (B, N, N, 1)

            # --- Acceleration loss (event-aware reweighting) ---
            raw_a = ba * a_std_t + a_mean_t
            acc_mag = raw_a.norm(dim=-1, keepdim=True)
            w = 1.0 + collision_lambda * (acc_mag > collision_eps).float()
            loss_acc = (w * (pred_norm - ba) ** 2).mean()

            # --- Gate BCE loss ---
            dv_norm = (raw_a * dt).norm(dim=-1)  # |Δv| per ball
            ball_active = (dv_norm > collision_eps).float()
            B, N = ball_active.shape
            pair_label = ball_active.unsqueeze(2).expand(B, N, N)
            mask = (1.0 - torch.eye(N, device=device)).unsqueeze(0)
            pair_label = pair_label * mask
            loss_gate = bce(gate_logit.squeeze(-1) * mask, pair_label)

            loss = loss_acc + loss_gate
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            eloss += loss.item(); nb += 1

        avg = eloss / nb
        sched.step(avg)
        if avg < best:
            best = avg; patience_ctr = 0
            torch.save(model.state_dict(), 'results/checkpoints/gated_force.pth')
        else:
            patience_ctr += 1
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}  loss={avg:.6f}  lr={opt.param_groups[0]['lr']:.1e}"
                  f"{'  *' if patience_ctr==0 else ''}")
        if patience_ctr >= 25:
            print(f"Early stop at {epoch+1}"); break

    torch.save({'a_mean': a_mean, 'a_std': a_std, 'dt': dt},
               'results/checkpoints/gated_force_stats.pth')
    print(f"Done. Best={best:.6f}")


def diagnose():
    device = torch.device('cpu')
    model = GatedForceNet(n_balls=P.N_BALLS, hidden_dim=64, n_layers=2, temperature=1.0).to(device)
    model.load_state_dict(torch.load('results/checkpoints/gated_force.pth', map_location=device))
    model.eval()
    stats = torch.load('results/checkpoints/gated_force_stats.pth', map_location=device, weights_only=False)
    dt = stats['dt']

    # -- Teacher-forced impulse accuracy --
    dv_true_n_list, dv_pred_n_list = [], []
    dv_true_norms, dv_pred_norms = [], []
    tang_pred_list = []
    mag_ratio = []
    gate_col, gate_nocol = [], []

    for seed in range(800, 900):
        traj = generate_trajectory(WorldConfig(seed=seed), n_steps=200)
        states, collisions = traj['states'], traj['collisions']
        seen = set()
        for c in collisions:
            step = int(round(c['time'] / dt))
            key = (step, c['ball_i'], c['ball_j'])
            if key in seen or step < 1 or step >= len(states)-1: continue
            seen.add(key)
            st = torch.FloatTensor(states[step]).unsqueeze(0)
            with torch.no_grad():
                acc_pred = model(st)
                dv_pred_all = (acc_pred * dt).numpy()[0]  # acceleration → Δv
                gv = torch.sigmoid(model._last_gate_logit).numpy()[0]
            for b in [c['ball_i'], c['ball_j']]:
                dv_t = states[step+1, b, 2:] - states[step, b, 2:]
                dv_p = dv_pred_all[b]
                n_vec = states[step, c['ball_j'], :2] - states[step, c['ball_i'], :2]
                d = np.linalg.norm(n_vec)
                if d < 1e-8: continue
                nhat = n_vec / d
                dn_t, dn_p = np.dot(dv_t, nhat), np.dot(dv_p, nhat)
                dv_true_n_list.append(dn_t); dv_pred_n_list.append(dn_p)
                dv_true_norms.append(np.linalg.norm(dv_t))
                dv_pred_norms.append(np.linalg.norm(dv_p))
                tang_pred_list.append(np.linalg.norm(dv_p - dn_p*nhat))
                if abs(dn_t) > 1e-6: mag_ratio.append(dn_p / dn_t)
            bi, bj = c['ball_i'], c['ball_j']
            gate_col.append(gv[bi, bj, 0])
            N = states.shape[1]
            for a in range(N):
                for b2 in range(a+1, N):
                    if (a, b2) != (bi, bj): gate_nocol.append(gv[a, b2, 0])

    mr = np.array(mag_ratio)
    gc, gnc = np.array(gate_col), np.array(gate_nocol)
    print(f"\n{'='*60}")
    print(f"GATED FORCE-EGNN DIAGNOSTIC — {len(dv_true_norms)//2} collisions")
    print(f"{'='*60}")
    print(f"|Δv_true|={np.mean(dv_true_norms):.4f}  |Δv_pred|={np.mean(dv_pred_norms):.4f}")
    print(f"Normal err:     {np.mean(np.abs(np.array(dv_true_n_list)-np.array(dv_pred_n_list))):.4f}")
    print(f"Tang pred mean: {np.mean(tang_pred_list):.6f}")
    print(f"Mag ratio: mean={mr.mean():.3f} median={np.median(mr):.3f} std={mr.std():.3f}")
    print(f"\nGate:  collision={gc.mean():.3f}  non-coll={gnc.mean():.3f}  sep={gc.mean()-gnc.mean():.3f}")

    # -- Energy rollout (symplectic) --
    drifts = []
    energies_true, energies_pred = [], []
    for seed in range(900, 910):
        traj = generate_trajectory(WorldConfig(seed=seed), n_steps=200)
        true_s = traj['states']
        cur = torch.FloatTensor(true_s[0]).unsqueeze(0)
        pred_e = [0.5 * (true_s[0, :, 2:]**2).sum()]
        with torch.no_grad():
            for _ in range(200):
                acc = model(cur)
                cur = symplectic_euler_step(cur, acc, dt)
                pred_e.append(0.5 * (cur[0, :, 2:]**2).sum().item())
        true_e = [0.5 * (true_s[t, :, 2:]**2).sum() for t in range(201)]
        drift = (pred_e[-1] - true_e[0]) / true_e[0] * 100
        drifts.append(drift)
        energies_true.append(true_e)
        energies_pred.append(pred_e)

    d = np.array(drifts)
    print(f"\nEnergy (symplectic, 200 steps, 10 seeds):")
    print(f"  mean={d.mean():+.1f}%  median={np.median(d):+.1f}%  std={d.std():.1f}  range=[{d.min():+.0f}%, {d.max():+.0f}%]")

    # Plot
    et = np.mean(energies_true, axis=0)
    ep = np.mean(energies_pred, axis=0)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Gated Force-EGNN — Diagnostics', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.scatter(dv_true_norms, dv_pred_norms, alpha=0.4, s=15, c='#3498db')
    lim = max(max(dv_true_norms), max(dv_pred_norms)) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', lw=1); ax.set_xlabel('|Δv_true|'); ax.set_ylabel('|Δv_pred|')
    ax.set_title('Impulse Magnitude'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(dv_true_n_list, dv_pred_n_list, alpha=0.4, s=15, c='#e74c3c')
    lim = max(abs(np.array(dv_true_n_list)).max(), abs(np.array(dv_pred_n_list)).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1); ax.set_xlabel('Δv_true·n̂'); ax.set_ylabel('Δv_pred·n̂')
    ax.set_title('Normal Component'); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.hist(np.clip(mr, -2, 4), bins=40, color='#2ecc71', alpha=0.7, edgecolor='k', lw=0.5)
    ax.axvline(1.0, color='red', lw=2, label=f'perfect (1.0)')
    ax.axvline(mr.mean(), color='blue', lw=1.5, ls='--', label=f'mean ({mr.mean():.2f})')
    ax.set_xlabel('pred_n / true_n'); ax.set_title('Impulse Ratio'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    steps = np.arange(len(et))
    ax.plot(steps, et, 'k-', lw=2, alpha=0.5, label='True')
    ax.plot(steps, ep, '--', color='#9b59b6', lw=1.5, label='Predicted (symplectic)')
    ax.set_title(f'Energy (drift: {d.mean():+.1f}%)'); ax.set_xlabel('Step'); ax.set_ylabel('KE')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/plots/gated_force_diagnostic.png', dpi=150)
    plt.close()
    print(f"Plot: results/plots/gated_force_diagnostic.png")


if __name__ == "__main__":
    train()
    diagnose()
