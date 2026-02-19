import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import physics as P
from physics.engine import generate_dataset, generate_trajectory, WorldConfig
from models.normal_impulse import NormalImpulseNet

def build_dv_dataset(n_trajectories=500, n_steps=100):
    trajs = generate_dataset(n_trajectories=n_trajectories, n_steps=n_steps)
    X, DV = [], []
    for traj in trajs:
        s = traj['states']
        X.append(s[:-1])
        DV.append(s[1:, :, 2:] - s[:-1, :, 2:])
    X = np.concatenate(X).astype(np.float32)
    DV = np.concatenate(DV).astype(np.float32)
    norms = np.linalg.norm(DV.reshape(-1, 2), axis=1)
    print(f"Dataset: {X.shape[0]} samples | |Δv| mean={norms.mean():.4f} max={norms.max():.2f} "
          f"pct>0.01={100*np.mean(norms>0.01):.1f}%")
    return X, DV

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, DV = build_dv_dataset(500, 100)

    # Collision labels per sample: 1 if any ball has |Δv| > eps
    # We'll compute per-pair labels inside the loop from the Δv norms
    eps_col = 0.01

    X_t = torch.FloatTensor(X)
    DV_t = torch.FloatTensor(DV)  # raw, no normalization

    model = NormalImpulseNet(n_balls=P.N_BALLS, hidden_dim=64, temperature=0.3).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}  temperature={model.temperature}")

    opt = optim.Adam(model.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5, min_lr=1e-5)
    best, patience_ctr = float('inf'), 0

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50.0], device=device))

    for epoch in range(150):
        perm = torch.randperm(X_t.size(0))
        eloss, nb = 0., 0
        model.train()
        for i in range(0, X_t.size(0), 256):
            bx = X_t[perm[i:i+256]].to(device)     # (B, N, 4)
            bdv = DV_t[perm[i:i+256]].to(device)    # (B, N, 2) true Δv

            pred_dv = model(bx)                       # (B, N, 2)
            gate_logit = model._last_gate_logit       # (B, N, N, 1)

            # --- Collision label per ball: |Δv_i| > eps ---
            dv_norm = bdv.norm(dim=-1)                # (B, N)
            ball_active = (dv_norm > eps_col).float() # (B, N)

            # Per-pair label: pair (i,j) is active if ball i has collision impulse
            # gate_logit[b,i,j] controls force on ball i from j
            # Label it as active if ball i is involved in a collision
            B, N = ball_active.shape
            pair_label = ball_active.unsqueeze(2).expand(B, N, N)  # (B, N, N)
            mask = (1.0 - torch.eye(N, device=device)).unsqueeze(0)  # (1, N, N)
            pair_label = pair_label * mask

            # BCE loss on gate (logits are raw, but need to be scaled by 1/T for BCE to match sigmoid(logit/T))
            loss_gate = bce(gate_logit.squeeze(-1) * mask / model.temperature, pair_label)

            # MSE on Δv only for collision samples (balls with |Δv| > eps)
            active_mask = (dv_norm > eps_col).unsqueeze(-1).float()  # (B, N, 1)
            if active_mask.sum() > 0:
                loss_mag = ((pred_dv - bdv)**2 * active_mask).sum() / active_mask.sum().clamp(min=1)
            else:
                loss_mag = torch.tensor(0., device=device)

            # Also penalize non-collision predictions being nonzero (keep quiet when no event)
            quiet_mask = 1.0 - active_mask
            loss_quiet = ((pred_dv)**2 * quiet_mask).mean()

            loss = loss_gate + loss_mag + 0.1 * loss_quiet

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            eloss += loss.item(); nb += 1
        avg = eloss / nb
        sched.step(avg)
        if avg < best:
            best = avg; patience_ctr = 0
            torch.save(model.state_dict(), 'results/checkpoints/normal_impulse.pth')
        else:
            patience_ctr += 1
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}  loss={avg:.6f}  lr={opt.param_groups[0]['lr']:.1e}"
                  f"{'  *' if patience_ctr==0 else ''}")
        if patience_ctr >= 25:
            print(f"Early stop at {epoch+1}"); break

    torch.save({}, 'results/checkpoints/normal_impulse_stats.pth')
    print(f"Done. Best={best:.6f}")

def diagnose():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NormalImpulseNet(n_balls=P.N_BALLS, hidden_dim=64, temperature=0.3).to(device)
    model.load_state_dict(torch.load('results/checkpoints/normal_impulse.pth', map_location=device))
    model.eval()
    dt = P.DT

    # -- Impulse accuracy (teacher-forced) --
    dv_true_n_list, dv_pred_n_list = [], []
    dv_true_norms, dv_pred_norms = [], []
    tang_true, tang_pred = [], []
    mag_ratio = []
    gate_at_collision, gate_at_noncollision = [], []

    for seed in range(800, 900):
        traj = generate_trajectory(WorldConfig(seed=seed), n_steps=200)
        states = traj['states']
        collisions = traj['collisions']
        seen = set()
        for c in collisions:
            step = int(round(c['time'] / dt))
            key = (step, c['ball_i'], c['ball_j'])
            if key in seen or step < 1 or step >= len(states)-1: continue
            seen.add(key)
            st = torch.FloatTensor(states[step]).unsqueeze(0).to(device)
            with torch.no_grad():
                dv_pred_all = model(st).cpu().numpy()[0]
                gate_vals = torch.sigmoid(model._last_gate_logit / model.temperature).cpu().numpy()[0]  # (N, N, 1)
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
                tang_true.append(np.linalg.norm(dv_t - dn_t*nhat))
                tang_pred.append(np.linalg.norm(dv_p - dn_p*nhat))
                if abs(dn_t) > 1e-6: mag_ratio.append(dn_p / dn_t)
            # Gate diagnostic: colliding pair vs non-colliding
            bi, bj = c['ball_i'], c['ball_j']
            gate_at_collision.append(gate_vals[bi, bj, 0])
            N = states.shape[1]
            for a in range(N):
                for b2 in range(a+1, N):
                    if (a, b2) != (bi, bj):
                        gate_at_noncollision.append(gate_vals[a, b2, 0])

    dv_true_norms = np.array(dv_true_norms)
    dv_pred_norms = np.array(dv_pred_norms)
    mag_ratio = np.array(mag_ratio)

    print(f"\n{'='*60}")
    print(f"IMPULSE DIAGNOSTIC — {len(dv_true_norms)//2} collisions")
    print(f"{'='*60}")
    print(f"|Δv_true|={dv_true_norms.mean():.4f}  |Δv_pred|={dv_pred_norms.mean():.4f}")
    print(f"Normal err:     {np.mean(np.abs(np.array(dv_true_n_list)-np.array(dv_pred_n_list))):.4f}")
    print(f"Tangential err: {np.mean(np.abs(np.array(tang_true)-np.array(tang_pred))):.6f}")
    print(f"Magnitude ratio: mean={mag_ratio.mean():.3f} median={np.median(mag_ratio):.3f} std={mag_ratio.std():.3f}")
    print(f"Tang pred mean:  {np.mean(tang_pred):.6f}  (should be ~0)")
    gc = np.array(gate_at_collision)
    gnc = np.array(gate_at_noncollision)
    print(f"\nGate activation:")
    print(f"  At collision:     mean={gc.mean():.3f} median={np.median(gc):.3f}")
    print(f"  At non-collision: mean={gnc.mean():.3f} median={np.median(gnc):.3f}")
    print(f"  Separation:       {gc.mean() - gnc.mean():.3f} (want >> 0)")

    energies_true, energies_pred = [], []
    for seed in [900, 901, 902, 903, 904]:
        traj = generate_trajectory(WorldConfig(seed=seed), n_steps=200)
        true_s = traj['states']
        cur = torch.FloatTensor(true_s[0]).unsqueeze(0).to(device)
        pred_s = [true_s[0]]
        with torch.no_grad():
            for _ in range(200):
                dv = model(cur)
                pos, vel = cur[:,:,:2], cur[:,:,2:]
                new_vel = vel + dv
                new_pos = pos + new_vel * dt
                cur = torch.cat([new_pos, new_vel], dim=-1)
                pred_s.append(cur.cpu().numpy()[0])
        pred_s = np.array(pred_s)
        energies_true.append(0.5 * (true_s[:,:,2:]**2).sum(axis=(1,2)))
        energies_pred.append(0.5 * (pred_s[:,:,2:]**2).sum(axis=(1,2)))

    et = np.mean(energies_true, axis=0)
    ep = np.mean(energies_pred, axis=0)

    os.makedirs('results/plots', exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Normal-Only Impulse Net — Diagnostics', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.scatter(dv_true_norms, dv_pred_norms, alpha=0.4, s=15, c='#3498db')
    lim = max(dv_true_norms.max(), dv_pred_norms.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label='perfect')
    ax.set_xlabel('|Δv_true|'); ax.set_ylabel('|Δv_pred|')
    ax.set_title('Impulse Magnitude'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(dv_true_n_list, dv_pred_n_list, alpha=0.4, s=15, c='#e74c3c')
    lim = max(abs(np.array(dv_true_n_list)).max(), abs(np.array(dv_pred_n_list)).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1, label='perfect')
    ax.set_xlabel('Δv_true·n̂'); ax.set_ylabel('Δv_pred·n̂')
    ax.set_title('Normal Component'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    clipped = np.clip(mag_ratio, -2, 4)
    ax.hist(clipped, bins=40, color='#2ecc71', alpha=0.7, edgecolor='k', lw=0.5)
    ax.axvline(1.0, color='red', lw=2, label='perfect (1.0)')
    ax.axvline(mag_ratio.mean(), color='blue', lw=1.5, ls='--', label=f'mean ({mag_ratio.mean():.2f})')
    ax.set_xlabel('pred_n / true_n'); ax.set_title('Impulse Ratio'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    steps = np.arange(len(et))
    ax.plot(steps, et, 'k-', lw=2, alpha=0.5, label='True')
    ax.plot(steps, ep, '--', color='#9b59b6', lw=1.5, label='Predicted')
    drift = (ep[-1] - et[0]) / et[0] * 100
    ax.set_title(f'Energy (drift: {drift:+.1f}%)')
    ax.set_xlabel('Step'); ax.set_ylabel('KE'); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/plots/normal_impulse_diagnostic.png', dpi=150)
    plt.close()
    print(f"\nPlot: results/plots/normal_impulse_diagnostic.png")


if __name__ == "__main__":
    train()
    diagnose()
