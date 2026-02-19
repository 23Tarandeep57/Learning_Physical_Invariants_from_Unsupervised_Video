"""
Impulse Diagnostic — Does the model learn the discrete collision law?

For each collision event, compare Δv_true vs Δv_pred in a [t-1, t, t+1] window.
Decompose error into normal and tangential components to test:
  - Is impulse magnitude correct?
  - Is impulse direction correct (normal-only)?
  - Does tangential leakage exist?
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import physics as P
from physics.engine import generate_trajectory, WorldConfig
from models.checkpoints import load_force_egnn
from models.rollout import rollout_force


def find_collision_steps(traj, dt):
    """Map collision log times → step indices, deduplicated per step."""
    seen = set()
    events = []
    for c in traj['collisions']:
        step = int(round(c['time'] / dt))
        key = (step, c['ball_i'], c['ball_j'])
        if key not in seen and 1 <= step < len(traj['states']) - 1:
            seen.add(key)
            events.append({'step': step, 'i': c['ball_i'], 'j': c['ball_j']})
    return events


def analyze():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_force_egnn(device)
    dt = P.DT
    n_steps = 200
    os.makedirs('results/plots', exist_ok=True)

    # Collect impulse data across many trajectories
    # TEACHER-FORCED: feed true state at t, predict acc, integrate one step,
    # compare Δv. This isolates impulse accuracy from rollout divergence.
    normal_err, tangent_err, magnitude_ratio = [], [], []
    dv_true_norms, dv_pred_norms = [], []
    dv_true_normal_comp, dv_pred_normal_comp = [], []
    dv_true_tangent_comp, dv_pred_tangent_comp = [], []
    total_collisions = 0

    for seed in range(800, 900):
        config = WorldConfig(seed=seed)
        traj = generate_trajectory(config, n_steps=n_steps)
        true_states = traj['states']
        events = find_collision_steps(traj, dt)
        if not events:
            continue

        for ev in events:
            t, i, j = ev['step'], ev['i'], ev['j']
            total_collisions += 1

            # Teacher-forced: feed TRUE state at t, get model's one-step prediction
            state_t = torch.FloatTensor(true_states[t]).unsqueeze(0).to(device)
            with torch.no_grad():
                acc = model(state_t)
                pred_next = symplectic_euler_step(state_t, acc, dt)
            pred_next_np = pred_next.cpu().numpy()[0]

            for b in [i, j]:
                dv_true = true_states[t + 1, b, 2:] - true_states[t, b, 2:]
                dv_pred = pred_next_np[b, 2:] - true_states[t, b, 2:]

                # Contact normal: direction between ball centers at collision
                pos_i = true_states[t, i, :2]
                pos_j = true_states[t, j, :2]
                n_vec = pos_j - pos_i
                n_norm = np.linalg.norm(n_vec)
                if n_norm < 1e-8:
                    continue
                n_hat = n_vec / n_norm

                # Decompose into normal and tangential
                dv_true_n = np.dot(dv_true, n_hat)
                dv_true_t = dv_true - dv_true_n * n_hat
                dv_pred_n = np.dot(dv_pred, n_hat)
                dv_pred_t = dv_pred - dv_pred_n * n_hat

                # Store
                dv_true_norms.append(np.linalg.norm(dv_true))
                dv_pred_norms.append(np.linalg.norm(dv_pred))
                normal_err.append(abs(dv_true_n - dv_pred_n))
                tangent_err.append(np.linalg.norm(dv_true_t - dv_pred_t))
                dv_true_normal_comp.append(dv_true_n)
                dv_pred_normal_comp.append(dv_pred_n)
                dv_true_tangent_comp.append(np.linalg.norm(dv_true_t))
                dv_pred_tangent_comp.append(np.linalg.norm(dv_pred_t))
                if abs(dv_true_n) > 1e-6:
                    magnitude_ratio.append(dv_pred_n / dv_true_n)

    # ── Print summary ──
    normal_err = np.array(normal_err)
    tangent_err = np.array(tangent_err)
    dv_true_norms = np.array(dv_true_norms)
    dv_pred_norms = np.array(dv_pred_norms)
    magnitude_ratio = np.array(magnitude_ratio)

    print(f"\n{'='*60}")
    print(f"IMPULSE DIAGNOSTIC — {total_collisions} collisions from 100 trajectories")
    print(f"{'='*60}")

    print(f"\n|Δv_true| mean: {dv_true_norms.mean():.4f}  |Δv_pred| mean: {dv_pred_norms.mean():.4f}")
    print(f"\nNormal component error:     {normal_err.mean():.4f} ± {normal_err.std():.4f}")
    print(f"Tangential component error: {tangent_err.mean():.4f} ± {tangent_err.std():.4f}")
    print(f"\nImpulse magnitude ratio (pred_n / true_n):")
    print(f"  mean:   {magnitude_ratio.mean():.3f}  (1.0 = perfect)")
    print(f"  median: {np.median(magnitude_ratio):.3f}")
    print(f"  std:    {magnitude_ratio.std():.3f}")

    # Is tangential leakage real?
    true_tang = np.array(dv_true_tangent_comp)
    pred_tang = np.array(dv_pred_tangent_comp)
    print(f"\nTrue tangential |Δv_⊥|:  {true_tang.mean():.6f}  (should be ~0 for hard-sphere)")
    print(f"Pred tangential |Δv_⊥|:  {pred_tang.mean():.6f}  (leakage if >> true)")

    # ── Plots ──
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f'Impulse Diagnostic — {total_collisions} collision events', fontsize=14, fontweight='bold')

    # 1. Δv magnitude: true vs pred
    ax = axes[0, 0]
    ax.scatter(dv_true_norms, dv_pred_norms, alpha=0.4, s=15, c='#3498db')
    lim = max(dv_true_norms.max(), dv_pred_norms.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1, label='perfect')
    ax.set_xlabel('|Δv_true|')
    ax.set_ylabel('|Δv_pred|')
    ax.set_title('Impulse Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 2. Normal component: true vs pred
    ax = axes[0, 1]
    ax.scatter(dv_true_normal_comp, dv_pred_normal_comp, alpha=0.4, s=15, c='#e74c3c')
    lim = max(abs(np.array(dv_true_normal_comp)).max(), abs(np.array(dv_pred_normal_comp)).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1, label='perfect')
    ax.set_xlabel('Δv_true · n̂')
    ax.set_ylabel('Δv_pred · n̂')
    ax.set_title('Normal Component (impulse law)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 3. Magnitude ratio histogram
    ax = axes[1, 0]
    clipped = np.clip(magnitude_ratio, -2, 4)
    ax.hist(clipped, bins=40, color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(1.0, color='red', linewidth=2, label='perfect (1.0)')
    ax.axvline(magnitude_ratio.mean(), color='blue', linewidth=1.5, linestyle='--',
               label=f'mean ({magnitude_ratio.mean():.2f})')
    ax.set_xlabel('pred_n / true_n')
    ax.set_ylabel('Count')
    ax.set_title('Impulse Ratio Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Normal vs tangential error
    ax = axes[1, 1]
    ax.scatter(normal_err, tangent_err, alpha=0.4, s=15, c='#9b59b6')
    ax.set_xlabel('Normal error |Δv_n_true - Δv_n_pred|')
    ax.set_ylabel('Tangential error |Δv_⊥_true - Δv_⊥_pred|')
    ax.set_title('Error Decomposition (normal vs tangential)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('results/plots/impulse_diagnostic.png', dpi=150)
    plt.close()
    print(f"\nPlot saved: results/plots/impulse_diagnostic.png")


if __name__ == "__main__":
    analyze()
