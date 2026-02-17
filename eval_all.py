"""
Comprehensive evaluation — all three ablation levels.

  Level 0: MLP           (no structure)
  Level 1: EGNN          (equivariance only)
  Level 2: Force EGNN    (equivariance + force structure + symplectic integration)

Metrics:
  1. Per-step MSE (trajectory accuracy)
  2. Energy conservation (total KE drift)
  3. Momentum conservation (total |p| drift)
  4. Per-ball trajectory comparison
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import physics as P
from physics.engine import generate_trajectory, WorldConfig
from models.mlp_dynamics import MLPDynamics
from models.egnn_dynamics import EGNNDynamics
from models.force_egnn import ForceEGNN, symplectic_euler_step, leapfrog_step


# ═══════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════

def load_mlp(device):
    model = MLPDynamics(n_balls=P.N_BALLS).to(device)
    model.load_state_dict(torch.load('results/checkpoints/mlp_baseline.pth', map_location=device))
    stats = torch.load('results/checkpoints/mlp_stats.pth', weights_only=False)
    model.eval()
    return model, stats

def load_egnn(device):
    model = EGNNDynamics(n_balls=P.N_BALLS, hidden_dim=64, n_layers=3).to(device)
    model.load_state_dict(torch.load('results/checkpoints/egnn_baseline.pth', map_location=device))
    stats = torch.load('results/checkpoints/egnn_stats.pth', weights_only=False)
    model.eval()
    return model, stats

def load_force_egnn(device):
    model = ForceEGNN(
        n_balls=P.N_BALLS, hidden_dim=64, n_layers=3,
        world_width=P.WORLD_WIDTH, world_height=P.WORLD_HEIGHT
    ).to(device)
    # Try best checkpoint first, fall back to final
    ckpt_path = 'results/checkpoints/force_egnn_best.pth'
    if not os.path.exists(ckpt_path):
        ckpt_path = 'results/checkpoints/force_egnn.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════
# Rollout functions
# ═══════════════════════════════════════════════════════════════════

def rollout_mlp(model, stats, init_state, n_steps, device):
    x_mean = torch.FloatTensor(stats['x_mean']).to(device)
    x_std = torch.FloatTensor(stats['x_std']).to(device)
    y_mean = torch.FloatTensor(stats['y_mean']).to(device)
    y_std = torch.FloatTensor(stats['y_std']).to(device)

    states = [init_state.copy()]
    current = torch.FloatTensor(init_state).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(n_steps):
            norm_in = (current - x_mean) / x_std
            norm_delta = model(norm_in)
            delta = norm_delta * y_std + y_mean
            current = current + delta
            states.append(current.cpu().numpy()[0])
    return np.array(states)

def rollout_egnn(model, stats, init_state, n_steps, device):
    y_mean = torch.FloatTensor(stats['y_mean']).to(device)
    y_std = torch.FloatTensor(stats['y_std']).to(device)

    states = [init_state.copy()]
    current = torch.FloatTensor(init_state).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(n_steps):
            pred_delta = model(current)
            current = current + pred_delta
            states.append(current.cpu().numpy()[0])
    return np.array(states)

def rollout_force_egnn(model, init_state, n_steps, device, dt=P.DT,
                       integrator='symplectic_euler'):
    """Force model rollout: predict acceleration → integrate."""
    states = [init_state.copy()]
    current = torch.FloatTensor(init_state).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(n_steps):
            if integrator == 'leapfrog':
                current = leapfrog_step(current, model, dt)
            else:
                acc = model(current)
                current = symplectic_euler_step(current, acc, dt)
            states.append(current.cpu().numpy()[0])
    return np.array(states)


# ═══════════════════════════════════════════════════════════════════
# Physics metrics
# ═══════════════════════════════════════════════════════════════════

def compute_energy(states, masses=None):
    if masses is None:
        masses = np.ones(states.shape[1])
    vel = states[:, :, 2:]
    ke = 0.5 * masses[None, :, None] * vel ** 2
    return ke.sum(axis=(1, 2))

def compute_momentum(states, masses=None):
    if masses is None:
        masses = np.ones(states.shape[1])
    vel = states[:, :, 2:]
    p = (masses[None, :, None] * vel).sum(axis=1)
    return np.linalg.norm(p, axis=1)


# ═══════════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════════

COLORS = {
    'MLP':   '#e74c3c',  # red
    'EGNN':  '#2ecc71',  # green
    'Force': '#3498db',  # blue
    'True':  '#2c3e50',  # dark
}

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_steps = 200
    n_test_seeds = 5
    os.makedirs('results/plots', exist_ok=True)

    # ── Load all models ──
    models = {}
    try:
        mlp_model, mlp_stats = load_mlp(device)
        models['MLP'] = ('mlp', mlp_model, mlp_stats)
        print("✓ Loaded MLP (Level 0)")
    except FileNotFoundError:
        print("✗ MLP checkpoint not found")

    try:
        egnn_model, egnn_stats = load_egnn(device)
        models['EGNN'] = ('egnn', egnn_model, egnn_stats)
        print("✓ Loaded EGNN (Level 1)")
    except FileNotFoundError:
        print("✗ EGNN checkpoint not found")

    try:
        force_model = load_force_egnn(device)
        models['Force'] = ('force', force_model, None)
        print("✓ Loaded Force EGNN (Level 2)")
    except FileNotFoundError:
        print("✗ Force EGNN checkpoint not found")

    if len(models) == 0:
        print("No models found. Train at least one model first.")
        return

    # ── Collect metrics ──
    all_mse = {name: [] for name in models}
    all_energy = {name: [] for name in models}
    all_energy_true = []
    all_mom = {name: [] for name in models}
    all_mom_true = []

    print(f"\nEvaluating on {n_test_seeds} test trajectories × {n_steps} steps...")
    for seed in range(900, 900 + n_test_seeds):
        config = WorldConfig(seed=seed)
        traj = generate_trajectory(config, n_steps=n_steps)
        true_states = traj['states']
        masses = traj['full_states'][0, :, 5]

        all_energy_true.append(compute_energy(true_states, masses))
        all_mom_true.append(compute_momentum(true_states, masses))

        for name, (kind, model, stats) in models.items():
            if kind == 'mlp':
                pred = rollout_mlp(model, stats, true_states[0], n_steps, device)
            elif kind == 'egnn':
                pred = rollout_egnn(model, stats, true_states[0], n_steps, device)
            elif kind == 'force':
                pred = rollout_force_egnn(model, true_states[0], n_steps, device)

            mse = np.mean((true_states - pred) ** 2, axis=(1, 2))
            all_mse[name].append(mse)
            all_energy[name].append(compute_energy(pred, masses))
            all_mom[name].append(compute_momentum(pred, masses))

    # Average
    avg_mse = {n: np.mean(v, axis=0) for n, v in all_mse.items()}
    avg_energy = {n: np.mean(v, axis=0) for n, v in all_energy.items()}
    avg_energy_true = np.mean(all_energy_true, axis=0)
    avg_mom = {n: np.mean(v, axis=0) for n, v in all_mom.items()}
    avg_mom_true = np.mean(all_mom_true, axis=0)

    # ── Print summary table ──
    print("\n" + "=" * 72)
    print("ABLATION RESULTS: Level 0 (MLP) vs Level 1 (EGNN) vs Level 2 (Force)")
    print("=" * 72)

    header = f"{'Step':>6}"
    for name in models:
        header += f"  {name:>12}"
    print(f"\n{header}")
    print("-" * len(header))
    for step in [1, 10, 50, 100, 200]:
        row = f"{step:>6}"
        for name in models:
            row += f"  {avg_mse[name][step]:>12.6f}"
        print(row)

    e0 = avg_energy_true[0]
    print(f"\nEnergy (true t=0: {e0:.4f}):")
    for name in models:
        final_e = avg_energy[name][-1]
        drift_pct = (final_e - e0) / e0 * 100
        print(f"  {name:>8} final: {final_e:.4f}  (drift: {drift_pct:+.2f}%)")

    print(f"\nMomentum |p| (true t=0: {avg_mom_true[0]:.4f}):")
    for name in models:
        print(f"  {name:>8} final: {avg_mom[name][-1]:.4f}")

    # ── Energy drift statistic: mean absolute relative error ──
    print(f"\nMean |ΔE/E₀| over rollout:")
    for name in models:
        rel_err = np.abs(avg_energy[name] - avg_energy_true) / (avg_energy_true + 1e-8)
        print(f"  {name:>8}: {rel_err.mean()*100:.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # Plot 1: 4-panel ablation (like the L0 vs L1 plot, now with L2)
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Ablation: Level 0 (MLP) → Level 1 (EGNN) → Level 2 (Force EGNN)',
                 fontsize=14, fontweight='bold')
    steps = np.arange(n_steps + 1)

    level_labels = {'MLP': 'MLP (Level 0)', 'EGNN': 'EGNN (Level 1)', 'Force': 'Force EGNN (Level 2)'}

    # 1. MSE
    ax = axes[0, 0]
    for name in models:
        ax.semilogy(steps, avg_mse[name], label=level_labels.get(name, name),
                    color=COLORS[name], alpha=0.8)
    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('MSE (log scale)')
    ax.set_title('Trajectory Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Energy
    ax = axes[0, 1]
    ax.plot(steps, avg_energy_true, label='Ground Truth',
            color=COLORS['True'], linewidth=2, alpha=0.6)
    for name in models:
        ax.plot(steps, avg_energy[name], label=level_labels.get(name, name),
                color=COLORS[name], linestyle='--', alpha=0.8)
    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('Total Kinetic Energy')
    ax.set_title('Energy Conservation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Energy drift (%)
    ax = axes[1, 0]
    for name in models:
        err = (avg_energy[name] - avg_energy_true) / (avg_energy_true + 1e-8) * 100
        ax.plot(steps, err, label=level_labels.get(name, name),
                color=COLORS[name], alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('Relative Energy Error (%)')
    ax.set_title('Energy Drift')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Momentum
    ax = axes[1, 1]
    ax.plot(steps, avg_mom_true, label='Ground Truth',
            color=COLORS['True'], linewidth=2, alpha=0.6)
    for name in models:
        ax.plot(steps, avg_mom[name], label=level_labels.get(name, name),
                color=COLORS[name], linestyle='--', alpha=0.8)
    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('|Total Momentum|')
    ax.set_title('Momentum Conservation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/plots/ablation_all_levels.png', dpi=150)
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # Plot 2: Single trajectory per-ball comparison
    # ═══════════════════════════════════════════════════════════════
    config = WorldConfig(seed=999)
    traj = generate_trajectory(config, n_steps=n_steps)
    true = traj['states']
    masses = traj['full_states'][0, :, 5]

    preds = {}
    for name, (kind, model, stats) in models.items():
        if kind == 'mlp':
            preds[name] = rollout_mlp(model, stats, true[0], n_steps, device)
        elif kind == 'egnn':
            preds[name] = rollout_egnn(model, stats, true[0], n_steps, device)
        elif kind == 'force':
            preds[name] = rollout_force_egnn(model, true[0], n_steps, device)

    fig, axes = plt.subplots(P.N_BALLS, 2, figsize=(14, 4 * P.N_BALLS))
    if P.N_BALLS == 1:
        axes = axes[None, :]

    for b in range(P.N_BALLS):
        for col, coord, label in [(0, 0, 'X'), (1, 1, 'Y')]:
            ax = axes[b, col]
            ax.plot(true[:, b, coord], label='True', color=COLORS['True'], alpha=0.6, linewidth=2)
            for name in models:
                ax.plot(preds[name][:, b, coord], label=level_labels.get(name, name),
                        color=COLORS[name], linestyle='--', alpha=0.7)
            ax.set_title(f'Ball {b} — {label} position')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Single Trajectory Rollout (seed=999)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/plots/trajectory_all_levels.png', dpi=150)
    plt.close()

    # ═══════════════════════════════════════════════════════════════
    # Plot 3: Energy over time for the single trajectory (detailed)
    # ═══════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 5))
    true_e = compute_energy(true, masses)
    ax.plot(steps, true_e, label='Ground Truth', color=COLORS['True'],
            linewidth=2, alpha=0.6)
    for name in models:
        pred_e = compute_energy(preds[name], masses)
        ax.plot(steps, pred_e, label=level_labels.get(name, name),
                color=COLORS[name], linestyle='--', alpha=0.8)

    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('Total Kinetic Energy')
    ax.set_title('Energy Conservation — Single Trajectory (seed=999)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/energy_detail_all.png', dpi=150)
    plt.close()

    print(f"\nPlots saved:")
    print(f"  results/plots/ablation_all_levels.png")
    print(f"  results/plots/trajectory_all_levels.png")
    print(f"  results/plots/energy_detail_all.png")


if __name__ == "__main__":
    evaluate()
