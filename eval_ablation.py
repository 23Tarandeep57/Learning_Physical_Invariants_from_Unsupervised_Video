"""
3-Axis Factorized Ablation: What causes physics to be lost?

  Axis 1 — Prediction target:  state residual  vs  acceleration
  Axis 2 — Architecture:       MLP (global)    vs  EGNN (pairwise equivariant)
  Axis 3 — Integrator:         Euler           vs  Symplectic Euler

Configurations tested:
  ┌─────────────────────┬──────────────────┬────────────┬─────────────┐
  │ Label               │ Target           │ Arch       │ Integrator  │
  ├─────────────────────┼──────────────────┼────────────┼─────────────┤
  │ MLP (Level 0)       │ state residual   │ MLP        │ (implicit)  │
  │ EGNN (Level 1)      │ state residual   │ EGNN       │ (implicit)  │
  │ Force + Euler       │ acceleration     │ Force EGNN │ Euler       │
  │ Force + Symplectic  │ acceleration     │ Force EGNN │ Symplectic  │
  └─────────────────────┴──────────────────┴────────────┴─────────────┘

What each comparison isolates:
  Level 0 → Level 1:    Does equivariance help?
  Level 1 → Force+Euler: Does predicting acceleration (force law) fix drift?
  Force+Euler → Force+Symp: Does symplectic integration fix remaining drift?
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import physics as P
from physics.engine import generate_trajectory, WorldConfig
from models.mlp_dynamics import MLPDynamics
from models.egnn_dynamics import EGNNDynamics
from models.force_egnn import ForceEGNN, symplectic_euler_step


# ═══════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════

def load_mlp(device):
    model = MLPDynamics(n_balls=P.N_BALLS).to(device)
    model.load_state_dict(torch.load(
        'results/checkpoints/mlp_baseline.pth', map_location=device))
    stats = torch.load('results/checkpoints/mlp_stats.pth', weights_only=False)
    model.eval()
    return model, stats


def load_egnn(device):
    model = EGNNDynamics(n_balls=P.N_BALLS, hidden_dim=64, n_layers=3).to(device)
    model.load_state_dict(torch.load(
        'results/checkpoints/egnn_baseline.pth', map_location=device))
    stats = torch.load('results/checkpoints/egnn_stats.pth', weights_only=False)
    model.eval()
    return model, stats


def load_force_egnn(device):
    model = ForceEGNN(n_balls=P.N_BALLS, hidden_dim=64, n_layers=3).to(device)
    model.load_state_dict(torch.load(
        'results/checkpoints/force_egnn.pth', map_location=device))
    stats = torch.load('results/checkpoints/force_egnn_stats.pth', weights_only=False)
    model.eval()
    return model, stats


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
    states = [init_state.copy()]
    current = torch.FloatTensor(init_state).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(n_steps):
            pred_delta = model(current)
            current = current + pred_delta
            states.append(current.cpu().numpy()[0])
    return np.array(states)


def rollout_force_euler(model, init_state, n_steps, dt, device):
    """Force model + standard Euler integrator."""
    states = [init_state.copy()]
    current = torch.FloatTensor(init_state).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(n_steps):
            acc = model(current)  # (1, N, 2)
            pos = current[:, :, :2]
            vel = current[:, :, 2:]
            # Standard (explicit) Euler:
            #   x_{t+1} = x_t + v_t * dt
            #   v_{t+1} = v_t + a_t * dt
            new_pos = pos + vel * dt
            new_vel = vel + acc * dt
            current = torch.cat([new_pos, new_vel], dim=-1)
            states.append(current.cpu().numpy()[0])
    return np.array(states)


def rollout_force_symplectic(model, init_state, n_steps, dt, device):
    """Force model + symplectic Euler integrator."""
    states = [init_state.copy()]
    current = torch.FloatTensor(init_state).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(n_steps):
            acc = model(current)  # (1, N, 2)
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
    return (0.5 * masses[None, :, None] * vel ** 2).sum(axis=(1, 2))


def compute_momentum(states, masses=None):
    if masses is None:
        masses = np.ones(states.shape[1])
    vel = states[:, :, 2:]
    p = (masses[None, :, None] * vel).sum(axis=1)
    return np.linalg.norm(p, axis=1)


# ═══════════════════════════════════════════════════════════════════
# Main ablation
# ═══════════════════════════════════════════════════════════════════

COLORS = {
    'mlp':       '#e74c3c',  # red
    'egnn':      '#2ecc71',  # green
    'force_e':   '#3498db',  # blue
    'force_s':   '#9b59b6',  # purple
    'true':      '#2c3e50',  # dark gray
}

LABELS = {
    'mlp':       'MLP (Lv0: state res.)',
    'egnn':      'EGNN (Lv1: state res.)',
    'force_e':   'Force+Euler (Lv2a)',
    'force_s':   'Force+Symplectic (Lv2b)',
    'true':      'Ground Truth',
}


def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_steps = 200
    dt = P.DT
    n_seeds = 5
    os.makedirs('results/plots', exist_ok=True)

    # ── Load all models ──
    models = {}
    try:
        models['mlp'] = load_mlp(device)
        print("✓ MLP loaded")
    except Exception as e:
        print(f"✗ MLP: {e}")

    try:
        models['egnn'] = load_egnn(device)
        print("✓ EGNN loaded")
    except Exception as e:
        print(f"✗ EGNN: {e}")

    try:
        models['force'] = load_force_egnn(device)
        print("✓ Force-EGNN loaded")
    except Exception as e:
        print(f"✗ Force-EGNN: {e}")

    if 'force' not in models:
        print("\nForce-EGNN not found. Run train_force_egnn.py first.")
        return

    # ── Collect metrics across test seeds ──
    metrics = {k: {'mse': [], 'energy': [], 'momentum': []}
               for k in ['true', 'mlp', 'egnn', 'force_e', 'force_s']}

    for seed in range(900, 900 + n_seeds):
        config = WorldConfig(seed=seed)
        traj = generate_trajectory(config, n_steps=n_steps)
        true_states = traj['states']
        masses = traj['full_states'][0, :, 5]

        # Ground truth
        metrics['true']['energy'].append(compute_energy(true_states, masses))
        metrics['true']['momentum'].append(compute_momentum(true_states, masses))

        # MLP rollout
        if 'mlp' in models:
            m, s = models['mlp']
            pred = rollout_mlp(m, s, true_states[0], n_steps, device)
            metrics['mlp']['mse'].append(
                np.mean((true_states - pred) ** 2, axis=(1, 2)))
            metrics['mlp']['energy'].append(compute_energy(pred, masses))
            metrics['mlp']['momentum'].append(compute_momentum(pred, masses))

        # EGNN rollout
        if 'egnn' in models:
            m, s = models['egnn']
            pred = rollout_egnn(m, s, true_states[0], n_steps, device)
            metrics['egnn']['mse'].append(
                np.mean((true_states - pred) ** 2, axis=(1, 2)))
            metrics['egnn']['energy'].append(compute_energy(pred, masses))
            metrics['egnn']['momentum'].append(compute_momentum(pred, masses))

        # Force + Euler
        fm, fs = models['force']
        pred = rollout_force_euler(fm, true_states[0], n_steps, dt, device)
        metrics['force_e']['mse'].append(
            np.mean((true_states - pred) ** 2, axis=(1, 2)))
        metrics['force_e']['energy'].append(compute_energy(pred, masses))
        metrics['force_e']['momentum'].append(compute_momentum(pred, masses))

        # Force + Symplectic
        pred = rollout_force_symplectic(fm, true_states[0], n_steps, dt, device)
        metrics['force_s']['mse'].append(
            np.mean((true_states - pred) ** 2, axis=(1, 2)))
        metrics['force_s']['energy'].append(compute_energy(pred, masses))
        metrics['force_s']['momentum'].append(compute_momentum(pred, masses))

    # Average across seeds
    avg = {}
    for key in metrics:
        avg[key] = {}
        for metric_name in metrics[key]:
            if len(metrics[key][metric_name]) > 0:
                avg[key][metric_name] = np.mean(metrics[key][metric_name], axis=0)

    # ══════════════════════════════════════════════════════════════
    # Print summary table
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("3-AXIS FACTORIZED ABLATION RESULTS")
    print("=" * 70)

    model_keys = [k for k in ['mlp', 'egnn', 'force_e', 'force_s']
                  if k in avg and 'mse' in avg[k]]

    print(f"\n{'Model':<30} {'MSE@10':>10} {'MSE@50':>10} {'MSE@200':>10}")
    print("-" * 65)
    for k in model_keys:
        mse = avg[k]['mse']
        print(f"{LABELS[k]:<30} {mse[10]:>10.6f} {mse[50]:>10.6f} {mse[200]:>10.6f}")

    e0 = avg['true']['energy'][0]
    print(f"\n{'Model':<30} {'E_final':>10} {'E_drift%':>10}")
    print("-" * 55)
    for k in model_keys:
        ef = avg[k]['energy'][-1]
        drift = (ef - e0) / e0 * 100
        print(f"{LABELS[k]:<30} {ef:>10.4f} {drift:>+10.2f}%")

    p0 = avg['true']['momentum'][0]
    print(f"\n{'Model':<30} {'|p|_final':>10}")
    print("-" * 45)
    print(f"{'Ground Truth':<30} {avg['true']['momentum'][-1]:>10.4f}")
    for k in model_keys:
        print(f"{LABELS[k]:<30} {avg[k]['momentum'][-1]:>10.4f}")

    # ══════════════════════════════════════════════════════════════
    # Ablation isolation analysis
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("WHAT EACH COMPARISON ISOLATES")
    print("=" * 70)

    if 'egnn' in avg and 'mse' in avg['egnn']:
        e_drift_egnn = abs(avg['egnn']['energy'][-1] - e0) / e0 * 100
        e_drift_fe = abs(avg['force_e']['energy'][-1] - e0) / e0 * 100
        e_drift_fs = abs(avg['force_s']['energy'][-1] - e0) / e0 * 100

        print(f"\n1. Target formulation (EGNN→Force+Euler):")
        print(f"   Energy drift: {e_drift_egnn:.1f}% → {e_drift_fe:.1f}%")
        if e_drift_fe < e_drift_egnn * 0.5:
            print(f"   → Predicting acceleration HELPS. Target was the problem.")
        elif e_drift_fe > e_drift_egnn * 1.5:
            print(f"   → Predicting acceleration HURTS. Architecture/loss is the issue.")
        else:
            print(f"   → Marginal difference. Need deeper structural change.")

        print(f"\n2. Integrator quality (Force+Euler→Force+Symplectic):")
        print(f"   Energy drift: {e_drift_fe:.1f}% → {e_drift_fs:.1f}%")
        if e_drift_fs < e_drift_fe * 0.5:
            print(f"   → Symplectic integration HELPS. Integrator error dominated.")
        elif e_drift_fs > e_drift_fe * 1.5:
            print(f"   → Symplectic doesn't help. Force prediction error dominates.")
        else:
            print(f"   → Marginal difference. Both integrators similar for this dt.")

    # ══════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════
    steps = np.arange(n_steps + 1)

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle('3-Axis Factorized Ablation: What Causes Physics to Be Lost?',
                 fontsize=15, fontweight='bold', y=0.98)

    # ── 1. Trajectory MSE ──
    ax = axes[0, 0]
    for k in model_keys:
        ax.semilogy(steps, avg[k]['mse'],
                     label=LABELS[k], color=COLORS[k], alpha=0.85, linewidth=1.5)
    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('MSE (log)')
    ax.set_title('Trajectory Error')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 2. Energy conservation ──
    ax = axes[0, 1]
    ax.plot(steps, avg['true']['energy'], label='Ground Truth',
            color=COLORS['true'], linewidth=2.5, alpha=0.4)
    for k in model_keys:
        ax.plot(steps, avg[k]['energy'],
                label=LABELS[k], color=COLORS[k], linestyle='--', alpha=0.85, linewidth=1.5)
    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('Total Kinetic Energy')
    ax.set_title('Energy Conservation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 3. Energy relative error ──
    ax = axes[0, 2]
    for k in model_keys:
        e_err = (avg[k]['energy'] - avg['true']['energy']) / (avg['true']['energy'] + 1e-8) * 100
        ax.plot(steps, e_err, label=LABELS[k], color=COLORS[k], alpha=0.85, linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('Energy Drift (%)')
    ax.set_title('Energy Error (Relative)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 4. Momentum conservation ──
    ax = axes[1, 0]
    ax.plot(steps, avg['true']['momentum'], label='Ground Truth',
            color=COLORS['true'], linewidth=2.5, alpha=0.4)
    for k in model_keys:
        ax.plot(steps, avg[k]['momentum'],
                label=LABELS[k], color=COLORS[k], linestyle='--', alpha=0.85, linewidth=1.5)
    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('|Total Momentum|')
    ax.set_title('Momentum Conservation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 5. Axis isolation: Target (EGNN vs Force+Euler) ──
    ax = axes[1, 1]
    if 'egnn' in avg and 'energy' in avg['egnn']:
        e_abs_egnn = np.abs(avg['egnn']['energy'] - avg['true']['energy']) / (avg['true']['energy'] + 1e-8) * 100
        e_abs_fe = np.abs(avg['force_e']['energy'] - avg['true']['energy']) / (avg['true']['energy'] + 1e-8) * 100
        ax.plot(steps, e_abs_egnn, label='EGNN (state residual)',
                color=COLORS['egnn'], linewidth=1.5)
        ax.plot(steps, e_abs_fe, label='Force+Euler (acceleration)',
                color=COLORS['force_e'], linewidth=1.5)
        ax.fill_between(steps,
                        np.minimum(e_abs_egnn, e_abs_fe),
                        np.maximum(e_abs_egnn, e_abs_fe),
                        alpha=0.15, color='gray')
    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('|Energy Error| (%)')
    ax.set_title('Axis 1: Target Formulation\n(state residual vs acceleration)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 6. Axis isolation: Integrator (Euler vs Symplectic) ──
    ax = axes[1, 2]
    e_abs_fe = np.abs(avg['force_e']['energy'] - avg['true']['energy']) / (avg['true']['energy'] + 1e-8) * 100
    e_abs_fs = np.abs(avg['force_s']['energy'] - avg['true']['energy']) / (avg['true']['energy'] + 1e-8) * 100
    ax.plot(steps, e_abs_fe, label='Force + Euler',
            color=COLORS['force_e'], linewidth=1.5)
    ax.plot(steps, e_abs_fs, label='Force + Symplectic',
            color=COLORS['force_s'], linewidth=1.5)
    ax.fill_between(steps,
                    np.minimum(e_abs_fe, e_abs_fs),
                    np.maximum(e_abs_fe, e_abs_fs),
                    alpha=0.15, color='gray')
    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('|Energy Error| (%)')
    ax.set_title('Axis 3: Integrator Quality\n(Euler vs Symplectic)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('results/plots/ablation_3axis.png', dpi=150)
    plt.close()

    # ── Single trajectory comparison ──
    config = WorldConfig(seed=999)
    traj = generate_trajectory(config, n_steps=n_steps)
    true = traj['states']
    fm, fs = models['force']

    rollouts = {'true': true}
    if 'mlp' in models:
        m, s = models['mlp']
        rollouts['mlp'] = rollout_mlp(m, s, true[0], n_steps, device)
    if 'egnn' in models:
        m, s = models['egnn']
        rollouts['egnn'] = rollout_egnn(m, s, true[0], n_steps, device)
    rollouts['force_e'] = rollout_force_euler(fm, true[0], n_steps, dt, device)
    rollouts['force_s'] = rollout_force_symplectic(fm, true[0], n_steps, dt, device)

    fig, axes = plt.subplots(P.N_BALLS, 2, figsize=(16, 4.5 * P.N_BALLS))
    if P.N_BALLS == 1:
        axes = axes[None, :]

    for b in range(P.N_BALLS):
        for col, coord, name in [(0, 0, 'X'), (1, 1, 'Y')]:
            ax = axes[b, col]
            ax.plot(true[:, b, coord], label='True',
                    color=COLORS['true'], linewidth=2, alpha=0.4)
            for k in ['mlp', 'egnn', 'force_e', 'force_s']:
                if k in rollouts:
                    ax.plot(rollouts[k][:, b, coord], label=LABELS[k],
                            color=COLORS[k], linestyle='--', alpha=0.75, linewidth=1.2)
            ax.set_title(f'Ball {b} — {name} position', fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Single Trajectory Rollout (seed=999) — All Models',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/plots/trajectory_3axis.png', dpi=150)
    plt.close()

    print(f"\nPlots saved:")
    print(f"  results/plots/ablation_3axis.png")
    print(f"  results/plots/trajectory_3axis.png")


if __name__ == "__main__":
    evaluate()
