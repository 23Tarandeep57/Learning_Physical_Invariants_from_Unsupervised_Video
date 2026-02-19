import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import physics as P
from physics.engine import generate_trajectory, WorldConfig
from physics.metrics import compute_energy, compute_momentum
from models.checkpoints import load_mlp, load_egnn
from models.rollout import rollout_mlp, rollout_egnn


def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_steps = 200
    n_test_seeds = 5  # average over multiple trajectories
    os.makedirs('results/plots', exist_ok=True)

    try:
        mlp_model, mlp_stats = load_mlp(device)
        print("Loaded MLP baseline.")
    except FileNotFoundError:
        print("MLP checkpoint not found. Run train_mlp.py first.")
        return

    try:
        egnn_model, egnn_stats = load_egnn(device)
        print("Loaded EGNN model.")
    except FileNotFoundError:
        print("EGNN checkpoint not found. Run train_egnn.py first.")
        return

    all_mse_mlp, all_mse_egnn = [], []
    all_energy_true, all_energy_mlp, all_energy_egnn = [], [], []
    all_mom_true, all_mom_mlp, all_mom_egnn = [], [], []

    for seed in range(900, 900 + n_test_seeds):
        config = WorldConfig(seed=seed)
        traj = generate_trajectory(config, n_steps=n_steps)
        true_states = traj['states']  # (T+1, N, 4)
        masses = traj['full_states'][0, :, 5]  # (N,)

        pred_mlp = rollout_mlp(mlp_model, mlp_stats, true_states[0], n_steps, device)
        pred_egnn = rollout_egnn(egnn_model, egnn_stats, true_states[0], n_steps, device)

        mse_mlp = np.mean((true_states - pred_mlp) ** 2, axis=(1, 2))
        mse_egnn = np.mean((true_states - pred_egnn) ** 2, axis=(1, 2))
        all_mse_mlp.append(mse_mlp)
        all_mse_egnn.append(mse_egnn)

        all_energy_true.append(compute_energy(true_states, masses))
        all_energy_mlp.append(compute_energy(pred_mlp, masses))
        all_energy_egnn.append(compute_energy(pred_egnn, masses))

        all_mom_true.append(compute_momentum(true_states, masses))
        all_mom_mlp.append(compute_momentum(pred_mlp, masses))
        all_mom_egnn.append(compute_momentum(pred_egnn, masses))

    # Average across seeds
    mse_mlp_avg = np.mean(all_mse_mlp, axis=0)
    mse_egnn_avg = np.mean(all_mse_egnn, axis=0)
    energy_true_avg = np.mean(all_energy_true, axis=0)
    energy_mlp_avg = np.mean(all_energy_mlp, axis=0)
    energy_egnn_avg = np.mean(all_energy_egnn, axis=0)
    mom_true_avg = np.mean(all_mom_true, axis=0)
    mom_mlp_avg = np.mean(all_mom_mlp, axis=0)
    mom_egnn_avg = np.mean(all_mom_egnn, axis=0)

    
    print("MLP (Level 0) vs EGNN (Level 1)")

    for step in [1, 10, 50, 100, 200]:
        print(f"\nStep {step:3d}:")
        print(f"  MSE    — MLP: {mse_mlp_avg[step]:.6f}  |  EGNN: {mse_egnn_avg[step]:.6f}")

    e0 = energy_true_avg[0]
    print(f"\nEnergy (true t=0: {e0:.4f}):")
    print(f"  MLP  final: {energy_mlp_avg[-1]:.4f}  (drift: {(energy_mlp_avg[-1] - e0)/e0*100:+.2f}%)")
    print(f"  EGNN final: {energy_egnn_avg[-1]:.4f}  (drift: {(energy_egnn_avg[-1] - e0)/e0*100:+.2f}%)")

    p0 = mom_true_avg[0]
    print(f"\nMomentum magnitude (true t=0: {p0:.4f}):")
    print(f"  MLP  final: {mom_mlp_avg[-1]:.4f}")
    print(f"  EGNN final: {mom_egnn_avg[-1]:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Level 0 (MLP) vs Level 1 (EGNN) Ablation', fontsize=14, fontweight='bold')
    steps = np.arange(n_steps + 1)

    ax = axes[0, 0]
    ax.semilogy(steps, mse_mlp_avg, label='MLP (Level 0)', color='#e74c3c', alpha=0.8)
    ax.semilogy(steps, mse_egnn_avg, label='EGNN (Level 1)', color='#2ecc71', alpha=0.8)
    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('MSE (log scale)')
    ax.set_title('Trajectory Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, energy_true_avg, label='Ground Truth', color='black', linewidth=2, alpha=0.5)
    ax.plot(steps, energy_mlp_avg, label='MLP', color='#e74c3c', linestyle='--', alpha=0.8)
    ax.plot(steps, energy_egnn_avg, label='EGNN', color='#2ecc71', linestyle='--', alpha=0.8)
    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('Total Kinetic Energy')
    ax.set_title('Energy Conservation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    e_err_mlp = (energy_mlp_avg - energy_true_avg) / (energy_true_avg + 1e-8) * 100
    e_err_egnn = (energy_egnn_avg - energy_true_avg) / (energy_true_avg + 1e-8) * 100
    ax.plot(steps, e_err_mlp, label='MLP', color='#e74c3c', alpha=0.8)
    ax.plot(steps, e_err_egnn, label='EGNN', color='#2ecc71', alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('Relative Energy Error (%)')
    ax.set_title('Energy Drift')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, mom_true_avg, label='Ground Truth', color='black', linewidth=2, alpha=0.5)
    ax.plot(steps, mom_mlp_avg, label='MLP', color='#e74c3c', linestyle='--', alpha=0.8)
    ax.plot(steps, mom_egnn_avg, label='EGNN', color='#2ecc71', linestyle='--', alpha=0.8)
    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('|Total Momentum|')
    ax.set_title('Momentum Conservation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/plots/ablation_mlp_vs_egnn.png', dpi=150)
    plt.close()

    #Single trajectory comparison plot 
    config = WorldConfig(seed=999)
    traj = generate_trajectory(config, n_steps=n_steps)
    true = traj['states']
    pred_m = rollout_mlp(mlp_model, mlp_stats, true[0], n_steps, device)
    pred_e = rollout_egnn(egnn_model, egnn_stats, true[0], n_steps, device)

    fig, axes = plt.subplots(P.N_BALLS, 2, figsize=(14, 4 * P.N_BALLS))
    if P.N_BALLS == 1:
        axes = axes[None, :]

    for b in range(P.N_BALLS):
        ax = axes[b, 0]
        ax.plot(true[:, b, 0], label='True', color='black', alpha=0.5)
        ax.plot(pred_m[:, b, 0], label='MLP', color='#e74c3c', linestyle='--', alpha=0.7)
        ax.plot(pred_e[:, b, 0], label='EGNN', color='#2ecc71', linestyle='--', alpha=0.7)
        ax.set_title(f'Ball {b} — X position')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[b, 1]
        ax.plot(true[:, b, 1], label='True', color='black', alpha=0.5)
        ax.plot(pred_m[:, b, 1], label='MLP', color='#e74c3c', linestyle='--', alpha=0.7)
        ax.plot(pred_e[:, b, 1], label='EGNN', color='#2ecc71', linestyle='--', alpha=0.7)
        ax.set_title(f'Ball {b} — Y position')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Single Trajectory Rollout (seed=999)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/plots/trajectory_mlp_vs_egnn.png', dpi=150)
    plt.close()

    print(f"\nPlots saved:")
    print(f"results/plots/ablation_mlp_vs_egnn.png")
    print(f"results/plots/trajectory_mlp_vs_egnn.png")


if __name__ == "__main__":
    evaluate()
