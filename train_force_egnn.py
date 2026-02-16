"""
Train Force-EGNN — predict ACCELERATION, not state residuals.

CLEAN ABLATION: Changes ONLY the prediction target.
  - Same data generation as MLP/EGNN baselines (500 traj × 100 steps)
  - Single-step MSE on acceleration (no multi-step, no energy penalty)
  - Inputs stay unnormalized (EGNN needs raw geometry)
  - Targets normalized (acceleration has huge dynamic range)

This isolates ONE variable:
  Level 0 (MLP):  predict Δs = s_{t+1} - s_t        (state residual)
  Level 1 (EGNN): predict Δs = s_{t+1} - s_t        (state residual + equivariance)
  Level 2 (this): predict a  = (v_{t+1} - v_t) / dt  (acceleration + equivariance)

At eval, the SAME trained model is rolled out with:
  - Euler integrator      → tests if target alone fixes drift
  - Symplectic integrator → tests if integrator quality matters

Training subtlety:
  Free flight: a ≈ 0 (no forces). Collision: |a| >> 0 (impulse).
  ~99% of samples have near-zero acceleration.
  The model must learn sparse, event-driven forces — this is correct physics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import physics as P
from physics.engine import generate_dataset
from models.force_egnn import ForceEGNN


def build_acceleration_dataset(n_trajectories=500, n_steps=100):
    """
    Build (state, acceleration) pairs from trajectory data.

    a_t = (v_{t+1} - v_t) / dt

    Returns:
        X: (N_total, n_balls, 4)  — state [x, y, vx, vy]
        A: (N_total, n_balls, 2)  — acceleration [ax, ay]
    """
    print(f"Generating {n_trajectories} trajectories × {n_steps} steps...")
    trajectories = generate_dataset(
        n_trajectories=n_trajectories,
        n_steps=n_steps,
    )

    X_list, A_list = [], []
    collision_count = 0

    for traj in trajectories:
        states = traj['states']  # (T+1, N, 4)
        dt = traj['config'].dt
        collision_count += len(traj['collisions'])

        # State at t
        s_t = states[:-1]  # (T, N, 4)

        # Acceleration: (v_{t+1} - v_t) / dt
        v_t = states[:-1, :, 2:]   # (T, N, 2)
        v_tp1 = states[1:, :, 2:]  # (T, N, 2)
        a_t = (v_tp1 - v_t) / dt   # (T, N, 2)

        X_list.append(s_t)
        A_list.append(a_t)

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    A = np.concatenate(A_list, axis=0).astype(np.float32)

    # Diagnostics: how sparse are the forces?
    acc_norms = np.linalg.norm(A.reshape(-1, 2), axis=1)
    nonzero_mask = acc_norms > 1e-2
    pct_nonzero = nonzero_mask.mean() * 100

    print(f"Dataset: {X.shape[0]} samples, {collision_count} collisions logged")
    print(f"Acceleration sparsity: {100 - pct_nonzero:.1f}% near-zero (|a| < 0.01)")
    print(f"  mean |a| = {acc_norms.mean():.4f}")
    print(f"  max  |a| = {acc_norms.max():.2f}")
    print(f"  median   = {np.median(acc_norms):.6f}")

    return X, A


def train():
    print("=" * 60)
    print("Force-EGNN Training (Clean Ablation: Acceleration Target)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ── Data ──
    X, A = build_acceleration_dataset(n_trajectories=500, n_steps=100)

    # Normalization: targets only (acceleration has huge dynamic range)
    # Inputs stay raw — EGNN needs geometric distances/directions
    a_mean = np.mean(A, axis=0)  # (N, 2)
    a_std = np.std(A, axis=0) + 1e-8  # (N, 2)

    A_norm = (A - a_mean) / a_std

    X_tensor = torch.FloatTensor(X)      # (S, N, 4) raw state
    A_tensor = torch.FloatTensor(A_norm)  # (S, N, 2) normalized acceleration

    print(f"\nTarget normalization:")
    print(f"  a_mean: {a_mean.flatten()[:4]}")
    print(f"  a_std:  {a_std.flatten()[:4]}")

    # ── Model ──
    model = ForceEGNN(
        n_balls=P.N_BALLS,
        hidden_dim=64,
        n_layers=3,
        world_width=P.WORLD_WIDTH,
        world_height=P.WORLD_HEIGHT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5, min_lr=1e-5
    )
    # Event-aware reweighting: upweight rare collision impulses
    collision_lambda = 50.0
    collision_eps = 0.01  # |a| threshold for "collision event"

    batch_size = 256
    n_epochs = 100
    best_loss = float('inf')
    patience_counter = 0

    a_mean_t = torch.FloatTensor(a_mean).to(device)
    a_std_t = torch.FloatTensor(a_std).to(device)

    print(f"\nTraining: {n_epochs} epochs, batch={batch_size}")
    print(f"Loss: single-step MSE on normalized acceleration")
    print("-" * 55)

    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(X_tensor.size(0))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, X_tensor.size(0), batch_size):
            idx = perm[i:i + batch_size]
            batch_x = X_tensor[idx].to(device)   # (B, N, 4) raw state
            batch_a = A_tensor[idx].to(device)    # (B, N, 2) normalized accel

            optimizer.zero_grad()

            # Model predicts raw-scale acceleration
            pred_acc = model(batch_x)  # (B, N, 2)

            # Normalize prediction to match target
            pred_norm = (pred_acc - a_mean_t) / a_std_t

            # Event-aware reweighting: w = 1 + λ·1(|a| > ε)
            raw_a = batch_a * a_std_t + a_mean_t
            acc_mag = raw_a.norm(dim=-1, keepdim=True)  # (B, N, 1)
            w = 1.0 + collision_lambda * (acc_mag > collision_eps).float()
            loss = (w * (pred_norm - batch_a) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        scheduler.step(avg_loss)
        lr = optimizer.param_groups[0]['lr']

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'results/checkpoints/force_egnn.pth')
            marker = " * (saved)"
        else:
            patience_counter += 1
            marker = ""

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs}  "
                  f"Loss: {avg_loss:.6f}  LR: {lr:.1e}{marker}")

        if patience_counter >= 25:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # ── Save ──
    os.makedirs('results/checkpoints', exist_ok=True)
    stats = {
        'a_mean': a_mean,
        'a_std': a_std,
        'dt': P.DT,
    }
    torch.save(stats, 'results/checkpoints/force_egnn_stats.pth')

    print(f"\nDone. Best loss: {best_loss:.6f}")
    print(f"  results/checkpoints/force_egnn.pth")
    print(f"  results/checkpoints/force_egnn_stats.pth")


if __name__ == "__main__":
    train()
