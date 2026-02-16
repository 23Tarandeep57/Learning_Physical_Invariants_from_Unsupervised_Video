"""
Train the EGNN dynamics model (Level 1: equivariant, no hard constraints).
Predicts state residuals Δs = s_{t+1} - s_t, same as MLP baseline.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import physics as P
from physics.engine import generate_dataset
from physics.dataset import PhysicsDataset
from models.egnn_dynamics import EGNNDynamics


def train():
    print("EGNN Training (Level 1: Equivariant)")
    dataset = generate_dataset(n_trajectories=500, n_steps=100)
    train_dataset = PhysicsDataset(dataset, mode='state_residual')
    print(f"Dataset shape: {train_dataset.X.shape}")
    print(f"Stats: X_mean={train_dataset.x_mean[:4]}, X_std={train_dataset.x_std[:4]}")

    # Normalization — EGNN operates on raw state (positions/velocities are geometric)
    # We normalize the TARGET residuals only, not the input state.
    # Reason: EGNN relies on distances/directions; normalizing positions breaks geometry.
    X_raw = torch.FloatTensor(train_dataset.X)  # unnormalized input
    Y_norm = (train_dataset.Y - train_dataset.y_mean) / train_dataset.y_std
    Y_tensor = torch.FloatTensor(Y_norm)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = EGNNDynamics(n_balls=P.N_BALLS, hidden_dim=64, n_layers=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    batch_size = 256
    n_epochs = 80
    best_loss = float('inf')

    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(X_raw.size(0))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, X_raw.size(0), batch_size):
            idx = perm[i:i + batch_size]
            batch_x = X_raw[idx].to(device)   # (B, N, 4) raw state
            batch_y = Y_tensor[idx].to(device) # (B, N, 4) normalized residual

            optimizer.zero_grad()
            pred_delta = model(batch_x)  # (B, N, 4) predicted residual (raw scale)

            # Normalize prediction to match target scale
            y_mean_t = torch.FloatTensor(train_dataset.y_mean).to(device)
            y_std_t = torch.FloatTensor(train_dataset.y_std).to(device)
            pred_norm = (pred_delta - y_mean_t) / y_std_t

            loss = criterion(pred_norm, batch_y)
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
            marker = " *"
        else:
            marker = ""

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs}, Loss: {avg_loss:.6f}, LR: {lr:.1e}{marker}")

    os.makedirs('results/checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'results/checkpoints/egnn_baseline.pth')
    stats = {
        'y_mean': train_dataset.y_mean,
        'y_std': train_dataset.y_std,
    }
    torch.save(stats, 'results/checkpoints/egnn_stats.pth')
    print(f"\nModel saved. Best loss: {best_loss:.6f}")
    print(f"results/checkpoints/egnn_baseline.pth")
    print(f"results/checkpoints/egnn_stats.pth")


if __name__ == "__main__":
    train()
