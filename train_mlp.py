import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import physics as P
from physics.engine import generate_dataset
from physics.dataset import PhysicsDataset
from models.mlp_dynamics import MLPDynamics

def train():
    print("Generating training data")
    dataset = generate_dataset(n_trajectories=500, n_steps=100)
    
    train_dataset = PhysicsDataset(dataset, mode='state_residual')
    print(f"Dataset shape: {train_dataset.X.shape}")
    print(f"Stats: X_mean={train_dataset.x_mean[:4]}, X_std={train_dataset.x_std[:4]}")

    X_norm = (train_dataset.X - train_dataset.x_mean) / train_dataset.x_std
    Y_norm = (train_dataset.Y - train_dataset.y_mean) / train_dataset.y_std
    
    X_tensor = torch.FloatTensor(X_norm)
    Y_tensor = torch.FloatTensor(Y_norm)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPDynamics(n_balls=P.N_BALLS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    batch_size = 256
    n_epochs = 50
    model.train()
    
    for epoch in range(n_epochs):
        perm = torch.randperm(X_tensor.size(0))
        epoch_loss = 0
        
        for i in range(0, X_tensor.size(0), batch_size):
            idx = perm[i:i+batch_size]
            batch_x = X_tensor[idx].to(device)
            batch_y = Y_tensor[idx].to(device)
            
            optimizer.zero_grad()
            pred_y = model(batch_x)
            loss = criterion(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(X_tensor):.6f}")
        
    os.makedirs('results/checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'results/checkpoints/mlp_baseline.pth')
    stats = {
        'x_mean': train_dataset.x_mean,
        'x_std': train_dataset.x_std,
        'y_mean': train_dataset.y_mean, # residual mean
        'y_std': train_dataset.y_std    # residual std
    }
    torch.save(stats, 'results/checkpoints/mlp_stats.pth')
    print("Model and stats saved.")

if __name__ == "__main__":
    train()
