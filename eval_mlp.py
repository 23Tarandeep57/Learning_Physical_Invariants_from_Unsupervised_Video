import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import physics as P
from physics.engine import generate_trajectory, WorldConfig, PhysicsEngine
from models.mlp_dynamics import MLPDynamics

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPDynamics(n_balls=P.N_BALLS).to(device)
    
    try:
        model.load_state_dict(torch.load('results/checkpoints/mlp_baseline.pth', map_location=device))
        print("Loaded model.")
        stats = torch.load('results/checkpoints/mlp_stats.pth', weights_only=False)
        print("Loaded stats.")
    except FileNotFoundError:
        print("Model or stats not found. Run train_mlp.py first.")
        return

    # Extract stats
    x_mean = torch.FloatTensor(stats['x_mean']).to(device)
    x_std = torch.FloatTensor(stats['x_std']).to(device)
    y_mean = torch.FloatTensor(stats['y_mean']).to(device) # residual mean
    y_std = torch.FloatTensor(stats['y_std']).to(device)   # residual std

    model.eval()

    print("Generating test trajectory")
    config = WorldConfig(seed=999) # unseen seed
    traj = generate_trajectory(config, n_steps=100)
    
    true_states = traj['states']      # (T+1, N, 4)
    true_energy = traj['energy']
    
    # 3. Rollout Prediction
    # Start from t=0, predict t=1..T autoregressively
    pred_states = [true_states[0]]
    current_state_tensor = torch.FloatTensor(true_states[0]).unsqueeze(0).to(device) # (1, N, 4)
    
    with torch.no_grad():
        for _ in range(100):
            norm_input = (current_state_tensor - x_mean) / x_std
            norm_delta = model(norm_input)
            pred_delta = norm_delta * y_std + y_mean
            next_state_tensor = current_state_tensor + pred_delta
            current_state_tensor = next_state_tensor
            pred_states.append(next_state_tensor.cpu().numpy()[0])
            
    pred_states = np.array(pred_states) # (T+1, N, 4)
    
    # 4. Metrics
    # MSE over time
    mse = np.mean((true_states - pred_states)**2, axis=(1,2))
    print(f"MSE at step 1:   {mse[1]:.6f}")
    print(f"MSE at step 10:  {mse[10]:.6f}")
    print(f"MSE at step 50:  {mse[50]:.6f}")
    print(f"MSE at step 100: {mse[100]:.6f}")
    
    # Energy
    pred_energy = []
    for t in range(len(pred_states)):
        vs = pred_states[t, :, 2:] # (N, 2)
        e = 0.5 * 1.0 * np.sum(vs**2)
        pred_energy.append(e)
    pred_energy = np.array(pred_energy)
    
    # Plotting
    os.makedirs('results/plots', exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(true_states[:, 0, 0], label='True (Ball 0 x)', color='black', alpha=0.5)
    plt.plot(pred_states[:, 0, 0], label='Pred (Ball 0 x)', color='red', linestyle='--')
    plt.title('Experiment 3 (Res+Norm): Rollout')
    plt.legend()
    plt.savefig('results/plots/mlp_rollout_norm.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(true_energy, label='True Energy')
    plt.plot(pred_energy, label='Pred Energy')
    plt.title('Energy Conservation (Exp 3)')
    plt.legend()
    plt.savefig('results/plots/mlp_energy_norm.png')
    plt.close()
    
    print("Plots saved to results/plots/")

if __name__ == "__main__":
    evaluate()
