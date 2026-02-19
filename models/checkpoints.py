import os
import torch
import physics as P
from models.mlp_dynamics import MLPDynamics
from models.egnn_dynamics import EGNNDynamics
from models.force_egnn import ForceEGNN


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
