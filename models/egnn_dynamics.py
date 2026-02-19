"""
EGNN Dynamics — E(n) Equivariant Graph Neural Network.

Key invariant: vectors never enter MLPs; only scalars (distances, norms,
dot products) go through learned functions.

Ref: Satorras et al., "E(n) Equivariant Graph Neural Networks" (2021)
"""

import torch
import torch.nn as nn
import physics as P


class EGNNLayer(nn.Module):
    
    def __init__(self, hidden_dim, act_fn=nn.SiLU):
        super().__init__()

        edge_input_dim = 2 * hidden_dim + 3
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn()
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, 1)
        )

        self.vel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, 1)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, h, pos, vel):
        batch, N, _ = h.shape
        
        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)
        speed_sq = (vel ** 2).sum(dim=-1, keepdim=True)

        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
        speed_i = speed_sq.unsqueeze(2).expand(-1, -1, N, -1)
        speed_j = speed_sq.unsqueeze(1).expand(-1, N, -1, -1)
        
        # All scalar features — equivariance-safe
        edge_input = torch.cat([h_i, h_j, dist_sq, speed_i, speed_j], dim=-1)
        m_ij = self.edge_mlp(edge_input)
        
        # vector × scalar = equivariant coordinate updates
        coord_w = self.coord_mlp(m_ij)
        pos_update = (rel_pos * coord_w).sum(dim=2)

        rel_vel = vel.unsqueeze(1) - vel.unsqueeze(2)
        vel_w = self.vel_mlp(m_ij)
        vel_update = (rel_vel * vel_w).sum(dim=2)
        
        m_agg = m_ij.sum(dim=2)
        h_new = self.node_mlp(torch.cat([h, m_agg], dim=-1))
        
        pos_new = pos + pos_update
        vel_new = vel + vel_update
        
        return h_new, pos_new, vel_new


class EGNNDynamics(nn.Module):
    
    def __init__(self, n_balls=P.N_BALLS, hidden_dim=64, n_layers=2):
        super().__init__()
        self.n_balls = n_balls

        # Scalar embedding from speed magnitude (invariant)
        self.h_init = nn.Linear(1, hidden_dim)
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim) for _ in range(n_layers)
        ])

    def forward(self, state):
        batch, N, _ = state.shape

        pos = state[:, :, :2]
        vel = state[:, :, 2:]

        speed = (vel ** 2).sum(dim=-1, keepdim=True).sqrt()
        h = self.h_init(speed)

        pos_0 = pos
        vel_0 = vel
        
        for layer in self.layers:
            h, pos, vel = layer(h, pos, vel)

        delta_pos = pos - pos_0
        delta_vel = vel - vel_0
        return torch.cat([delta_pos, delta_vel], dim=-1)
