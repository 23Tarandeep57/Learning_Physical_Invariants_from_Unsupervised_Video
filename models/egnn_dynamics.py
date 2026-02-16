"""
EGNN Dynamics Model — E(n) Equivariant Graph Neural Network

Predicts Δs (residual) per ball, with rotational equivariance.

Key principle: vectors NEVER enter MLPs.
Only scalars (distances, norms, dot products) are processed by learned functions.
Vectors only appear in geometric operations (weighted sums).

References: Satorras et al., "E(n) Equivariant Graph Neural Networks" (2021)
"""

import torch
import torch.nn as nn
import physics as P


class EGNNLayer(nn.Module):
    """
    One message-passing layer of the EGNN.
    
    Inputs per node:
        h_i:   (batch, n_balls, hidden_dim)  — scalar features
        pos_i: (batch, n_balls, 2)           — position vectors
        vel_i: (batch, n_balls, 2)           — velocity vectors
    
    Operations:
        1. Compute pairwise scalar features (distances, speed norms)
        2. Message MLP: scalars → scalars (equivariance-safe)
        3. Coordinate update: vectors × scalar weights (equivariant)
        4. Node update: aggregate scalar messages
    """
    
    def __init__(self, hidden_dim, act_fn=nn.SiLU):
        super().__init__()
        
        # === Message MLP (φ_e) ===
        # Input: h_i, h_j, ||pos_i - pos_j||², ||vel_i||², ||vel_j||²
        # That's: hidden + hidden + 1 + 1 + 1 = 2*hidden + 3
        edge_input_dim = 2 * hidden_dim + 3
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn()
        )

        # === Coordinate weight (φ_x) ===
        # Takes message, outputs a scalar weight per edge
        # This scalar multiplies the direction vector (pos_j - pos_i)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, 1)
        )
        
        # === Velocity weight (φ_v) ===
        # Same idea but for velocity updates
        self.vel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, 1)
        )

        # === Node update MLP (φ_h) ===
        # Input: h_i (current) + aggregated messages
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, h, pos, vel):
        """
        h:   (batch, N, hidden_dim)
        pos: (batch, N, 2)
        vel: (batch, N, 2)
        
        Returns updated (h', pos', vel')
        """
        batch, N, _ = h.shape
        
        # --- Step 1: Compute pairwise scalars ---
        # Relative position vectors: (batch, N, N, 2)
        # rel_pos[b, i, j] = pos[b, j] - pos[b, i]
        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(2)  # (B, N, N, 2)
        
        # Squared distance: (batch, N, N, 1) — a SCALAR, safe for MLP
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)  # (B, N, N, 1)
        
        # Speed squared per ball: (batch, N, 1)
        speed_sq = (vel ** 2).sum(dim=-1, keepdim=True)  # (B, N, 1)
        
        # Expand h for pairwise: (B, N, N, hidden)
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # row ball
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # col ball
        
        # Speed of each ball in the pair
        speed_i = speed_sq.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, 1)
        speed_j = speed_sq.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, 1)
        
        # --- Step 2: Message computation ---
        # Concatenate ALL SCALAR features: [h_i, h_j, dist², speed_i², speed_j²]
        edge_input = torch.cat([h_i, h_j, dist_sq, speed_i, speed_j], dim=-1)
        
        # φ_e: scalars → scalars (no vectors, equivariance preserved)
        m_ij = self.edge_mlp(edge_input)  # (B, N, N, hidden)
        
        # --- Step 3: Coordinate updates (equivariant) ---
        # Weight per edge: scalar
        coord_w = self.coord_mlp(m_ij)  # (B, N, N, 1)
        
        # Position update: Σ_j (pos_j - pos_i) * weight_ij
        # vector × scalar = vector (equivariant!)
        pos_update = (rel_pos * coord_w).sum(dim=2)  # (B, N, 2)
        
        # Velocity update: same structure
        rel_vel = vel.unsqueeze(1) - vel.unsqueeze(2)  # (B, N, N, 2)
        vel_w = self.vel_mlp(m_ij)  # (B, N, N, 1)
        vel_update = (rel_vel * vel_w).sum(dim=2)  # (B, N, 2)
        
        # --- Step 4: Node scalar update ---
        # Aggregate messages per node
        m_agg = m_ij.sum(dim=2)  # (B, N, hidden)
        
        # φ_h: [h_i, agg_messages] → h_i'
        h_new = self.node_mlp(torch.cat([h, m_agg], dim=-1))  # (B, N, hidden)
        
        # Apply updates
        pos_new = pos + pos_update
        vel_new = vel + vel_update
        
        return h_new, pos_new, vel_new


class EGNNDynamics(nn.Module):
    """
    Full EGNN dynamics model.
    
    Input:  state (batch, N, 4) = [x, y, vx, vy]
    Output: delta (batch, N, 4) = predicted residual
    """
    
    def __init__(self, n_balls=P.N_BALLS, hidden_dim=64, n_layers=2):
        super().__init__()
        self.n_balls = n_balls
        
        # Initial scalar embedding: (x, y, vx, vy) → we only keep SCALARS
        # for h_0. Speed magnitude, distance to walls, etc.
        # For simplicity: h_0 = learned embedding from nothing (or zeros)
        # Positions and velocities stay as vectors, never enter MLPs.
        self.h_init = nn.Linear(1, hidden_dim)  # from speed magnitude
        
        # Stack of EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim) for _ in range(n_layers)
        ])
        
        # Final readout: predict residual from scalar features
        # The vector part of the residual comes from pos_new - pos_old
        # and vel_new - vel_old (already equivariant)

    def forward(self, state):
        """
        state: (batch, N, 4) — [x, y, vx, vy]
        returns: (batch, N, 4) — predicted Δstate
        """
        batch, N, _ = state.shape
        
        # Split into vectors
        pos = state[:, :, :2]   # (B, N, 2) — position
        vel = state[:, :, 2:]   # (B, N, 2) — velocity
        
        # Initial scalar features: speed magnitude (rotation-invariant)
        speed = (vel ** 2).sum(dim=-1, keepdim=True).sqrt()  # (B, N, 1)
        h = self.h_init(speed)  # (B, N, hidden)
        
        # Store original for residual
        pos_0 = pos
        vel_0 = vel
        
        # Message passing
        for layer in self.layers:
            h, pos, vel = layer(h, pos, vel)
        
        # Residual: how much did EGNN shift pos and vel?
        delta_pos = pos - pos_0  # (B, N, 2)
        delta_vel = vel - vel_0  # (B, N, 2)
        
        # Combine into state residual
        delta = torch.cat([delta_pos, delta_vel], dim=-1)  # (B, N, 4)
        
        return delta
