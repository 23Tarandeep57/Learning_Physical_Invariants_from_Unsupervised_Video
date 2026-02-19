"""
Normal-Only Impulse Model — Minimal structural bias.

Output: Δv_i = Σ_{j≠i} α_ij(s) · n̂_ij  +  wall impulses
Where:
  n̂_ij = (q_j - q_i) / ||q_j - q_i||
  α_ij  = scalar from edge MLP (inputs: dist, approach speed)
  No free 2D vector output. No hidden node state. No stacking.
"""
import torch
import torch.nn as nn
import physics as P


class NormalImpulseNet(nn.Module):
    def __init__(self, n_balls=P.N_BALLS, hidden_dim=64,
                 world_width=P.WORLD_WIDTH, world_height=P.WORLD_HEIGHT,
                 temperature=1.0, hard_threshold=None):
        super().__init__()
        self.temperature = temperature
        self.hard_threshold = hard_threshold  # If set, gate becomes binary at inference
        # Gate: dist → logit (is this pair in contact?)
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Magnitude: (dist, approach_speed) → scalar β (how hard?)
        self.mag_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Wall: same gate × magnitude factoring
        self.wall_gate = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.wall_mag = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.world_width = world_width
        self.world_height = world_height

    def forward(self, state):
        """
        state: (B, N, 4) = [x, y, vx, vy]
        returns: Δv (B, N, 2) — velocity impulse per ball
        """
        pos = state[:, :, :2]  # (B, N, 2)
        vel = state[:, :, 2:]  # (B, N, 2)
        B, N, _ = pos.shape

        # ── Ball-ball impulses ──
        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(2)          # (B, N, N, 2)
        dist = (rel_pos ** 2).sum(-1, keepdim=True).add(1e-8).sqrt()  # (B, N, N, 1)
        n_ij = rel_pos / dist                                  # (B, N, N, 2)

        rel_vel = vel.unsqueeze(1) - vel.unsqueeze(2)          # (B, N, N, 2)
        approach = (rel_vel * n_ij).sum(-1, keepdim=True)      # (B, N, N, 1)

        # Gate × Magnitude factoring: α = sigmoid(gate/T) × β
        gate_logit = self.gate_mlp(dist)                                    # (B, N, N, 1)
        w = torch.sigmoid(gate_logit / self.temperature)                    # (B, N, N, 1)
        if self.hard_threshold is not None and not self.training:
            w = (w > self.hard_threshold).float()                           # binary gate at inference
        beta = self.mag_mlp(torch.cat([dist, approach], dim=-1))            # (B, N, N, 1)
        alpha = w * beta                                                    # (B, N, N, 1)

        # Mask self-interactions
        mask = (1.0 - torch.eye(N, device=state.device)).view(1, N, N, 1)
        dv_ball = (alpha * n_ij * mask).sum(dim=2)  # (B, N, 2)

        # ── Wall impulses (4 walls, fixed normals) ──
        # wall_dists: (B, N, 4), wall_vel: (B, N, 4) = vel component toward wall
        wall_dists = torch.stack([
            pos[:, :, 0],                          # left
            self.world_width - pos[:, :, 0],       # right
            pos[:, :, 1],                          # bottom
            self.world_height - pos[:, :, 1],      # top
        ], dim=-1)
        wall_vel = torch.stack([
            -vel[:, :, 0],  # left:   moving left = negative vx
             vel[:, :, 0],  # right:  moving right = positive vx
            -vel[:, :, 1],  # bottom: moving down = negative vy
             vel[:, :, 1],  # top:    moving up = positive vy
        ], dim=-1)

        wall_input = torch.stack([wall_dists, wall_vel], dim=-1)       # (B, N, 4, 2)
        wall_w = torch.sigmoid(self.wall_gate(wall_dists.unsqueeze(-1)) / self.temperature)  # (B, N, 4, 1)
        if self.hard_threshold is not None and not self.training:
            wall_w = (wall_w > self.hard_threshold).float()
        wall_beta = self.wall_mag(wall_input)                             # (B, N, 4, 1)
        wall_alpha = (wall_w * wall_beta).squeeze(-1)                     # (B, N, 4)

        wall_normals = torch.tensor(
            [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]],
            device=state.device, dtype=state.dtype)  # (4, 2)
        dv_wall = (wall_alpha.unsqueeze(-1) * wall_normals).sum(dim=-2)  # (B, N, 2)

        # Store gate logits for BCE loss (ball-ball only, masked)
        self._last_gate_logit = gate_logit * mask  # (B, N, N, 1)

        return dv_ball + dv_wall
