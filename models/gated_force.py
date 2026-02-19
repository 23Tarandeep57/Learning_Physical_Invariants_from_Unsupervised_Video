"""
Gated Force-EGNN — ForceEGNN with gate × magnitude factoring.

Merges two threads:
  1. ForceEGNN + Symplectic → -0.92% drift (force-based + integrator)
  2. Gate × Mag factoring → 0.977 median impulse ratio (classification ≠ regression)

Only the force layer changes: monolithic force_mlp → gate(dist) × mag(features).
Integrators and KE reused from force_egnn.
"""

import torch
import torch.nn as nn
import physics as P
from models.force_egnn import symplectic_euler_step, leapfrog_step, kinetic_energy


class GatedForceLayer(nn.Module):
    """ForceEGNNLayer but with factored force: gate(dist) × mag(dist, approach, h_i, h_j) × n̂."""

    def __init__(self, hidden_dim, temperature=1.0):
        super().__init__()
        self.temperature = temperature

        # Gate: dist only → contact detection (classification)
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Magnitude: dist + approach + hidden states → impulse scale (regression)
        self.mag_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, pos, vel):
        B, N, _ = h.shape
        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = (rel_pos ** 2).sum(-1, keepdim=True).add(1e-8).sqrt()
        normal = rel_pos / dist
        approach = ((vel.unsqueeze(1) - vel.unsqueeze(2)) * normal).sum(-1, keepdim=True)

        gate_logit = self.gate_mlp(dist)
        w = torch.sigmoid(gate_logit / self.temperature)

        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
        beta = self.mag_mlp(torch.cat([h_i, h_j, dist, approach], dim=-1))

        mask = (1.0 - torch.eye(N, device=h.device)).view(1, N, N, 1)
        force = (w * beta * normal * mask).sum(dim=2)

        m_agg = (w * beta * mask).sum(dim=2)
        a_agg = (approach.abs() * mask).sum(dim=2)
        h_new = self.node_mlp(torch.cat([h, m_agg, a_agg], dim=-1))

        return h_new, force, gate_logit * mask


class GatedForceNet(nn.Module):
    """ForceEGNN with gated pairwise forces. Outputs acceleration."""

    def __init__(self, n_balls=P.N_BALLS, hidden_dim=64, n_layers=2,
                 world_width=P.WORLD_WIDTH, world_height=P.WORLD_HEIGHT,
                 temperature=1.0):
        super().__init__()
        self.world_width = world_width
        self.world_height = world_height
        self.temperature = temperature

        self.h_init = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU())
        self.layers = nn.ModuleList([
            GatedForceLayer(hidden_dim, temperature) for _ in range(n_layers)
        ])
        # Wall: gate × magnitude (same factoring)
        self.wall_gate = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.wall_mag = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state):
        B, N, _ = state.shape
        pos, vel = state[:, :, :2], state[:, :, 2:]

        h = self.h_init((vel ** 2).sum(-1, keepdim=True).sqrt())
        total_force = torch.zeros_like(pos)
        for layer in self.layers:
            h, force, gate_logit = layer(h, pos, vel)
            total_force = total_force + force
        self._last_gate_logit = gate_logit  # last layer's gate for BCE

        # Gated wall forces
        wd = torch.stack([pos[:,:,0], self.world_width - pos[:,:,0],
                          pos[:,:,1], self.world_height - pos[:,:,1]], dim=-1)
        wv = torch.stack([-vel[:,:,0], vel[:,:,0], -vel[:,:,1], vel[:,:,1]], dim=-1)
        ww = torch.sigmoid(self.wall_gate(wd.unsqueeze(-1)) / self.temperature).squeeze(-1)
        wb = self.wall_mag(torch.stack([wd, wv], dim=-1)).squeeze(-1)
        wn = torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],
                          device=state.device, dtype=state.dtype)
        wall_force = ((ww * wb).unsqueeze(-1) * wn).sum(dim=-2)

        return total_force + wall_force
