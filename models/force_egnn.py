"""
Force-Predicting EGNN — predicts acceleration (force/mass) instead of state
residuals.  Forces act along pair normals (hard-sphere bias); state is
advanced via a symplectic integrator for energy preservation.
"""

import torch
import torch.nn as nn
import physics as P

class ForceEGNNLayer(nn.Module):
    """Normal-only pairwise force layer (E(n)-equivariant)."""

    def __init__(self, hidden_dim, act_fn=nn.SiLU):
        super().__init__()

        # h_i, h_j, dist², speed²_i, speed²_j, approach_speed
        edge_input_dim = 2 * hidden_dim + 4

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn()
        )

        self.force_mlp = nn.Sequential(
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
        dist = (dist_sq + 1e-8).sqrt()
        normal = rel_pos / dist

        speed_sq = (vel ** 2).sum(dim=-1, keepdim=True)

        # Approach speed: (v_j - v_i) · n_ij (key feature for collision prediction)
        rel_vel = vel.unsqueeze(1) - vel.unsqueeze(2)
        approach = (rel_vel * normal).sum(dim=-1, keepdim=True)

        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
        speed_i = speed_sq.unsqueeze(2).expand(-1, -1, N, -1)
        speed_j = speed_sq.unsqueeze(1).expand(-1, N, -1, -1)

        edge_feat = torch.cat([h_i, h_j, dist_sq, speed_i, speed_j, approach], dim=-1)
        m_ij = self.edge_mlp(edge_feat)
        force_weight = self.force_mlp(m_ij)

        # Normal-only by construction — no tangential leakage
        force_ij = force_weight * normal
        mask = (1.0 - torch.eye(N, device=h.device)).unsqueeze(0).unsqueeze(-1)
        force_ij = force_ij * mask
        force = force_ij.sum(dim=2)

        m_agg = (m_ij * mask).sum(dim=2)
        h_new = self.node_mlp(torch.cat([h, m_agg], dim=-1))

        return h_new, force


class ForceEGNN(nn.Module):
    """Force-predicting model: input state (B,N,4) → acceleration (B,N,2)."""

    def __init__(self, n_balls=P.N_BALLS, hidden_dim=64, n_layers=3,
                 world_width=P.WORLD_WIDTH, world_height=P.WORLD_HEIGHT):
        super().__init__()
        self.n_balls = n_balls
        self.world_width = world_width
        self.world_height = world_height

        self.h_init = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU()
        )
        self.layers = nn.ModuleList([
            ForceEGNNLayer(hidden_dim) for _ in range(n_layers)
        ])

        # Wall force: scalar features → magnitudes; direction is fixed geometry
        self.wall_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, state):
        batch, N, _ = state.shape
        pos = state[:, :, :2]
        vel = state[:, :, 2:]

        # Scalar embedding: speed magnitude
        speed = (vel ** 2).sum(dim=-1, keepdim=True).sqrt()  # (B, N, 1)
        h = self.h_init(speed)

        # Ball-ball forces through stacked EGNN layers
        total_force = torch.zeros_like(pos)  # (B, N, 2)
        for layer in self.layers:
            h, force = layer(h, pos, vel)
            total_force = total_force + force

        wall_dists = torch.stack([
            pos[:, :, 0],                          # left wall
            self.world_width - pos[:, :, 0],       # right wall
            pos[:, :, 1],                          # bottom wall
            self.world_height - pos[:, :, 1],      # top wall
        ], dim=-1)

        wall_magnitudes = self.wall_mlp(
            torch.cat([h, wall_dists], dim=-1)
        )

        wall_normals = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            dtype=state.dtype, device=state.device
        )
        wall_force = (wall_magnitudes.unsqueeze(-1) * wall_normals).sum(dim=-2)

        acceleration = total_force + wall_force
        return acceleration


def symplectic_euler_step(state, acc, dt):
    """Symplectic (semi-implicit) Euler: v updated first, then x uses new v."""
    pos, vel = state[:, :, :2], state[:, :, 2:]
    new_vel = vel + acc * dt
    new_pos = pos + new_vel * dt
    return torch.cat([new_pos, new_vel], dim=-1)


def leapfrog_step(state, force_model, dt):
    """Velocity Verlet / Leapfrog (2nd-order symplectic, two force evals)."""
    pos, vel = state[:, :, :2], state[:, :, 2:]

    acc = force_model(state)
    vel_half = vel + 0.5 * acc * dt
    new_pos = pos + vel_half * dt

    state_half = torch.cat([new_pos, vel_half], dim=-1)
    acc_new = force_model(state_half)
    new_vel = vel_half + 0.5 * acc_new * dt

    return torch.cat([new_pos, new_vel], dim=-1)



def kinetic_energy(state, masses=None):
    """Total KE per sample: (B, N, 4) → (B,)"""
    vel = state[:, :, 2:]
    if masses is not None:
        ke = 0.5 * masses.unsqueeze(-1) * vel ** 2
    else:
        ke = 0.5 * vel ** 2
    return ke.sum(dim=(1, 2))  # (B,)
