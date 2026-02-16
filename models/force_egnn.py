"""
Force-Predicting EGNN — Level 2: Equivariance + Force Structure + Symplectic Integration

Key differences from the state-residual EGNN (Level 1):
  1. Output is ACCELERATION (force/mass), not state residual
  2. Forces act along pair normals → hard-sphere collision bias
  3. State is advanced via symplectic integrator → energy preservation
  4. Wall forces handled separately (walls break translational symmetry)

The architecture directly addresses the Level 0→1 failure diagnosis:
  - State-residual models learn s_{t+1} = f(s_t) → no force law → energy drift
  - Force models learn a = F(s)/m, then integrate → physics-aligned pipeline

Ablation hierarchy:
  Level 0: MLP          → no structure
  Level 1: EGNN         → equivariance only, still learns state transitions
  Level 2: Force EGNN   → equivariance + force structure + symplectic integration
"""

import torch
import torch.nn as nn
import physics as P

class ForceEGNNLayer(nn.Module):
    """
    One message-passing layer that computes pairwise force contributions.

    Forces = scalar_weight × unit_normal.
    This is E(n)-equivariant: direction transforms under rotation,
    scalar weight is rotation-invariant.

    Critically, this encodes the NORMAL-ONLY impulse constraint:
    all forces between pairs act along the line connecting centers.
    No tangential component is possible by construction.
    """

    def __init__(self, hidden_dim, act_fn=nn.SiLU):
        super().__init__()

        # Edge MLP: all-scalar inputs → message
        # h_i, h_j, dist², speed²_i, speed²_j, approach_speed
        edge_input_dim = 2 * hidden_dim + 4

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn()
        )

        # Force magnitude: scalar message → scalar weight per edge
        # Sign determines attractive vs repulsive
        self.force_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, 1)
        )

        # Node update: aggregate scalar messages
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, pos, vel):
        """
        h:   (B, N, hidden)
        pos: (B, N, 2)
        vel: (B, N, 2)

        Returns: h_new (B, N, hidden), force (B, N, 2)
        """
        batch, N, _ = h.shape

        # Relative position: rel_pos[b,i,j] = pos_j - pos_i
        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(2)  # (B, N, N, 2)

        # Distance (scalar — safe for MLP)
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)  # (B, N, N, 1)
        dist = (dist_sq + 1e-8).sqrt()

        # Unit normal j→i
        normal = rel_pos / dist  # (B, N, N, 2)

        # Speed squared per ball (scalar)
        speed_sq = (vel ** 2).sum(dim=-1, keepdim=True)  # (B, N, 1)

        # Approach speed: (v_j - v_i) · n_ij
        # Negative = approaching, Positive = separating
        # This is the KEY feature for collision prediction
        rel_vel = vel.unsqueeze(1) - vel.unsqueeze(2)  # (B, N, N, 2)
        approach = (rel_vel * normal).sum(dim=-1, keepdim=True)  # (B, N, N, 1)

        # Expand for pairwise
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
        speed_i = speed_sq.unsqueeze(2).expand(-1, -1, N, -1)
        speed_j = speed_sq.unsqueeze(1).expand(-1, N, -1, -1)

        # Edge features: ALL scalars (equivariance-safe)
        edge_feat = torch.cat([h_i, h_j, dist_sq, speed_i, speed_j, approach], dim=-1)

        # Message
        m_ij = self.edge_mlp(edge_feat)  # (B, N, N, hidden)

        # Force weight per edge (scalar)
        force_weight = self.force_mlp(m_ij)  # (B, N, N, 1)

        # Force on i from j: weight × direction
        # This is NORMAL-ONLY by construction — no tangential leakage
        force_ij = force_weight * normal  # (B, N, N, 2)

        # Mask self-interactions (no self-force)
        mask = (1.0 - torch.eye(N, device=h.device)).unsqueeze(0).unsqueeze(-1)
        force_ij = force_ij * mask

        # Aggregate: total force per ball
        force = force_ij.sum(dim=2)  # (B, N, 2)

        # Node scalar update
        m_agg = (m_ij * mask).sum(dim=2)
        h_new = self.node_mlp(torch.cat([h, m_agg], dim=-1))

        return h_new, force


class ForceEGNN(nn.Module):
    """
    Full force-predicting model.

    Input:  state (B, N, 4) = [x, y, vx, vy]
    Output: acceleration (B, N, 2) per ball

    Components:
      - Ball-ball forces: EGNN pairwise message passing (normal-only)
      - Wall forces: distance-based with fixed normals (walls break symmetry)
    """

    def __init__(self, n_balls=P.N_BALLS, hidden_dim=64, n_layers=3,
                 world_width=P.WORLD_WIDTH, world_height=P.WORLD_HEIGHT):
        super().__init__()
        self.n_balls = n_balls
        self.world_width = world_width
        self.world_height = world_height

        # Node embedding from scalar features (speed magnitude)
        self.h_init = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU()
        )

        # EGNN force layers (ball-ball interactions)
        self.layers = nn.ModuleList([
            ForceEGNNLayer(hidden_dim) for _ in range(n_layers)
        ])

        # Wall force: scalar features → 4 force magnitudes (one per wall)
        # Direction is fixed by wall normal geometry (not learned)
        self.wall_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, state):
        """
        state: (B, N, 4) = [x, y, vx, vy]
        returns: acceleration (B, N, 2)
        """
        batch, N, _ = state.shape
        pos = state[:, :, :2]  # (B, N, 2)
        vel = state[:, :, 2:]  # (B, N, 2)

        # Scalar embedding: speed magnitude
        speed = (vel ** 2).sum(dim=-1, keepdim=True).sqrt()  # (B, N, 1)
        h = self.h_init(speed)

        # Ball-ball forces through stacked EGNN layers
        total_force = torch.zeros_like(pos)  # (B, N, 2)
        for layer in self.layers:
            h, force = layer(h, pos, vel)
            total_force = total_force + force

        # Wall forces
        wall_dists = torch.stack([
            pos[:, :, 0],                          # left wall
            self.world_width - pos[:, :, 0],       # right wall
            pos[:, :, 1],                          # bottom wall
            self.world_height - pos[:, :, 1],      # top wall
        ], dim=-1)  # (B, N, 4)

        wall_magnitudes = self.wall_mlp(
            torch.cat([h, wall_dists], dim=-1)
        )  # (B, N, 4)

        # Wall normals: left→(+x), right→(-x), bottom→(+y), top→(-y)
        wall_normals = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            dtype=state.dtype, device=state.device
        )  # (4, 2)

        # Wall force = magnitude × fixed normal direction
        wall_force = (wall_magnitudes.unsqueeze(-1) * wall_normals).sum(dim=-2)  # (B, N, 2)

        acceleration = total_force + wall_force
        return acceleration


# ═══════════════════════════════════════════════════════════════════
# Integrators
# ═══════════════════════════════════════════════════════════════════

def symplectic_euler_step(state, acc, dt):
    """
    Symplectic (semi-implicit) Euler integrator.

    v_{t+1} = v_t + a_t * dt          (kick)
    x_{t+1} = x_t + v_{t+1} * dt      (drift with NEW velocity)

    This is symplectic: preserves phase-space volume.
    Energy oscillates around true value instead of monotonic drift.
    """
    pos, vel = state[:, :, :2], state[:, :, 2:]
    new_vel = vel + acc * dt
    new_pos = pos + new_vel * dt
    return torch.cat([new_pos, new_vel], dim=-1)


def leapfrog_step(state, force_model, dt):
    """
    Velocity Verlet / Leapfrog integrator (2nd-order symplectic).
    Two force evaluations per step → more accurate.

    1. v_{1/2} = v + 0.5 * a(s) * dt
    2. x_new   = x + v_{1/2} * dt
    3. a_new   = F(x_new, v_{1/2})
    4. v_new   = v_{1/2} + 0.5 * a_new * dt
    """
    pos, vel = state[:, :, :2], state[:, :, 2:]

    acc = force_model(state)
    vel_half = vel + 0.5 * acc * dt
    new_pos = pos + vel_half * dt

    state_half = torch.cat([new_pos, vel_half], dim=-1)
    acc_new = force_model(state_half)
    new_vel = vel_half + 0.5 * acc_new * dt

    return torch.cat([new_pos, new_vel], dim=-1)


# ═══════════════════════════════════════════════════════════════════
# Physics quantities (differentiable, for loss computation)
# ═══════════════════════════════════════════════════════════════════

def kinetic_energy(state, masses=None):
    """Total KE per sample. state: (B, N, 4) → (B,)"""
    vel = state[:, :, 2:]
    if masses is not None:
        ke = 0.5 * masses.unsqueeze(-1) * vel ** 2
    else:
        ke = 0.5 * vel ** 2
    return ke.sum(dim=(1, 2))  # (B,)
