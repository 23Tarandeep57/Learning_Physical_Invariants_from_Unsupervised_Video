"""
Hybrid Neural ODE — Event-driven collision dynamics.

Architecture:
  Flow:  ds/dt = (v, f_θ(s))      — Neural ODE, f_θ should learn ≈ 0
  Event: g(s) = min_ij(d_ij - R_ij) = 0   — True geometric contact
  Jump:  v⁺ = v⁻ + α_ψ(d, v_approach) × n̂  — Learned impulse magnitude

No gate. No multiplicative degeneracy. No leakage.
The ODE integrator handles free-flight exactly.
The jump map handles collisions discretely.

This separates what previous models conflated:
  - Smooth dynamics (ODE) vs discrete events (jump)
  - Contact detection (geometry) vs impulse law (learned)
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint
import physics as P


class ZeroFlow(nn.Module):
    """
    Trivial flow: dq/dt = v, dv/dt = 0.
    
    For elastic billiards with no gravity/friction, this is EXACT.
    No parameters. No learning. No hallucinated forces.
    """
    def __init__(self, n_balls=P.N_BALLS):
        super().__init__()
        self.n_balls = n_balls
    
    def forward(self, t, s):
        B = s.shape[0]
        s4 = s.view(B, self.n_balls, 4)
        dq = s4[:, :, 2:]                              # velocity → dq/dt
        dv = torch.zeros_like(dq)                       # zero acceleration
        return torch.cat([dq, dv], dim=-1).view_as(s)


class FlowODE(nn.Module):
    """
    Continuous dynamics: dq/dt = v, dv/dt = f_θ(s).
    
    For elastic billiards with no gravity/friction, f_θ should learn ≈ 0.
    This is a sanity check: if f_θ is nonzero, the model is hallucinating forces.
    """
    def __init__(self, n_balls=P.N_BALLS, hidden_dim=32):
        super().__init__()
        # Tiny MLP — should converge to near-zero output
        self.net = nn.Sequential(
            nn.Linear(n_balls * 4, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, n_balls * 2),
        )
        self.n_balls = n_balls
        # Initialize near zero
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, t, s):
        """
        s: (B, N, 4) or (B, N*4)
        returns: ds/dt = (v, f_θ) same shape as s
        """
        B = s.shape[0]
        s_flat = s.view(B, -1)                        # (B, N*4)
        s4 = s.view(B, self.n_balls, 4)
        
        dq = s4[:, :, 2:]                             # velocity → dq/dt
        dv = self.net(s_flat).view(B, self.n_balls, 2) # learned acceleration
        
        return torch.cat([dq, dv], dim=-1).view_as(s)  # (B, N, 4)


class JumpMap(nn.Module):
    """
    Impulse law: v⁺ = v⁻ + α(dist, v_approach) × n̂
    
    Applied to the colliding pair only.
    α is a scalar learned from (distance, approach speed).
    No gate — the event detector decides WHEN; this decides HOW MUCH.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.magnitude_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, pair_i, pair_j):
        """
        state: (B, N, 4)
        pair_i, pair_j: indices of colliding pair (scalars, same for all B)
        
        Returns: state_after (B, N, 4) with velocities updated for the pair
        """
        pos = state[:, :, :2]
        vel = state[:, :, 2:]
        
        qi, qj = pos[:, pair_i], pos[:, pair_j]        # (B, 2)
        vi, vj = vel[:, pair_i], vel[:, pair_j]        # (B, 2)
        
        # Normal: j - i
        diff = qj - qi                                  # (B, 2)
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, 1)
        nhat = diff / dist                              # (B, 2)
        
        # Approach speed: (v_j - v_i) · n̂  (negative = approaching)
        rel_vel = vj - vi                               # (B, 2)
        approach = (rel_vel * nhat).sum(-1, keepdim=True)  # (B, 1)
        
        # Learned magnitude
        alpha = self.magnitude_mlp(torch.cat([dist, approach], dim=-1))  # (B, 1)
        
        # Apply impulse along normal (Newton's 3rd law by construction)
        impulse = alpha * nhat                           # (B, 2)
        
        new_vel = vel.clone()
        new_vel[:, pair_i] = vi + impulse
        new_vel[:, pair_j] = vj - impulse
        
        return torch.cat([pos, new_vel], dim=-1)


class WallJumpMap(nn.Module):
    """
    Wall collision: reflect velocity component perpendicular to wall.
    
    For walls, the physics is simple enough to hardcode:
    v_normal → -v_normal (perfect reflection).
    No learning needed — walls are not the research question.
    """
    def forward(self, state, ball_idx, wall_normal, wall_pos, radius):
        """
        state: (B, N, 4)
        ball_idx: which ball hit the wall
        wall_normal: (2,) inward-pointing normal
        wall_pos: scalar position of wall along normal direction
        radius: ball radius
        """
        pos = state[:, :, :2].clone()
        vel = state[:, :, 2:].clone()
        
        v = vel[:, ball_idx]                             # (B, 2)
        p = pos[:, ball_idx]                             # (B, 2)
        
        # Reflect velocity along wall normal
        vn = (v * wall_normal).sum(-1, keepdim=True)     # (B, 1)
        vel[:, ball_idx] = v - 2 * vn * wall_normal      # reflection
        
        # Push position out of wall
        pn = (p * wall_normal).sum(-1, keepdim=True)     # (B, 1)
        penetration = wall_pos + radius - pn              # (B, 1) how deep inside
        pos[:, ball_idx] = p + penetration.clamp(min=0) * wall_normal
        
        return torch.cat([pos, vel], dim=-1)


class HybridODENet(nn.Module):
    """
    Full hybrid model: Neural ODE flow + event-driven jumps.
    
    Integration loop:
      1. Detect next event time (ball-ball or ball-wall contact)
      2. Integrate ODE from current time to event time
      3. Apply jump map
      4. Repeat until target time
      
    Milestone 1: TRUE event detection (geometry-based).
    """
    def __init__(self, n_balls=P.N_BALLS, hidden_dim=32, jump_hidden=64,
                 world_width=P.WORLD_WIDTH, world_height=P.WORLD_HEIGHT,
                 radii=None):
        super().__init__()
        self.flow = FlowODE(n_balls=n_balls, hidden_dim=hidden_dim)
        self.jump = JumpMap(hidden_dim=jump_hidden)
        self.n_balls = n_balls
        self.world_width = world_width
        self.world_height = world_height
        self.radii = radii  # (N,) tensor — set per trajectory
        
        # Wall normals (inward-pointing) and positions
        # left: normal=(1,0), pos=0; right: normal=(-1,0), pos=-W
        # bottom: normal=(0,1), pos=0; top: normal=(0,-1), pos=-H
    
    def set_radii(self, radii):
        """Set ball radii for current trajectory batch. radii: (N,) tensor."""
        self.radii = radii
    
    def detect_events(self, state):
        """
        Find the earliest collision event from current state.
        
        Returns:
            event_type: 'ball' or 'wall' or None
            event_info: (i, j) for ball, (ball_idx, wall_id) for wall
            gap: distance to contact (negative = overlap)
        """
        pos = state[0, :, :2]  # (N, 2) — use first batch element
        vel = state[0, :, 2:]  # (N, 2)
        N = self.n_balls
        radii = self.radii
        
        best_event = None
        min_gap = float('inf')
        
        # Ball-ball
        for i in range(N):
            for j in range(i+1, N):
                diff = pos[j] - pos[i]
                dist = diff.norm().item()
                gap = dist - (radii[i] + radii[j]).item()
                
                # Check if approaching
                rel_vel = vel[j] - vel[i]
                nhat = diff / (dist + 1e-8)
                approach = (rel_vel * nhat).sum().item()
                
                if gap < min_gap and approach < 0:  # approaching
                    min_gap = gap
                    best_event = ('ball', i, j, gap)
        
        # Ball-wall
        for i in range(N):
            r = radii[i].item()
            checks = [
                (pos[i, 0].item() - r, 'left', i),        # left wall
                (self.world_width - pos[i, 0].item() - r, 'right', i),  # right
                (pos[i, 1].item() - r, 'bottom', i),      # bottom
                (self.world_height - pos[i, 1].item() - r, 'top', i),   # top
            ]
            for gap, wall_id, ball_idx in checks:
                if gap < min_gap:
                    # Check approaching
                    if wall_id == 'left' and vel[i, 0].item() < 0:
                        min_gap = gap
                        best_event = ('wall', ball_idx, wall_id, gap)
                    elif wall_id == 'right' and vel[i, 0].item() > 0:
                        min_gap = gap
                        best_event = ('wall', ball_idx, wall_id, gap)
                    elif wall_id == 'bottom' and vel[i, 1].item() < 0:
                        min_gap = gap
                        best_event = ('wall', ball_idx, wall_id, gap)
                    elif wall_id == 'top' and vel[i, 1].item() > 0:
                        min_gap = gap
                        best_event = ('wall', ball_idx, wall_id, gap)
        
        return best_event
    
    def integrate_to_time(self, state, t_start, t_end, n_eval=2):
        """
        Integrate the ODE from t_start to t_end.
        
        state: (B, N, 4)
        Returns: state at t_end
        """
        if t_end <= t_start + 1e-10:
            return state
        
        B, N, D = state.shape
        s_flat = state.view(B, -1)  # (B, N*4) for odeint
        
        t_span = torch.tensor([t_start, t_end], dtype=state.dtype, device=state.device)
        
        # Adaptive solver — handles free-flight exactly (linear motion)
        s_out = odeint(self.flow, s_flat, t_span, method='dopri5',
                       rtol=1e-5, atol=1e-6)  # (2, B, N*4)
        
        return s_out[-1].view(B, N, D)
    
    def apply_ball_jump(self, state, i, j):
        """Apply learned impulse to colliding pair (i, j)."""
        return self.jump(state, i, j)
    
    def apply_wall_jump(self, state, ball_idx, wall_id):
        """Apply wall reflection (hardcoded physics)."""
        device = state.device
        dtype = state.dtype
        r = self.radii[ball_idx].item()
        
        wall_info = {
            'left':   (torch.tensor([1., 0.], device=device, dtype=dtype), 0.0),
            'right':  (torch.tensor([-1., 0.], device=device, dtype=dtype), -self.world_width),
            'bottom': (torch.tensor([0., 1.], device=device, dtype=dtype), 0.0),
            'top':    (torch.tensor([0., -1.], device=device, dtype=dtype), -self.world_height),
        }
        normal, wall_pos = wall_info[wall_id]
        
        pos = state[:, :, :2].clone()
        vel = state[:, :, 2:].clone()
        
        v = vel[:, ball_idx]
        vn = (v * normal).sum(-1, keepdim=True)
        vel[:, ball_idx] = v - 2 * vn * normal
        
        # Position correction
        p = pos[:, ball_idx]
        pn = (p * normal).sum(-1, keepdim=True)
        pos[:, ball_idx] = p + (wall_pos + r - pn).clamp(min=0) * normal
        
        return torch.cat([pos, vel], dim=-1)
    
    def rollout(self, state, dt, n_steps, max_events_per_step=5):
        """
        Full hybrid rollout: ODE flow + event-driven jumps.
        
        state: (B, N, 4) initial state
        Returns: trajectory (n_steps+1, B, N, 4)
        """
        trajectory = [state]
        current = state
        
        for step in range(n_steps):
            t_start = step * dt
            t_end = (step + 1) * dt
            t_current = t_start
            
            for _ in range(max_events_per_step):
                event = self.detect_events(current)
                
                if event is None or event[3] > 0.05:
                    # No imminent event — integrate to end of step
                    break
                
                if event[3] <= 0:  # Contact or overlap
                    # Apply jump immediately
                    if event[0] == 'ball':
                        current = self.apply_ball_jump(current, event[1], event[2])
                    else:
                        current = self.apply_wall_jump(current, event[1], event[2])
                else:
                    # Integrate to near-contact, then jump
                    # Approximate event time from gap and approach speed
                    pos = current[0, :, :2]
                    vel = current[0, :, 2:]
                    gap = event[3]
                    
                    if event[0] == 'ball':
                        i, j = event[1], event[2]
                        rel_vel = vel[j] - vel[i]
                        diff = pos[j] - pos[i]
                        nhat = diff / diff.norm().clamp(min=1e-8)
                        approach_speed = -(rel_vel * nhat).sum().item()
                    else:
                        approach_speed = abs(vel[event[1]].max().item())
                    
                    if approach_speed > 1e-6:
                        t_event = t_current + gap / approach_speed
                        if t_event < t_end:
                            current = self.integrate_to_time(current, t_current, t_event)
                            t_current = t_event
                            if event[0] == 'ball':
                                current = self.apply_ball_jump(current, event[1], event[2])
                            else:
                                current = self.apply_wall_jump(current, event[1], event[2])
                        else:
                            break
                    else:
                        break
            
            # Integrate remaining time
            current = self.integrate_to_time(current, t_current, t_end)
            trajectory.append(current)
        
        return torch.stack(trajectory, dim=0)  # (T+1, B, N, 4)
