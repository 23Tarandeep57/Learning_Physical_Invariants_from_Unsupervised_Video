import torch
import torch.nn as nn
import physics as P

class ZeroFlow(nn.Module):
    def __init__(self, n_balls=P.N_BALLS):
        super().__init__()
        self.n = n_balls

    def forward(self, t, s):
        B = s.shape[0]
        s4 = s.view(B, self.n, 4)
        return torch.cat([s4[:, :, 2:], torch.zeros_like(s4[:, :, 2:])], dim=-1).view_as(s)


class JumpMap(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s, i, j):
        pos, vel = s[:, :, :2], s[:, :, 2:]
        diff = pos[:, j] - pos[:, i]
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        nhat = diff / dist

        dv = vel[:, j] - vel[:, i]
        app = (dv * nhat).sum(-1, keepdim=True)
        imp = self.net(torch.cat([dist, app], dim=-1)) * nhat

        v_new = vel.clone()
        v_new[:, i] += imp
        v_new[:, j] -= imp
        return torch.cat([pos, v_new], dim=-1)

class WallJumpMap(nn.Module):
    def forward(self, s, i, wn, wp, r):
        pos, vel = s[:, :, :2], s[:, :, 2:]
        v, p = vel[:, i], pos[:, i]
        
        vn = (v * wn).sum(-1, keepdim=True)
        vel_new = vel.clone()
        vel_new[:, i] = v - 2 * vn * wn
        
        pn = (p * wn).sum(-1, keepdim=True)
        pen = wp + r - pn
        pos_new = pos.clone()
        pos_new[:, i] = p + pen.clamp(min=0) * wn
        
        return torch.cat([pos_new, vel_new], dim=-1)


class HybridODENet(nn.Module):
    def __init__(self, n_balls=P.N_BALLS):
        super().__init__()
        self.flow = ZeroFlow(n_balls)
        self.jump = JumpMap()
        self.wall = WallJumpMap()
        
        self.n = n_balls
        self.world_width = P.WORLD_WIDTH
        self.world_height = P.WORLD_HEIGHT
        
        self.register_buffer('radii', torch.tensor([
            P.RADIUS_RANGE[0] for _ in range(n_balls)
        ]))

    def detect_events(self, state):
        pos = state[0, :, :2]
        vel = state[0, :, 2:]
        N = self.n
        radii = self.radii
        
        best = None
        min_gap = float('inf')
        

        for i in range(N):
            for j in range(i+1, N):
                diff = pos[j] - pos[i]
                gap = diff.norm().item() - (radii[i] + radii[j]).item()
                
                rel_vel = vel[j] - vel[i]
                nhat = diff / (diff.norm().item() + 1e-8)
                approach = (rel_vel * nhat).sum().item()
                
                if gap < min_gap and approach < 0:
                    min_gap = gap
                    best = ('ball', i, j, gap)
        

        for i in range(N):
            r = radii[i].item()
            checks = [
                (pos[i, 0].item() - r, 'left', i),
                (self.world_width - pos[i, 0].item() - r, 'right', i),
                (pos[i, 1].item() - r, 'bottom', i),
                (self.world_height - pos[i, 1].item() - r, 'top', i),
            ]
            for gap, wall_id, ball_idx in checks:
                if gap < min_gap:
                    if wall_id == 'left' and vel[i, 0].item() < 0:
                        min_gap = gap
                        best = ('wall', ball_idx, wall_id, gap)
                    elif wall_id == 'right' and vel[i, 0].item() > 0:
                        min_gap = gap
                        best = ('wall', ball_idx, wall_id, gap)
                    elif wall_id == 'bottom' and vel[i, 1].item() < 0:
                        min_gap = gap
                        best = ('wall', ball_idx, wall_id, gap)
                    elif wall_id == 'top' and vel[i, 1].item() > 0:
                        min_gap = gap
                        best = ('wall', ball_idx, wall_id, gap)
        
        return best
    
    def integrate_to_time(self, state, t_start, t_end):
        if t_end <= t_start + 1e-10:
            return state
        
        B, N, D = state.shape
        s_flat = state.view(B, -1)
        t_span = torch.tensor([t_start, t_end], dtype=state.dtype, device=state.device)
        
        s_out = odeint(self.flow, s_flat, t_span, method='dopri5', rtol=1e-5, atol=1e-6)
        return s_out[-1].view(B, N, D)
    
    def rollout(self, state, dt, n_steps, max_events_per_step=5):
        traj = [state]
        curr = state
        
        for step in range(n_steps):
            t_s = step * dt
            t_e = (step + 1) * dt
            t_c = t_s
            
            for _ in range(max_events_per_step):
                ev = self.detect_events(curr)
                if ev is None or ev[3] > 0.05: break
                
                if ev[3] <= 0:
                    if ev[0] == 'ball': curr = self.jump(curr, ev[1], ev[2])
                    else: curr = self.wall(curr, ev[1], self._get_wall_params(ev[2], curr.device)[0], self._get_wall_params(ev[2], curr.device)[1], self.radii[ev[1]])
                else:
                    gap = ev[3]
                    # Approach speed
                    if ev[0] == 'ball':
                        v = curr[0, :, 2:]
                        rv = v[ev[2]] - v[ev[1]]
                        n = (curr[0, ev[2], :2] - curr[0, ev[1], :2])
                        app = -(rv * n / n.norm()).sum().item()
                    else:
                        app = abs(curr[0, ev[1], 2:].max().item()) # Rough approx
                    
                    if app > 1e-6:
                        t_ev = t_c + gap / app
                        if t_ev < t_e:
                            curr = self.integrate_to_time(curr, t_c, t_ev)
                            t_c = t_ev
                            if ev[0] == 'ball': curr = self.jump(curr, ev[1], ev[2])
                            else: curr = self.wall(curr, ev[1], self._get_wall_params(ev[2], curr.device)[0], self._get_wall_params(ev[2], curr.device)[1], self.radii[ev[1]])
                        else: break
                    else: break
            
            curr = self.integrate_to_time(curr, t_c, t_e)
            traj.append(curr)
        
        return torch.stack(traj, dim=0)

    def _get_wall_params(self, wall_id, device):
        if wall_id == 'left': return torch.tensor([1., 0.], device=device), 0.0
        if wall_id == 'right': return torch.tensor([-1., 0.], device=device), -self.world_width
        if wall_id == 'bottom': return torch.tensor([0., 1.], device=device), 0.0
        if wall_id == 'top': return torch.tensor([0., -1.], device=device), -self.world_height

