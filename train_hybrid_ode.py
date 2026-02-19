"""
Train Hybrid ODE — event-driven collision dynamics.

Milestone 1: True event detection + learned jump magnitude.
- Neural ODE for free-flight (should learn f_θ ≈ 0)
- Geometry-based collision detection (not learned)
- Learned impulse law α(dist, approach) × n̂

Training strategy:
- Short windows (5-20 steps) with teacher forcing at events
- Compare predicted states at fixed timestamps against ground truth
- Curriculum: start with 5-step windows, increase
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import physics as P
from physics.engine import generate_trajectory, WorldConfig
from models.hybrid_ode import HybridODENet, FlowODE, ZeroFlow, JumpMap


def build_collision_dataset(n_trajectories=200, n_steps=100):
    """
    Build dataset of (state_before, state_after, collision_info) around collision events.
    Each sample: a short window around a collision.
    
    For milestone 1 (teacher-forced), we extract:
    - State at collision timestep
    - True Δv at collision
    - Colliding pair info
    """
    trajs = []
    collision_samples = []
    free_flight_samples = []
    
    for seed in range(42, 42 + n_trajectories):
        traj = generate_trajectory(WorldConfig(seed=seed), n_steps=n_steps)
        states = traj['states']       # (T+1, N, 4)
        full_states = traj['full_states']  # (T+1, N, 6) includes radius, mass
        collisions = traj['collisions']
        dt = traj['config'].dt
        radii = full_states[0, :, 4]  # (N,) constant
        
        trajs.append({
            'states': states,
            'radii': radii,
            'collisions': collisions,
            'dt': dt,
        })
        
        # Extract collision events
        seen = set()
        for c in collisions:
            step = int(round(c['time'] / dt))
            key = (step, c['ball_i'], c['ball_j'])
            if key in seen or step < 1 or step >= len(states) - 1:
                continue
            seen.add(key)
            
            collision_samples.append({
                'state': states[step],          # (N, 4)
                'state_next': states[step + 1], # (N, 4)
                'radii': radii,
                'ball_i': c['ball_i'],
                'ball_j': c['ball_j'],
                'dv': states[step + 1, :, 2:] - states[step, :, 2:],  # (N, 2)
            })
        
        # Extract some free-flight samples (no collision)
        collision_steps = set()
        for c in collisions:
            collision_steps.add(int(round(c['time'] / dt)))
        
        for step in range(1, len(states) - 1):
            if step not in collision_steps and np.random.random() < 0.02:
                free_flight_samples.append({
                    'state': states[step],
                    'state_next': states[step + 1],
                    'radii': radii,
                    'dv': states[step + 1, :, 2:] - states[step, :, 2:],
                })
    
    print(f"Dataset: {n_trajectories} trajectories")
    print(f"  Collision samples: {len(collision_samples)}")
    print(f"  Free-flight samples: {len(free_flight_samples)}")
    
    return trajs, collision_samples, free_flight_samples


def train_jump_map():
    """
    Phase 1: Train ONLY the jump map on collision events (teacher-forced).
    
    At each collision, feed true pre-collision state, predict Δv.
    Compare against true Δv.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    trajs, collision_samples, free_flight_samples = build_collision_dataset(200, 100)
    
    # Jump map only
    jump = JumpMap(hidden_dim=64).to(device)
    n_params = sum(p.numel() for p in jump.parameters())
    print(f"JumpMap parameters: {n_params:,}")
    
    opt = optim.Adam(jump.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5)
    best_loss = float('inf')
    patience_ctr = 0
    
    for epoch in range(200):
        np.random.shuffle(collision_samples)
        epoch_loss = 0.
        n_batches = 0
        
        # Mini-batch over collision events
        batch_size = 64
        for i in range(0, len(collision_samples), batch_size):
            batch = collision_samples[i:i+batch_size]
            
            # Stack batch
            states = torch.FloatTensor(np.array([s['state'] for s in batch])).to(device)  # (B, N, 4)
            true_dv = torch.FloatTensor(np.array([s['dv'] for s in batch])).to(device)    # (B, N, 2)
            
            # Apply jump for each sample's colliding pair
            total_loss = torch.tensor(0., device=device)
            for k, sample in enumerate(batch):
                bi, bj = sample['ball_i'], sample['ball_j']
                s_k = states[k:k+1]  # (1, N, 4)
                
                s_after = jump(s_k, bi, bj)
                pred_dv = s_after[0, :, 2:] - s_k[0, :, 2:]  # (N, 2)
                true_dv_k = true_dv[k]                         # (N, 2)
                
                total_loss = total_loss + ((pred_dv - true_dv_k)**2).sum()
            
            loss = total_loss / len(batch)
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(jump.parameters(), 1.0)
            opt.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        sched.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_ctr = 0
            torch.save(jump.state_dict(), 'results/checkpoints/hybrid_jump.pth')
        else:
            patience_ctr += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}  loss={avg_loss:.6f}  lr={opt.param_groups[0]['lr']:.1e}"
                  f"{'  *' if patience_ctr==0 else ''}")
        
        if patience_ctr >= 25:
            print(f"Early stop at {epoch+1}")
            break
    
    print(f"Done. Best loss: {best_loss:.6f}")
    return jump


def train_flow():
    """
    Phase 2: Train the flow ODE on free-flight segments.
    
    Between collisions, dv/dt should be ≈ 0 (no forces).
    This verifies the ODE learns trivial dynamics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trajs, _, free_flight_samples = build_collision_dataset(200, 100)
    
    flow = FlowODE(n_balls=P.N_BALLS, hidden_dim=32).to(device)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"FlowODE parameters: {n_params:,}")
    
    opt = optim.Adam(flow.parameters(), lr=1e-3)
    dt = P.DT
    best_loss = float('inf')
    
    for epoch in range(50):
        np.random.shuffle(free_flight_samples)
        epoch_loss = 0.
        n_batches = 0
        batch_size = 128
        
        for i in range(0, len(free_flight_samples), batch_size):
            batch = free_flight_samples[i:i+batch_size]
            states = torch.FloatTensor(np.array([s['state'] for s in batch])).to(device)
            states_next = torch.FloatTensor(np.array([s['state_next'] for s in batch])).to(device)
            
            B, N, D = states.shape
            s_flat = states.view(B, -1)
            
            t_span = torch.tensor([0., dt], dtype=states.dtype, device=device)
            s_out = torch.stack([
                flow(t_span[0], s_flat[k:k+1])[0] for k in range(B)
            ])  # Use Euler step instead of odeint for speed during training
            
            # Simple Euler: s_{t+1} = s_t + ds/dt * dt
            ds = flow(torch.tensor(0.), s_flat)  # (B, N*4)
            pred_next = (s_flat + ds * dt).view(B, N, D)
            
            loss = ((pred_next - states_next)**2).mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(flow.state_dict(), 'results/checkpoints/hybrid_flow.pth')
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Flow Epoch {epoch+1:3d}  loss={avg_loss:.8f}")
    
    print(f"Flow done. Best={best_loss:.8f}")
    return flow


def diagnose():
    """
    Evaluate the hybrid model:
    1. Teacher-forced impulse accuracy (same as previous diagnostics)
    2. Short rollout energy conservation
    3. Flow ODE magnitude (should be ≈ 0)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load jump map
    jump = JumpMap(hidden_dim=64).to(device)
    jump.load_state_dict(torch.load('results/checkpoints/hybrid_jump.pth',
                                     map_location=device, weights_only=True))
    jump.eval()
    
    # Load flow (ZeroFlow — exact for no-gravity billiards)
    flow = ZeroFlow(n_balls=P.N_BALLS).to(device)
    flow.eval()
    
    dt = P.DT
    
    # ── 1. Impulse accuracy ──
    mag_ratios = []
    normal_errs = []
    tang_errs = []
    
    for seed in range(800, 900):
        traj = generate_trajectory(WorldConfig(seed=seed), n_steps=200)
        states = traj['states']
        full_states = traj['full_states']
        collisions = traj['collisions']
        
        seen = set()
        for c in collisions:
            step = int(round(c['time'] / dt))
            key = (step, c['ball_i'], c['ball_j'])
            if key in seen or step < 1 or step >= len(states) - 1:
                continue
            seen.add(key)
            
            st = torch.FloatTensor(states[step]).unsqueeze(0).to(device)
            with torch.no_grad():
                s_after = jump(st, c['ball_i'], c['ball_j'])
            dv_pred = (s_after[0, :, 2:] - st[0, :, 2:]).cpu().numpy()
            
            for b in [c['ball_i'], c['ball_j']]:
                dv_t = states[step+1, b, 2:] - states[step, b, 2:]
                dv_p = dv_pred[b]
                
                n_vec = states[step, c['ball_j'], :2] - states[step, c['ball_i'], :2]
                d = np.linalg.norm(n_vec)
                if d < 1e-8:
                    continue
                nhat = n_vec / d
                
                dn_t = np.dot(dv_t, nhat)
                dn_p = np.dot(dv_p, nhat)
                
                normal_errs.append(abs(dn_t - dn_p))
                tang_errs.append(np.linalg.norm(dv_p - dn_p * nhat))
                if abs(dn_t) > 1e-6:
                    mag_ratios.append(dn_p / dn_t)
    
    mag_ratios = np.array(mag_ratios)
    normal_errs = np.array(normal_errs)
    tang_errs = np.array(tang_errs)
    
    print(f"\n{'='*60}")
    print(f"HYBRID ODE — IMPULSE DIAGNOSTIC ({len(mag_ratios)//2} collisions)")
    print(f"{'='*60}")
    print(f"Magnitude ratio: mean={mag_ratios.mean():.3f} median={np.median(mag_ratios):.3f} "
          f"std={mag_ratios.std():.3f}")
    print(f"Normal error:    mean={normal_errs.mean():.4f}")
    print(f"Tangential err:  mean={tang_errs.mean():.6f} (should be ~0)")
    
    # ── 2. Short rollout with hybrid integration ──
    model = HybridODENet(n_balls=P.N_BALLS, hidden_dim=32, jump_hidden=64)
    model.flow = flow
    model.jump = jump
    model.eval()
    
    # Flow magnitude should be exactly 0 with ZeroFlow
    print(f"\nFlow type: {type(flow).__name__} (exact zero acceleration)")
    
    energies_true, energies_pred = [], []
    for seed in [900, 901, 902, 903, 904]:
        traj = generate_trajectory(WorldConfig(seed=seed), n_steps=200)
        true_s = traj['states']
        radii = torch.FloatTensor(traj['full_states'][0, :, 4])
        model.set_radii(radii)
        
        cur = torch.FloatTensor(true_s[0]).unsqueeze(0)
        
        # Manual rollout: detect events, integrate, jump
        pred_states = [true_s[0]]
        with torch.no_grad():
            for step in range(200):
                # Simple: integrate one dt with Euler (flow should be ~zero)
                B, N, D = cur.shape
                ds = flow(torch.tensor(0.), cur.view(B, -1))
                ds = ds.view(B, N, D)
                next_state = cur + ds * dt
                
                # Check for collisions in next_state
                pos = next_state[0, :, :2]
                for i in range(N):
                    for j in range(i+1, N):
                        diff = pos[j] - pos[i]
                        dist = diff.norm().item()
                        gap = dist - (radii[i] + radii[j]).item()
                        if gap < 0:
                            # Apply jump
                            next_state = jump(next_state, i, j)
                
                # Wall collisions (hardcoded reflection)
                pos = next_state[0, :, :2]
                vel = next_state[0, :, 2:]
                for i in range(N):
                    r = radii[i].item()
                    if pos[i, 0].item() < r:
                        next_state = next_state.clone()
                        next_state[0, i, 0] = r
                        next_state[0, i, 2] = abs(vel[i, 0].item())
                    if pos[i, 0].item() > model.world_width - r:
                        next_state = next_state.clone()
                        next_state[0, i, 0] = model.world_width - r
                        next_state[0, i, 2] = -abs(vel[i, 0].item())
                    if pos[i, 1].item() < r:
                        next_state = next_state.clone()
                        next_state[0, i, 1] = r
                        next_state[0, i, 3] = abs(vel[i, 1].item())
                    if pos[i, 1].item() > model.world_height - r:
                        next_state = next_state.clone()
                        next_state[0, i, 1] = model.world_height - r
                        next_state[0, i, 3] = -abs(vel[i, 1].item())
                
                cur = next_state
                pred_states.append(cur[0].numpy())
        
        pred_states = np.array(pred_states)
        energies_true.append(0.5 * (true_s[:, :, 2:]**2).sum(axis=(1, 2)))
        energies_pred.append(0.5 * (pred_states[:, :, 2:]**2).sum(axis=(1, 2)))
    
    et = np.mean(energies_true, axis=0)
    ep = np.mean(energies_pred, axis=0)
    drift = (ep[-1] - et[0]) / et[0] * 100
    
    print(f"\nEnergy drift (200-step rollout): {drift:+.1f}%")
    print(f"  True KE₀={et[0]:.3f}  Pred KE₂₀₀={ep[-1]:.3f}")
    
    # ── Plot ──
    os.makedirs('results/plots', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Hybrid Neural ODE — Diagnostics', fontsize=14, fontweight='bold')
    
    ax = axes[0]
    clipped = np.clip(mag_ratios, -2, 4)
    ax.hist(clipped, bins=40, color='#2ecc71', alpha=0.7, edgecolor='k', lw=0.5)
    ax.axvline(1.0, color='red', lw=2, label='perfect (1.0)')
    ax.axvline(mag_ratios.mean(), color='blue', lw=1.5, ls='--',
               label=f'mean ({mag_ratios.mean():.2f})')
    ax.set_xlabel('pred_n / true_n')
    ax.set_title('Impulse Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    steps = np.arange(len(et))
    ax.plot(steps, et, 'k-', lw=2, alpha=0.5, label='True')
    ax.plot(steps, ep, '--', color='#9b59b6', lw=1.5, label='Predicted')
    ax.set_title(f'Energy (drift: {drift:+.1f}%)')
    ax.set_xlabel('Step')
    ax.set_ylabel('KE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/hybrid_ode_diagnostic.png', dpi=150)
    plt.close()
    print(f"\nPlot: results/plots/hybrid_ode_diagnostic.png")


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 1: Train Jump Map (teacher-forced on collisions)")
    print("=" * 60)
    train_jump_map()
    
    print("\n" + "=" * 60)
    print("PHASE 2: Diagnostics (ZeroFlow — no flow training needed)")
    print("=" * 60)
    diagnose()
