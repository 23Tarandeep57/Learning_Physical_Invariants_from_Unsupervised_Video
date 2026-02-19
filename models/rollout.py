import numpy as np
import torch
import physics as P
from models.force_egnn import symplectic_euler_step, leapfrog_step

def rollout_mlp(model, stats, init_state, n_steps, device):
    x_mean = torch.FloatTensor(stats['x_mean']).to(device)
    x_std = torch.FloatTensor(stats['x_std']).to(device)
    y_mean = torch.FloatTensor(stats['y_mean']).to(device)
    y_std = torch.FloatTensor(stats['y_std']).to(device)

    states = [init_state.copy()]
    current = torch.FloatTensor(init_state).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(n_steps):
            norm_in = (current - x_mean) / x_std
            norm_delta = model(norm_in)
            delta = norm_delta * y_std + y_mean
            current = current + delta
            states.append(current.cpu().numpy()[0])
    return np.array(states)


def rollout_egnn(model, stats, init_state, n_steps, device):
    states = [init_state.copy()]
    current = torch.FloatTensor(init_state).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(n_steps):
            pred_delta = model(current)
            current = current + pred_delta
            states.append(current.cpu().numpy()[0])
    return np.array(states)


def rollout_force(model, init_state, n_steps, device, dt=P.DT,
                  integrator='symplectic_euler'):
    """Force model rollout: predict acceleration → integrate.

    integrator: 'symplectic_euler' (default), 'euler', or 'leapfrog'
    """
    states = [init_state.copy()]
    current = torch.FloatTensor(init_state).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(n_steps):
            if integrator == 'leapfrog':
                current = leapfrog_step(current, model, dt)
            elif integrator == 'euler':
                acc = model(current)
                pos = current[:, :, :2]
                vel = current[:, :, 2:]
                new_pos = pos + vel * dt
                new_vel = vel + acc * dt
                current = torch.cat([new_pos, new_vel], dim=-1)
            else:  # symplectic_euler
                acc = model(current)
                current = symplectic_euler_step(current, acc, dt)
            states.append(current.cpu().numpy()[0])
    return np.array(states)
