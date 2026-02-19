import numpy as np
def compute_energy(states, masses=None):
    if masses is None:
        masses = np.ones(states.shape[1])
    vel = states[:, :, 2:]
    return (0.5 * masses[None, :, None] * vel ** 2).sum(axis=(1, 2))


def compute_momentum(states, masses=None):
    if masses is None:
        masses = np.ones(states.shape[1])
    vel = states[:, :, 2:]
    p = (masses[None, :, None] * vel).sum(axis=1)
    return np.linalg.norm(p, axis=1)
