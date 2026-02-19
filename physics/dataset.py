import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List

from physics.engine import generate_dataset

class PhysicsDataset(Dataset):
    def __init__(self, data: List[Dict], mode='state_residual'):
        self.mode = mode
        X_list = []
        Y_list = []

        for traj in data:
            states = traj['states']
            X = states[:-1]
            
            if mode == 'state_next':
                Y = states[1:]
            elif mode == 'state_residual':
                Y = states[1:] - states[:-1]
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
            X_list.append(X)
            Y_list.append(Y)
            
        self.X = np.concatenate(X_list, axis=0).astype(np.float32)
        self.Y = np.concatenate(Y_list, axis=0).astype(np.float32)

        self.x_mean = np.mean(self.X, axis=0)
        self.x_std = np.std(self.X, axis=0) + 1e-6

        self.y_mean = np.mean(self.Y, axis=0)
        self.y_std = np.std(self.Y, axis=0) + 1e-6

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def normalize(self, x: np.ndarray, is_target=False) -> np.ndarray:
        if is_target:
            return (x - self.y_mean) / self.y_std
        return (x - self.x_mean) / self.x_std

    def denormalize(self, x: np.ndarray, is_target=False) -> np.ndarray:
        if is_target:
            return x * self.y_std + self.y_mean
        return x * self.x_std + self.x_mean

    @staticmethod
    def from_config(n_trajectories=100, n_steps=100, **kwargs):
        raw_data = generate_dataset(n_trajectories=n_trajectories, n_steps=n_steps, **kwargs)
        return PhysicsDataset(raw_data)
