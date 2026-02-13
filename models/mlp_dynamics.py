import torch
import torch.nn as nn
import physics as P

class MLPDynamics(nn.Module):
    def __init__(self, n_balls=P.N_BALLS, hidden_dim=256):
        super().__init__()
        self.n_balls = n_balls
        input_dim = n_balls * 4
        output_dim = n_balls * 4

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, state):
        batch_size = state.shape[0]
        x = state.view(batch_size, -1)
        out = self.net(x)
        return out.view(batch_size, self.n_balls, 4)
