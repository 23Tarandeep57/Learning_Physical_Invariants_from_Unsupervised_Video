"""
Quick demo — watch the balls bounce.
Run: venv/bin/python demo.py
Press Q or close window to exit.
"""
from physics.engine import generate_trajectory, WorldConfig
from physics.renderer import Renderer, AppearanceConfig
import physics as P

# Generate trajectory using centralized defaults
config = WorldConfig(seed=P.SEED)
traj = generate_trajectory(config, n_steps=P.N_STEPS)

print(f"Collisions: {len(traj['collisions'])}")
print(f"Energy drift: {abs(traj['energy'][-1] - traj['energy'][0]):.10f}")

# Render and play
renderer = Renderer(config.width, config.height, AppearanceConfig(seed=7))
renderer.play(traj, fps=60)
