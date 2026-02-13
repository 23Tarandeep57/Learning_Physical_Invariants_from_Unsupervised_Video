import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import os

import physics as P

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame


@dataclass
class AppearanceConfig:
    """Nuisance variables — affect pixels, never physics."""
    resolution: int = P.RESOLUTION
    ball_colors: Optional[List[Tuple[int,int,int]]] = None
    outline: bool = False
    bg_color: Tuple[int,int,int] = P.BG_COLOR
    bg_noise_std: float = P.BG_NOISE_STD
    seed: Optional[int] = None


class Renderer:
    """Maps physics state → pixel frames."""

    def __init__(self, world_width: float, world_height: float,
                 config: Optional[AppearanceConfig] = None):
        self.world_w = world_width
        self.world_h = world_height
        self.config = config or AppearanceConfig()
        self.rng = np.random.RandomState(self.config.seed)
        self._display_initialized = False

    def _world_to_pixel(self, wx: float, wy: float) -> Tuple[int, int]:
        res = self.config.resolution
        px = int(wx / self.world_w * res)
        py = int((1.0 - wy / self.world_h) * res)
        return px, py

    def _world_radius_to_pixel(self, r: float) -> int:
        return max(1, int(r / self.world_w * self.config.resolution))

    def _get_colors(self, n_balls: int) -> List[Tuple[int,int,int]]:
        if self.config.ball_colors is not None:
            return self.config.ball_colors[:n_balls]
        return [tuple(self.rng.randint(80, 255, size=3).tolist())
                for _ in range(n_balls)]

    def render(self, state: np.ndarray, radii: np.ndarray,
               colors: Optional[List[Tuple[int,int,int]]] = None) -> np.ndarray:
        """Render single frame → (res, res, 3) uint8."""
        res = self.config.resolution
        n_balls = state.shape[0]
        if colors is None:
            colors = self._get_colors(n_balls)

        surface = pygame.Surface((res, res))
        surface.fill(self.config.bg_color)

        for i in range(n_balls):
            px, py = self._world_to_pixel(state[i, 0], state[i, 1])
            pr = self._world_radius_to_pixel(radii[i])
            if self.config.outline:
                pygame.draw.circle(surface, colors[i], (px, py), pr, 2)
            else:
                pygame.draw.circle(surface, colors[i], (px, py), pr)

        frame = pygame.surfarray.array3d(surface).transpose(1, 0, 2)

        if self.config.bg_noise_std > 0:
            noise = self.rng.normal(0, self.config.bg_noise_std, size=frame.shape)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return frame

    def render_trajectory(self, trajectory: Dict,
                          colors: Optional[List[Tuple[int,int,int]]] = None
                          ) -> np.ndarray:
        """Render full trajectory → (T+1, res, res, 3) uint8."""
        states = trajectory['states']
        radii = trajectory['full_states'][0, :, 4]
        T = states.shape[0]

        if colors is None:
            colors = self._get_colors(states.shape[1])

        frames = np.zeros((T, self.config.resolution, self.config.resolution, 3),
                          dtype=np.uint8)
        for t in range(T):
            frames[t] = self.render(states[t], radii, colors=colors)
        return frames

    def play(self, trajectory: Dict, fps: int = 30,
             colors: Optional[List[Tuple[int,int,int]]] = None):
        """Play trajectory in pygame window. Press Q to exit."""
        if not self._display_initialized:
            pygame.init()
            self._display_initialized = True

        res = self.config.resolution
        display_size = res
        screen = pygame.display.set_mode((display_size, display_size))
        pygame.display.set_caption('Physics Simulator')
        clock = pygame.time.Clock()

        states = trajectory['states']
        radii = trajectory['full_states'][0, :, 4]
        if colors is None:
            colors = self._get_colors(states.shape[1])

        running = True
        t = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False

            frame = self.render(states[t], radii, colors=colors)
            surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
            surf = pygame.transform.scale(surf, (display_size, display_size))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            t = (t + 1) % len(states)
            clock.tick(fps)

        pygame.quit()
        self._display_initialized = False


def save_frames_as_video(frames: np.ndarray, path: str, fps: int = 30):
    """Save frames as individual PNGs."""
    os.makedirs(path, exist_ok=True)
    for t in range(len(frames)):
        surf = pygame.surfarray.make_surface(frames[t].transpose(1, 0, 2))
        pygame.image.save(surf, os.path.join(path, f'frame_{t:05d}.png'))
