"""
2D Physics Engine — deterministic elastic collision simulator.

- N balls in a rectangular container
- Elastic ball-ball & ball-wall collisions
- No gravity/friction (Phase 1), optional (Phase 4)
- State per ball: (x, y, vx, vy, radius, mass)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import physics as P
import copy


@dataclass
class Ball:
    """Physics-only state container. No appearance variables."""
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    mass: float
    ball_id: int = 0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy])

    @velocity.setter
    def velocity(self, v: np.ndarray):
        self.vx, self.vy = float(v[0]), float(v[1])

    @property
    def speed(self) -> float:
        return np.sqrt(self.vx**2 + self.vy**2)

    @property
    def state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy])

    @property
    def full_state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy, self.radius, self.mass])


@dataclass
class WorldConfig:
    width: float = P.WORLD_WIDTH
    height: float = P.WORLD_HEIGHT
    n_balls: int = P.N_BALLS
    dt: float = P.DT
    radius_range: Tuple[float, float] = P.RADIUS_RANGE
    speed_range: Tuple[float, float] = P.SPEED_RANGE
    mass_range: Tuple[float, float] = P.MASS_RANGE
    gravity: float = P.GRAVITY
    friction: float = P.FRICTION
    seed: Optional[int] = None


class PhysicsEngine:
    """
    Deterministic 2D elastic collision engine.

    Step: forces → velocity → position → wall collisions → ball collisions
    """

    def __init__(self, config: WorldConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.balls: List[Ball] = []
        self.time: float = 0.0
        self.collision_log: List[Dict] = []
        self.intervention_log: List[Dict] = []

    def initialize(self, balls: Optional[List[Ball]] = None) -> List[Ball]:
        if balls is not None:
            self.balls = [copy.deepcopy(b) for b in balls]
            self.time = 0.0
            self.collision_log = []
            self.intervention_log = []
            return self.balls

        self.balls = []
        for i in range(self.config.n_balls):
            ball = self._create_random_ball(i)
            for _ in range(100):
                if not self._overlaps_any(ball):
                    break
                ball = self._create_random_ball(i)
            self.balls.append(ball)

        self.time = 0.0
        self.collision_log = []
        self.intervention_log = []
        return self.balls

    def _create_random_ball(self, ball_id: int) -> Ball:
        r = self.rng.uniform(*self.config.radius_range)
        x = self.rng.uniform(r, self.config.width - r)
        y = self.rng.uniform(r, self.config.height - r)
        speed = self.rng.uniform(*self.config.speed_range)
        angle = self.rng.uniform(0, 2 * np.pi)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        mass = self.rng.uniform(*self.config.mass_range)
        return Ball(x=x, y=y, vx=vx, vy=vy, radius=r, mass=mass, ball_id=ball_id)

    def _overlaps_any(self, ball: Ball) -> bool:
        for other in self.balls:
            dx = ball.x - other.x
            dy = ball.y - other.y
            if np.sqrt(dx**2 + dy**2) < ball.radius + other.radius:
                return True
        return False

    def step(self, n_substeps: int = 4) -> List[Ball]:
        sub_dt = self.config.dt / n_substeps
        for _ in range(n_substeps):
            self._apply_forces(sub_dt)
            for ball in self.balls:
                ball.x += ball.vx * sub_dt
                ball.y += ball.vy * sub_dt
            self._resolve_wall_collisions()
            self._resolve_ball_collisions()
        self.time += self.config.dt
        return self.balls

    def _apply_forces(self, dt: float):
        for ball in self.balls:
            if self.config.gravity != 0:
                ball.vy -= self.config.gravity * dt
            if self.config.friction != 0:
                ball.vx *= (1 - self.config.friction * dt)
                ball.vy *= (1 - self.config.friction * dt)

    def _resolve_wall_collisions(self):
        for ball in self.balls:
            if ball.x - ball.radius < 0:
                ball.x = ball.radius
                ball.vx = abs(ball.vx)
            if ball.x + ball.radius > self.config.width:
                ball.x = self.config.width - ball.radius
                ball.vx = -abs(ball.vx)
            if ball.y - ball.radius < 0:
                ball.y = ball.radius
                ball.vy = abs(ball.vy)
            if ball.y + ball.radius > self.config.height:
                ball.y = self.config.height - ball.radius
                ball.vy = -abs(ball.vy)

    def _resolve_ball_collisions(self):
        """
        Elastic collision resolution.
        Convention: normal i→j, dv = vj-vi, resolve when dvn < 0.
        """
        n = len(self.balls)
        for i in range(n):
            for j in range(i + 1, n):
                bi, bj = self.balls[i], self.balls[j]
                dx = bj.x - bi.x
                dy = bj.y - bi.y
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = bi.radius + bj.radius

                if dist >= min_dist:
                    continue

                if dist < 1e-8:
                    dx = 1e-4 * (1 + bi.ball_id)
                    dy = 0.0
                    dist = abs(dx)

                nx, ny = dx / dist, dy / dist

                # Separate (mass-weighted)
                overlap = min_dist - dist
                inv_total = 1.0 / (bi.mass + bj.mass)
                bi.x -= nx * overlap * bj.mass * inv_total
                bi.y -= ny * overlap * bj.mass * inv_total
                bj.x += nx * overlap * bi.mass * inv_total
                bj.y += ny * overlap * bi.mass * inv_total

                # Relative velocity along normal
                dvn = (bj.vx - bi.vx) * nx + (bj.vy - bi.vy) * ny
                if dvn < 0:
                    impulse = (2.0 * dvn) / ((1.0 / bi.mass) + (1.0 / bj.mass))
                    bi.vx += impulse * nx / bi.mass
                    bi.vy += impulse * ny / bi.mass
                    bj.vx -= impulse * nx / bj.mass
                    bj.vy -= impulse * ny / bj.mass

                    self.collision_log.append({
                        'time': self.time, 'ball_i': bi.ball_id, 'ball_j': bj.ball_id,
                    })

    # State access

    def get_state(self) -> np.ndarray:
        """(n_balls, 4) → [x, y, vx, vy]"""
        return np.array([b.state for b in self.balls])

    def get_full_state(self) -> np.ndarray:
        """(n_balls, 6) → [x, y, vx, vy, radius, mass]"""
        return np.array([b.full_state for b in self.balls])

    def set_state(self, state: np.ndarray):
        assert state.shape == (len(self.balls), 4)
        for i, ball in enumerate(self.balls):
            ball.x, ball.y, ball.vx, ball.vy = state[i]

    # Interventions (do-operator)

    def remove_ball(self, ball_id: int):
        self.intervention_log.append({
            'time': self.time, 'type': 'remove', 'ball_id': ball_id,
        })
        self.balls = [b for b in self.balls if b.ball_id != ball_id]

    def freeze_ball(self, ball_id: int):
        self.intervention_log.append({
            'time': self.time, 'type': 'freeze', 'ball_id': ball_id,
        })
        for b in self.balls:
            if b.ball_id == ball_id:
                b.vx, b.vy = 0.0, 0.0

    def teleport_ball(self, ball_id: int, new_x: float, new_y: float,
                      preserve_momentum: bool = False):
        self.intervention_log.append({
            'time': self.time, 'type': 'teleport', 'ball_id': ball_id,
            'new_pos': (new_x, new_y),
        })
        for b in self.balls:
            if b.ball_id == ball_id:
                b.x, b.y = new_x, new_y

    # Conserved quantities

    def total_kinetic_energy(self) -> float:
        return sum(0.5 * b.mass * (b.vx**2 + b.vy**2) for b in self.balls)

    def total_momentum(self) -> np.ndarray:
        px = sum(b.mass * b.vx for b in self.balls)
        py = sum(b.mass * b.vy for b in self.balls)
        return np.array([px, py])

    def center_of_mass(self) -> np.ndarray:
        total_mass = sum(b.mass for b in self.balls)
        cx = sum(b.mass * b.x for b in self.balls) / total_mass
        cy = sum(b.mass * b.y for b in self.balls) / total_mass
        return np.array([cx, cy])

    def invariants(self) -> Dict[str, np.ndarray]:
        return {
            'energy': self.total_kinetic_energy(),
            'momentum': self.total_momentum(),
            'center_of_mass': self.center_of_mass(),
        }


def generate_trajectory(config: WorldConfig, n_steps: int = 200,
                        n_substeps: int = 4) -> Dict:
    """Returns dict with states, full_states, energy, momentum, com, collisions."""
    engine = PhysicsEngine(config)
    engine.initialize()

    states = [engine.get_state()]
    full_states = [engine.get_full_state()]
    energy = [engine.total_kinetic_energy()]
    momentum = [engine.total_momentum()]
    com = [engine.center_of_mass()]

    for _ in range(n_steps):
        engine.step(n_substeps=n_substeps)
        states.append(engine.get_state())
        full_states.append(engine.get_full_state())
        energy.append(engine.total_kinetic_energy())
        momentum.append(engine.total_momentum())
        com.append(engine.center_of_mass())

    return {
        'states': np.array(states),
        'full_states': np.array(full_states),
        'config': config,
        'collisions': engine.collision_log,
        'energy': np.array(energy),
        'momentum': np.array(momentum),
        'com': np.array(com),
    }


def generate_dataset(n_trajectories: int = P.N_TRAJECTORIES,
                     n_steps: int = P.N_STEPS,
                     n_balls: int = P.N_BALLS,
                     seed: int = P.SEED,
                     **kwargs) -> List[Dict]:
    trajectories = []
    for i in range(n_trajectories):
        config = WorldConfig(n_balls=n_balls, seed=seed + i, **kwargs)
        traj = generate_trajectory(config, n_steps=n_steps)
        trajectories.append(traj)
    return trajectories
