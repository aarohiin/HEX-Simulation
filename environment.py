import gym
from gym import spaces
import numpy as np

class HexacopterEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.state = {'position': np.zeros(3), 'velocity': np.zeros(3), 'attitude': np.zeros(3)}
        self.dt = 0.02

    def reset(self):
        self.state = {'position': np.zeros(3), 'velocity': np.zeros(3), 'attitude': np.zeros(3)}
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.state['position'], self.state['velocity'], self.state['attitude']])

    def get_state(self):
        return self.state

    def step(self, action: np.ndarray):
        self.state['position'] += action[:3] * self.dt
        return self._get_obs(), 0.0, False, False, {}
