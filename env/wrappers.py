from __future__ import annotations
import gymnasium as gym
import numpy as np


class MultiDiscreteToBoxWrapper(gym.ActionWrapper):
    """
    Expose a MultiDiscrete action space as Box[-1, 1]^(H),
    where H = number of discrete heads. Each head i with n_i choices
    is mapped from y in [-1,1] to d in {0..n_i-1} via:
        d = round( (y + 1) / 2 * (n_i - 1) )
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete), \
            "This wrapper expects a MultiDiscrete action space."
        self.sizes = env.action_space.nvec.astype(np.int64)
        self.dim = int(self.sizes.size)  # one Box dim per discrete head
        low = -np.ones(self.dim, dtype=np.float32)
        high = np.ones(self.dim, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, act: np.ndarray):
        act = np.asarray(act, dtype=np.float32).ravel()
        if act.size != self.dim:
            reps = int(np.ceil(self.dim / max(1, act.size)))
            act = np.tile(act, reps)[:self.dim]
        out = []
        for i, n in enumerate(self.sizes):
            y = float(np.clip(act[i], -1.0, 1.0))
            d = int(np.round((y + 1.0) * 0.5 * (n - 1)))
            out.append(d)
        return np.array(out, dtype=np.int64)
