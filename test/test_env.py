from rl.utils import load_config
from env.vec_env import EPTaskEnv

def test_env_step():
    cfg = load_config("configs/small.yaml")
    env = EPTaskEnv(cfg, seed=123)
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    env.close()
