from src.env import RockOrRollEnv
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

env = make_vec_env(RockOrRollEnv, n_envs=1, vec_env_cls=DummyVecEnv)
model = MaskablePPO("MlpPolicy", env, n_steps=64, verbose=0)
model.learn(total_timesteps=512)
print("training smoke test OK")