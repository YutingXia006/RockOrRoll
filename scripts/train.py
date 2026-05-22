import os
import mlflow
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_util import make_vec_env
from src.env import RockOrRollEnv
import numpy as np

# --- Config ---
TIMESTEPS = 507_904
MODEL_PATH = "models/ppo_rockorroll"
LOG_PATH = "logs/"
EXPERIMENT_NAME = "RockOrRoll"

HP = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "n_envs": 4,
    "timesteps": TIMESTEPS,
}

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# --- MLflow ---
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="PPO_baseline"):
    mlflow.log_params(HP)

    # --- Environment ---
    env = make_vec_env(RockOrRollEnv, n_envs=HP["n_envs"])

    # --- Model ---
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=HP["learning_rate"],
        n_steps=HP["n_steps"],
        batch_size=HP["batch_size"],
        gamma=HP["gamma"],
        tensorboard_log=LOG_PATH,
    )

    # --- Train ---
    print("=== Starting Training ===")
    try:
        model.learn(total_timesteps=HP["timesteps"])
    except KeyboardInterrupt:
        print("\nUnterbrochen — speichere...")
    finally:
        save_path = os.path.join(MODEL_PATH, "final_model")
        model.save(save_path)
        print(f"✓ Model gespeichert: {save_path}.zip")

    # --- Evaluate ---
    eval_env = RockOrRollEnv()
    wins = 0
    total_rewards = []
    total_health = []

    for _ in range(20):
        obs, _ = eval_env.reset()
        terminated = truncated = False
        ep_reward = 0
        while not terminated and not truncated:
            action_masks = np.array(eval_env.action_masks())
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_reward += reward
        if eval_env.game.round > 43:
            wins += 1
        total_rewards.append(ep_reward)
        total_health.append(max(eval_env.game.health, 0))

    win_rate = wins / 20
    avg_reward = sum(total_rewards) / 20
    avg_health = sum(total_health) / 20

    # --- Log Metrics ---
    mlflow.log_metric("win_rate", win_rate)
    mlflow.log_metric("avg_reward", avg_reward)
    mlflow.log_metric("avg_health", avg_health)
    mlflow.log_artifact(save_path + ".zip")

    print(f"\n=== MLflow Run Complete ===")
    print(f"Win Rate:   {win_rate*100:.0f}%")
    print(f"Avg Reward: {avg_reward:.3f}")
    print(f"Avg Health: {avg_health:.1f}")
