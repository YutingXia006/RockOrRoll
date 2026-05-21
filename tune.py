import os
import mlflow
import optuna
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_util import make_vec_env
from env import RockOrRollEnv
import numpy as np

# --- Config ---
N_TRIALS = 20        # Anzahl Optuna Trials
TIMESTEPS = 507_904
MODEL_PATH = "models/ppo_rockorroll"
EXPERIMENT_NAME = "RockOrRoll"

os.makedirs(MODEL_PATH, exist_ok=True)

mlflow.set_experiment(EXPERIMENT_NAME)

def evaluate_model(model, n_episodes=20):
    eval_env = RockOrRollEnv()
    wins = 0
    total_rewards = []
    total_health = []

    for _ in range(n_episodes):
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

    return {
        "win_rate": wins / n_episodes,
        "avg_reward": sum(total_rewards) / n_episodes,
        "avg_health": sum(total_health) / n_episodes,
    }

def objective(trial):
    # --- Hyperparameter Search Space ---
    hp = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "gamma": trial.suggest_float("gamma", 0.90, 0.999),
        "n_envs": 4,
    }

    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        mlflow.log_params(hp)

        try:
            rollout_buffer = hp["n_steps"] *  hp["n_envs"]
            if hp["batch_size"] > rollout_buffer:
                raise optuna.exceptions.TrialPruned()
            total_timesteps = (TIMESTEPS // rollout_buffer) * rollout_buffer 
            env = make_vec_env(RockOrRollEnv, n_envs=hp["n_envs"])
            model = MaskablePPO(
                "MlpPolicy",
                env,
                verbose=0,
                learning_rate=hp["learning_rate"],
                n_steps=hp["n_steps"],
                batch_size=hp["batch_size"],
                gamma=hp["gamma"],
            )
            model.learn(total_timesteps=total_timesteps)

            metrics = evaluate_model(model)
            mlflow.log_metrics(metrics)

            # Bestes Modell speichern
            save_path = os.path.join(MODEL_PATH, f"trial_{trial.number}")
            model.save(save_path)
            mlflow.log_artifact(save_path + ".zip")

            print(f"Trial {trial.number}: win_rate={metrics['win_rate']:.2f} | "
                  f"avg_reward={metrics['avg_reward']:.3f} | "
                  f"lr={hp['learning_rate']:.6f} | "
                  f"n_steps={hp['n_steps']} | "
                  f"batch_size={hp['batch_size']} | "
                  f"gamma={hp['gamma']:.3f}")
            print(f"DEBUG - Eval Reward: {metrics['avg_reward']}")
            return metrics["avg_reward"]

        except Exception as e:
            print(f"Trial {trial.number} fehlgeschlagen: {e}")
            mlflow.log_metric("win_rate", 0)
            mlflow.log_metric("avg_reward", 0)
            mlflow.log_metric("avg_health", 0)
            return 0.0

# --- Run Optuna ---
study = optuna.create_study(
    direction="maximize",
    storage="sqlite:///optuna.db",  # speichert in Datei
    study_name="RockOrRoll",
    load_if_exists=True
)
study.optimize(objective, n_trials=N_TRIALS)

# --- Beste Parameter ---
print(f"\n{'='*50}")
print(f"BESTE PARAMETER:")
print(f"{'='*50}")
for key, value in study.best_params.items():
    print(f"{key}: {value}")
print(f"Bester Reward: {study.best_value:.3f}")

# --- Bestes Modell nochmal mit mehr Steps trainieren ---
print(f"\n{'='*50}")
print("Trainiere bestes Modell mit vollen Steps...")
print(f"{'='*50}")

best_hp = study.best_params
best_hp["n_envs"] = 4
best_rollout_buffer = best_hp["n_steps"] * best_hp["n_envs"]
best_total_timesteps = (TIMESTEPS // best_rollout_buffer) * best_rollout_buffer 

with mlflow.start_run(run_name="PPO_best"):
    mlflow.log_params(best_hp)

    env = make_vec_env(RockOrRollEnv, n_envs=best_hp["n_envs"])
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=best_hp["learning_rate"],
        n_steps=best_hp["n_steps"],
        batch_size=best_hp["batch_size"],
        gamma=best_hp["gamma"],
        tensorboard_log="logs/",
    )
    model.learn(total_timesteps=best_total_timesteps)

    metrics = evaluate_model(model)
    mlflow.log_metrics(metrics)

    save_path = os.path.join(MODEL_PATH, "best_model")
    model.save(save_path)
    mlflow.log_artifact(save_path + ".zip")

    print(f"\n=== Bestes Modell fertig ===")
    print(f"Win Rate:   {metrics['win_rate']*100:.0f}%")
    print(f"Avg Reward: {metrics['avg_reward']:.3f}")
    print(f"Avg Health: {metrics['avg_health']:.1f}")