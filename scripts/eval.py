import os
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from RockOrRoll.src.env import RockOrRollEnv
import numpy as np

MODEL_PATH = "./models/ppo_rockorroll/final_model.zip"

def evaluate(n_episodes=10, render=True):
    env = RockOrRollEnv()
    model = MaskablePPO.load(MODEL_PATH, env=env)

    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step = 0

        print(f"\n{'='*50}")
        print(f"Episode {ep + 1}")
        print(f"{'='*50}")

        while not terminated and not truncated:
            action_masks = np.array(env.action_masks())
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step += 1

            if render:
                action_name = ["End Turn", "Roll", "Level Up"][action]
                print(f"Step {step:>3} | Action: {action_name:<10} | ", end="")
                env.render()

        # Episode Ergebnis
        state = env.game.get_state()
        survived = env.game.round > 43
        result = {
            "episode": ep + 1,
            "survived": survived,
            "final_round": env.game.round,
            "final_health": state["health"],
            "final_board_cost": state["board_cost"],
            "total_reward": round(total_reward, 3),
            "steps": step,
        }
        results.append(result)

        print(f"\n--- Episode {ep + 1} Result ---")
        print(f"Survived: {'✓ YES' if survived else '✗ NO'}")
        print(f"Final Round: {env.game.round}")
        print(f"Final Health: {state['health']}")
        print(f"Board Cost: {state['board_cost']}")
        print(f"Total Reward: {total_reward:.3f}")

    # Zusammenfassung
    wins = sum(1 for r in results if r["survived"])
    avg_round = sum(r["final_round"] for r in results) / n_episodes
    avg_health = sum(r["final_health"] for r in results) / n_episodes
    avg_reward = sum(r["total_reward"] for r in results) / n_episodes

    print(f"\n{'='*50}")
    print(f"SUMMARY ({n_episodes} episodes)")
    print(f"{'='*50}")
    print(f"Win Rate:      {wins}/{n_episodes} ({wins/n_episodes*100:.0f}%)")
    print(f"Avg Round:     {avg_round:.1f}")
    print(f"Avg Health:    {avg_health:.1f}")
    print(f"Avg Reward:    {avg_reward:.3f}")

if __name__ == "__main__":
    evaluate(n_episodes=10, render=True)