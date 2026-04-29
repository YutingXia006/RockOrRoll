import gymnasium as gym
import numpy as np
from gymnasium import spaces
from game import TFTGame, opponent_strength, get_stage

MAX_VALUES = {
    "gold": 100,
    "health": 100,
    "round": 43,
    "win_streak": 15,
    "loss_streak": 15,
    "board_cost": 600,
    "level": 9,
    "interest": 5,
    "income": 20,
    "last_opponent_strength": 600,
}

class RockOrRollEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.game = TFTGame()
        self.last_opponent_strength = 0.0

        # 0: End turn, 1: Roll, 2: Level up
        self.action_space = spaces.Discrete(3)

        # 10 normalized observations
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32
        )

    def _get_obs(self):
        state = self.game.get_state()
        obs = np.array([
            min(state["gold"]                / MAX_VALUES["gold"], 1.0),
            min(state["health"]              / MAX_VALUES["health"], 1.0),
            min(state["round"]               / MAX_VALUES["round"], 1.0),
            min(state["win_streak"]          / MAX_VALUES["win_streak"], 1.0),
            min(state["loss_streak"]         / MAX_VALUES["loss_streak"], 1.0),
            min(state["board_cost"]          / MAX_VALUES["board_cost"], 1.0),
            min(state["level"]               / MAX_VALUES["level"], 1.0),
            min(state["interest"]            / MAX_VALUES["interest"], 1.0),
            min(state["income"]              / MAX_VALUES["income"], 1.0),
            min(self.last_opponent_strength  / MAX_VALUES["last_opponent_strength"], 1.0),
        ], dtype=np.float32)
        return obs

    def _get_reward(self, game_over):
        if game_over:
            if self.game.round > 43:
                return 1.0   # Gewonnen
            else:
                return -1.0  # Gestorben
        # Kleines Signal jede Runde
        return (self.game.health / MAX_VALUES["health"]) * 0.1

    def reset(self, seed=None, options=None): #type: ignore
        super().reset(seed=seed, options=options)
        self.game.reset()
        self.last_opponent_strength = 0.0
        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        truncated = False
        terminated = False  # Default hier setzen

        if action == 0:
            opp = opponent_strength(self.game.round)
            self.last_opponent_strength = opp
            self.game.next_round()
            game_over = self.game.is_game_over()
            reward = self._get_reward(game_over)
            terminated = game_over
        
        elif action == 1:
            success = self.game.action_roll()
            reward = -0.01 if not success else 0.0
        
        elif action == 2:
            success = self.game.action_level_up()
            reward = -0.01 if not success else 0.0

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def render(self):
        state = self.game.get_state()
        print(f"Round {state['round']} | "
              f"Gold {state['gold']} | "
              f"Health {state['health']} | "
              f"Board {state['board_cost']} | "
              f"Opponent {self.last_opponent_strength:.1f} | "
              f"Level {state['level']}")