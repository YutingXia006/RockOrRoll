import random

# Gold income constants
BASE_GOLD = 5
MAX_INTEREST = 5  # max 5 bonus gold from interest
NATURAL_XP = 2

# Interest thresholds
INTEREST_THRESHOLDS = [10, 20, 30, 40, 50]

# Streak bonuses
STREAK_BONUS = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
    4: 1,
    5: 2,
    6: 3    # +5 Steak = 3 gold
}

# XP needed to reach each level
XP_TO_LEVEL = {
    1: 2, 
    2: 2,
    3: 6,
    4: 10,
    5: 20,
    6: 36,
    7: 60,
    8: 68,
    9: 68,
}

# How much board strength a roll gives per level
ROLL_STRENGTH_GAIN = {
    1: (0, 1.5),   # (min, max) random gain
    2: (0, 2.0),
    3: (0, 2.5),
    4: (0, 3.5),
    5: (0, 4.5),
    6: (0, 5.5),
    7: (0, 6.5),
    8: (0, 8.0),
    9: (0, 10.0),
    10: (0, 12.5)
}

# Stage damage on loss
STAGE_DAMAGE = {
    1: 0,
    2: 2,
    3: 6,
    4: 7,
    5: 10,
    6: 12,
    7: 17,
    8: 10000,  # instant death
}

# Round to stage mapping
def get_stage(round_num):
    return min((round_num - 1) // 7 + 2, 8)

# Opponent strength per round (gets harder over time)
def opponent_strength(round_num):
    natural_xp = round_num * 2
    estimated_gold = (5 * round_num) + (round_num * 0.5)
    gold_invested = estimated_gold * 1.5
    base = natural_xp + gold_invested  # kein fixer Startwert mehr
    variance = 1 + (round_num * 0.5)
    return base + random.uniform(-variance, variance)


class TFTGame:
    def __init__(self):
        self.gold = 0
        self.health = 100
        self.round = 1
        self.win_streak = 0
        self.loss_streak = 0
        self.level = 3
        self.experience = 0
        self.board_cost = 18
    
    def reset(self):
        self.__init__()

    # ---------- INCOME ----------

    def calculate_interest(self):
        interest = 0
        for threshold in INTEREST_THRESHOLDS:
            if self.gold >= threshold:
                interest += 1
        return min(interest, MAX_INTEREST)

    def calculate_streak_bonus(self):
        streak = max(self.win_streak, self.loss_streak)
        return STREAK_BONUS.get(min(streak, 5), 3)

    def calculate_income(self):
        interest = self.calculate_interest()
        streak = self.calculate_streak_bonus()
        return BASE_GOLD + interest + streak

    def collect_income(self):
        self.gold += self.calculate_income()

    # ---------- ACTIONS ----------

    def action_save(self):
        # Do nothing, just collect income at end of round
        return True

    def action_roll(self):
        if self.gold < 2:
            return False
        self.gold -= 2
        self.board_cost += 2
        return True

    def action_level_up(self):
        if self.gold < 4:
            return False
        if self.level >= 10:
            return False
        self.gold -= 4
        self.board_cost += 4
        self.experience += 4  # extra XP on top of natural gain
        next_level = self.level + 1
        if next_level in XP_TO_LEVEL:
            if self.experience >= XP_TO_LEVEL[next_level]:
                self.level += 1
                self.experience -= XP_TO_LEVEL[next_level]
        return True

    def action_roll_and_level(self):
        # Try both, order matters — level first
        leveled = self.action_level_up()
        rolled = self.action_roll()
        return leveled or rolled  # success if at least one worked

    # ---------- COMBAT ----------

    def simulate_combat(self):
        opponent = opponent_strength(self.round)
        stage = get_stage(self.round)
        
        if self.board_cost >= opponent:
            self.win_streak += 1
            self.loss_streak = 0
            self.gold += 1
        else:
            self.health -= STAGE_DAMAGE[stage]
            self.loss_streak += 1
            self.win_streak = 0

    # ---------- ROUND ----------

    def next_round(self):
        self.simulate_combat()
        self._natural_level_up()
        self.collect_income()
        self.round += 1

    def _natural_level_up(self):
        self.experience += NATURAL_XP
        self.board_cost += NATURAL_XP
        next_level = self.level + 1
        if next_level in XP_TO_LEVEL:
            if self.experience >= XP_TO_LEVEL[next_level]:
                self.level += 1
                self.experience -= XP_TO_LEVEL[next_level]

    def is_game_over(self):
        return self.health <= 0 or self.round > 43

    # ---------- STATE ----------

    def get_state(self):
        return {
            "gold": self.gold,
            "health": self.health,
            "round": self.round,
            "win_streak": self.win_streak,
            "loss_streak": self.loss_streak,
            "board_cost": self.board_cost,
            "level": self.level,
            "experience": self.experience,
            "interest": self.calculate_interest(),
            "income": self.calculate_income(),
        }
    
if __name__ == "__main__":
    game = TFTGame()
    
    print("=== Starting Game ===")
    print(game.get_state())
    
    roll_count = 0
    # Optimale Strategie: Interest bis 50 sparen, dann alles rollen
    while not game.is_game_over():
        while game.gold >= 50:  # maximales Interest erreicht
            game.action_roll()
        if game.gold >= 4 and game.level < 8:
            game.action_level_up()
        else:
            game.action_save()
        game.next_round()
        print(f"Round {game.round} | Stage {get_stage(game.round)} | {game.get_state()}")

    print(f"Total rolls: {roll_count}")
    for r in [1, 10, 20, 30, 43]:
        strengths = [opponent_strength(r) for _ in range(5)]
        print(f"Round {r}: avg opponent = {sum(strengths)/5:.1f}")
    
    print("=== Game Over ===")
    print(f"Survived until round {game.round} with {game.health} HP")