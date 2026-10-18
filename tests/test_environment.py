"""ai/environment.py ve ai/rule_based_agent.py için testler."""

import random
import subprocess
import sys

import pytest

from ai.environment import (
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STRAIGHT,
    STATE_SIZE,
    SnakeEnv,
    turn,
)
from ai.evaluate import evaluate, make_agent
from ai.rule_based_agent import FloodFillRuleAgent, RuleBasedAgent
from snake import config


def make_env(width=10, height=10):
    return SnakeEnv(width, height, rng=random.Random(0))


def test_reset_returns_state_of_zeros_and_ones():
    state = make_env().reset()
    assert len(state) == STATE_SIZE
    assert all(value in (0, 1) for value in state)


def test_initial_state_values():
    env = make_env()
    env.reset()
    env.game.food = (8, 2)  # Baş (5, 5)'te; yem sağda ve yukarıda
    state = env.get_state()
    assert state == [
        0, 0, 0,     # tehlike yok
        0, 1, 0, 0,  # sağa gidiyor
        0, 1, 1, 0,  # yem sağda ve yukarıda
    ]


@pytest.mark.parametrize("direction, action, expected", [
    (config.RIGHT, ACTION_STRAIGHT, config.RIGHT),
    (config.RIGHT, ACTION_RIGHT, config.DOWN),
    (config.RIGHT, ACTION_LEFT, config.UP),
    (config.UP, ACTION_LEFT, config.LEFT),
    (config.UP, ACTION_RIGHT, config.RIGHT),  # listenin sonundan başına sarma
])
def test_turn(direction, action, expected):
    assert turn(direction, action) == expected


def test_danger_straight_at_wall():
    env = make_env()
    env.reset()
    env.game.snake = [(9, 5), (8, 5), (7, 5)]  # Baş sağ duvarın dibinde, sağa gidiyor
    state = env.get_state()
    assert state[0:3] == [1, 0, 0]


def test_danger_right_and_left_from_body():
    env = make_env()
    env.reset()
    # Baş (5, 5)'te, sağa gidiyor. Sağı (5, 6) ve solu (5, 4) gövdeyle dolu.
    env.game.snake = [(5, 5), (4, 5), (4, 4), (5, 4), (6, 4), (6, 3), (4, 6), (5, 6), (6, 6), (7, 6)]
    state = env.get_state()
    assert state[0:3] == [0, 1, 1]


def test_tail_cell_is_not_dangerous():
    env = make_env()
    env.reset()
    # Yukarı giden yılan; solundaki (4, 5) kuyruk, bir sonraki adımda boşalacak.
    env.game.snake = [(5, 5), (5, 6), (4, 6), (4, 5)]
    env.game.direction = config.UP
    state = env.get_state()
    assert state[2] == 0


def test_step_rewards():
    env = make_env()
    env.reset()
    head_x, head_y = env.game.snake[0]

    env.game.food = (head_x + 1, head_y)
    state, reward, done = env.step(ACTION_STRAIGHT)
    assert (reward, done) == (config.REWARD_FOOD, False)

    env.game.food = (0, 0)
    state, reward, done = env.step(ACTION_RIGHT)
    assert (reward, done) == (config.REWARD_STEP, False)
    assert env.game.direction == config.DOWN


def test_death_gives_negative_reward_and_ends_episode():
    env = make_env()
    env.reset()
    env.game.food = (0, 0)
    done = False
    while not done:
        state, reward, done = env.step(ACTION_STRAIGHT)  # Sağ duvara kadar düz git
    assert reward == config.REWARD_DEATH
    assert env.game.game_over


def test_episode_ends_if_no_food_for_too_long():
    env = make_env()
    env.reset()
    env.game.food = (0, 0)
    # Hep sağa dönerek 2x2'lik bir kare çizen yılan asla ölmez ama yem de yemez.
    limit = config.MAX_STEPS_PER_LENGTH * len(env.game.snake)
    for _ in range(limit):
        state, reward, done = env.step(ACTION_RIGHT)
        assert not done
    state, reward, done = env.step(ACTION_RIGHT)
    assert done
    assert reward == config.REWARD_DEATH


def test_environment_does_not_import_pygame():
    # Ayrı bir Python süreci başlatıp ortamı içe aktarıyoruz; pygame yüklenmemeli.
    # (Aynı süreçte kontrol etseydik, başka bir test pygame'i yüklemiş olabilirdi.)
    code = "import sys, ai.environment, ai.rule_based_agent; print('pygame' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_state_version_2_adds_three_area_values():
    env = SnakeEnv(10, 10, rng=random.Random(0), state_version=2)
    state = env.reset()
    assert len(state) == env.state_size == STATE_SIZE + 3
    assert state[:STATE_SIZE] == env.basic_state()  # İlk 11 sayı sürüm 1 ile aynı
    assert state[STATE_SIZE:] == [1.0, 1.0, 1.0]   # Açık alanda her yön güvenli


def test_reachable_area_is_zero_for_collision():
    env = make_env()
    env.reset()
    env.game.snake = [(9, 5), (8, 5), (7, 5)]  # Önü duvar
    assert env.reachable_area(env.next_head(ACTION_STRAIGHT), limit=3) == 0


def test_reachable_area_detects_closed_pocket():
    # 5x5 tahta. Sola giden 9 karelik yılan, gövdesiyle üstteki iki satırı kapatıyor:
    #
    #   y=0   .  .  .  .  .      Cep: y=0 satırının tamamı + (0, 1) ve (1, 1) = 7 kare
    #   y=1   .  .  H  b  b      H: baş (2, 1), sola gidiyor
    #   y=2   b  b  b  b  b      b: gövde
    #   y=3   T  .  .  .  .      T: kuyruk (0, 3)
    #
    env = SnakeEnv(5, 5, rng=random.Random(0), state_version=2)
    env.reset()
    env.game.snake = [(2, 1), (3, 1), (4, 1), (4, 2), (3, 2), (2, 2), (1, 2), (0, 2), (0, 3)]
    env.game.direction = config.LEFT

    straight, right, left = env.reachable_area_features()
    assert straight == 7 / 9  # Düz (1, 1): 7 karelik cep, yılanın boyundan (9) küçük
    assert right == 7 / 9     # Sağa dönmek = yukarı (2, 0): aynı cep
    assert left == 0.0        # Sola dönmek = aşağı (2, 2): gövde


def test_rule_agent_avoids_danger():
    agent = RuleBasedAgent()
    # Önde ve sağda tehlike var, sağa gidiyor, yem sağda: tek güvenli yol sol.
    state = [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    assert agent.choose_action(state) == ACTION_LEFT


def test_rule_agent_turns_toward_food():
    agent = RuleBasedAgent()
    # Tehlike yok, sağa gidiyor, yem aşağıda: sağa dönmek (aşağı) yeme götürür.
    state = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    assert agent.choose_action(state) == ACTION_RIGHT


def test_flood_fill_rule_agent_prefers_larger_area_over_food():
    agent = FloodFillRuleAgent()
    # Tehlike yok, sağa gidiyor, yem aşağıda (sağa dönmek yeme götürür).
    # Ama sağa dönmek yarım boyluk bir cebe giriyor; düz gitmek ise geniş alana açılıyor.
    state = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1] + [1.0, 0.5, 0.3]
    assert agent.choose_action(state) == ACTION_STRAIGHT


def test_flood_fill_rule_agent_goes_toward_food_when_areas_are_equal():
    agent = FloodFillRuleAgent()
    # Açık alan: üç yön de 1.0. Bu durumda eski ajan gibi yeme doğru döner.
    state = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1] + [1.0, 1.0, 1.0]
    assert agent.choose_action(state) == ACTION_RIGHT


def test_flood_fill_rule_agent_goes_straight_when_every_move_is_fatal():
    agent = FloodFillRuleAgent()
    state = [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0] + [0.0, 0.0, 0.0]
    assert agent.choose_action(state) == ACTION_STRAIGHT


def test_flood_fill_rule_agent_uses_state_version_2():
    agent = make_agent("rule_ff")
    assert isinstance(agent, FloodFillRuleAgent)
    assert agent.state_version == 2
    assert RuleBasedAgent.state_version == 1  # Eski ajan değişmedi


def test_flood_fill_rule_agent_plays_full_games():
    scores = evaluate(FloodFillRuleAgent(), num_games=3, seed=1)
    assert len(scores) == 3
    assert sum(scores) > 0


def test_rule_agent_plays_full_games():
    scores = evaluate(RuleBasedAgent(), num_games=5, seed=1)
    assert len(scores) == 5
    assert sum(scores) > 0
