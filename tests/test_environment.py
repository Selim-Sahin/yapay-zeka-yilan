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
from ai.evaluate import evaluate
from ai.rule_based_agent import RuleBasedAgent
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


def test_rule_agent_plays_full_games():
    scores = evaluate(RuleBasedAgent(), num_games=5, seed=1)
    assert len(scores) == 5
    assert sum(scores) > 0
