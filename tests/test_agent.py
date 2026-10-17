"""ai/model.py, ai/agent.py ve ai/plot.py için testler."""

import random

import torch

from ai.agent import DQNAgent, load_agent
from ai.environment import NUM_ACTIONS, STATE_SIZE, SnakeEnv
from ai.evaluate import evaluate, make_agent, summarize
from ai.model import QNetwork
from ai.plot import moving_average, save_training_curve
from snake import config


def random_state(rng):
    return [rng.randint(0, 1) for _ in range(STATE_SIZE)]


def test_model_output_shape():
    model = QNetwork()
    states = torch.zeros((5, STATE_SIZE))  # 5 durumluk bir grup
    assert model(states).shape == (5, NUM_ACTIONS)


def test_choose_action_returns_valid_action():
    agent = DQNAgent(seed=0)
    rng = random.Random(0)
    for _ in range(20):
        assert agent.choose_action(random_state(rng)) in range(NUM_ACTIONS)


def test_epsilon_decreases_to_minimum():
    agent = DQNAgent(seed=0)
    assert agent.epsilon() == config.EPSILON_START
    agent.games_played = config.EPSILON_DECAY_GAMES // 2
    assert config.EPSILON_MIN < agent.epsilon() < config.EPSILON_START
    agent.games_played = config.EPSILON_DECAY_GAMES * 10
    assert agent.epsilon() == config.EPSILON_MIN


def test_act_is_random_at_start():
    agent = DQNAgent(seed=0)  # epsilon = 1.0: her hamle rastgele
    state = [0] * STATE_SIZE
    actions = {agent.act(state) for _ in range(100)}  # `{...}` bir küme (set): tekrar etmeyen elemanlar
    assert actions == set(range(NUM_ACTIONS))


def test_train_step_reduces_error_on_repeated_example():
    torch.manual_seed(0)
    agent = DQNAgent(seed=0)
    state = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    # Aynı (bitmiş) deneyimi tekrar tekrar öğrenince hata azalmalı.
    first_loss = agent.train_short(state, 0, config.REWARD_DEATH, state, True)
    for _ in range(50):
        last_loss = agent.train_short(state, 0, config.REWARD_DEATH, state, True)
    assert last_loss < first_loss


def test_train_long_uses_memory():
    agent = DQNAgent(seed=0)
    assert agent.train_long() is None  # Hafıza boşken öğrenecek bir şey yok
    rng = random.Random(0)
    for _ in range(10):
        agent.remember(random_state(rng), rng.randrange(NUM_ACTIONS), 0, random_state(rng), False)
    assert len(agent.memory) == 10
    assert agent.train_long() >= 0


def test_save_and_load_give_same_outputs(tmp_path):
    # tmp_path: pytest'in her test için verdiği geçici klasör; models/ klasörüne dokunmuyoruz.
    path = str(tmp_path / "models" / "test.pth")
    agent = DQNAgent(seed=0)
    agent.save(path)

    loaded = make_agent("dqn", path)
    states = torch.rand((4, STATE_SIZE))
    assert torch.equal(agent.model(states), loaded.model(states))


def test_state_version_2_model_has_14_inputs():
    agent = DQNAgent(seed=0, state_version=2)
    assert agent.model(torch.zeros((1, STATE_SIZE + 3))).shape == (1, NUM_ACTIONS)


def test_save_and_load_keep_state_version(tmp_path):
    path = str(tmp_path / "v2.pth")
    DQNAgent(seed=0, state_version=2).save(path)
    assert load_agent(path).state_version == 2


def test_old_model_file_loads_as_version_1(tmp_path):
    # İlk eğitimden kalan models/best.pth sadece ağırlıkları içeriyor (sürüm bilgisi yok).
    path = str(tmp_path / "old.pth")
    torch.save(QNetwork().state_dict(), path)
    agent = load_agent(path)
    assert agent.state_version == 1


def test_summarize():
    stats = summarize([10, 20, 30, 40])
    assert stats["mean"] == 25
    assert stats["median"] == 25
    assert round(stats["std"], 4) == 12.9099
    assert round(stats["sem"], 4) == round(12.9099 / 2, 4)  # std / karekök(4)


def test_dqn_agent_works_with_evaluate():
    # Sürüm 2 ajanı ile de: evaluate, ortamı ajanın durum sürümüne göre kurmalı.
    scores = evaluate(DQNAgent(seed=0, state_version=2), num_games=1, seed=0)
    assert len(scores) == 1
    # Eğitilmemiş ajan da oyunu sonuna kadar oynayabilmeli (adım sınırı sayesinde).
    scores = evaluate(DQNAgent(seed=0), num_games=2, seed=0)
    assert len(scores) == 2


def test_one_training_game_runs():
    from train import play_training_game
    env = SnakeEnv(10, 10, rng=random.Random(0))
    agent = DQNAgent(seed=0)
    play_training_game(env, agent)
    assert agent.games_played == 1
    assert len(agent.memory) > 0


def test_moving_average():
    assert moving_average([2, 4, 6, 8], window=2) == [2, 3, 5, 7]


def test_save_training_curve_creates_png(tmp_path):
    path = str(tmp_path / "curve.png")
    save_training_curve([1, 2, 3], [2], [2.5], baseline=2.0, path=path)
    with open(path, "rb") as file:
        assert file.read(8) == b"\x89PNG\r\n\x1a\n"  # PNG dosyalarının ilk 8 baytı hep budur
