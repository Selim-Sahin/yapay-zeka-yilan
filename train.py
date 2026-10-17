"""DQN ajanını pencere açmadan eğitir. Çalıştırmak için:
    python train.py --name v2 --state-version 2              # süre sınırı: config.TRAIN_MINUTES
    python train.py --name deneme --minutes 5                # kısa bir deneme

Her EVAL_EVERY_GAMES oyunda bir, ajan rastgelelik kapalı olarak EVAL_GAMES oyun
oynatılıp ölçülür; ortalaması o ana kadarki en iyisiyse models/best_<ad>.pth'e kaydedilir.
Grafik assets/training_curve_<ad>.png olur. Her deneyin adı ayrı olduğu için
önceki modellerin üstüne yazılmaz.
"""

import argparse
import random
import time

import torch

from ai.agent import DQNAgent
from ai.environment import SnakeEnv
from ai.evaluate import evaluate
from ai.plot import save_training_curve
from ai.rule_based_agent import RuleBasedAgent
from snake import config


def play_training_game(env, agent):
    """Bir eğitim oyunu oynar; her hamleden ve oyun sonunda hafızadan öğrenir."""
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train_short(state, action, reward, next_state, done)
        state = next_state
    agent.train_long()
    agent.games_played += 1
    return env.game.score


def average(values):
    return sum(values) / len(values)


def train(name, state_version, minutes, max_games, seed):
    best_path = f"{config.MODELS_DIR}/best_{name}.pth"
    last_path = f"{config.MODELS_DIR}/last_{name}.pth"
    curve_path = f"{config.ASSETS_DIR}/training_curve_{name}.png"

    random.seed(seed)
    torch.manual_seed(seed)  # Ağın başlangıç ağırlıkları her seferinde aynı olsun

    env = SnakeEnv(rng=random.Random(seed), state_version=state_version)
    agent = DQNAgent(seed=seed, state_version=state_version)
    baseline = average(evaluate(RuleBasedAgent(), num_games=100, seed=0))

    scores = []
    eval_games = []
    eval_scores = []
    best_eval = None
    start_time = time.time()
    deadline = start_time + minutes * 60

    print(f"Eğitim '{name}' başladı: durum sürümü {state_version} ({env.state_size} sayı), "
          f"en fazla {minutes} dakika / {max_games} oyun. "
          f"Kural tabanlı ajan ortalaması (100 oyun): {baseline:.2f}", flush=True)

    while len(scores) < max_games and time.time() < deadline:
        scores.append(play_training_game(env, agent))

        if agent.games_played % config.EVAL_EVERY_GAMES == 0:
            eval_average = average(evaluate(agent, config.EVAL_GAMES, seed=config.EVAL_SEED))
            eval_games.append(agent.games_played)
            eval_scores.append(eval_average)

            # İlk ölçümse (None) ya da öncekilerden iyiyse kaydet.
            is_best = best_eval is None or eval_average > best_eval
            if is_best:
                best_eval = eval_average
                agent.save(best_path)
            agent.save(last_path)
            save_training_curve(scores, eval_games, eval_scores, baseline, curve_path)

            elapsed = (time.time() - start_time) / 60
            recent = average(scores[-config.EVAL_EVERY_GAMES:])
            print(f"[{elapsed:5.1f} dk] oyun {agent.games_played:5d} | epsilon {agent.epsilon():.3f} | "
                  f"son {config.EVAL_EVERY_GAMES} oyun ort. {recent:6.2f} | ölçüm ort. {eval_average:6.2f}"
                  f"{'  <- en iyi, kaydedildi' if is_best else ''}", flush=True)

    agent.save(last_path)
    save_training_curve(scores, eval_games, eval_scores, baseline, curve_path)
    elapsed = (time.time() - start_time) / 60
    print(f"Eğitim bitti: {agent.games_played} oyun, {elapsed:.1f} dakika. "
          f"En iyi ölçüm ortalaması: {best_eval}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="DQN ajanını eğitir.")
    parser.add_argument("--name", required=True, help="deneyin adı; dosya adlarında kullanılır")
    parser.add_argument("--state-version", type=int, default=1, choices=[1, 2])
    parser.add_argument("--minutes", type=float, default=config.TRAIN_MINUTES)
    parser.add_argument("--max-games", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    # argparse, "--max-games" adını Python'a uygun "max_games" adına çevirir.
    train(args.name, args.state_version, args.minutes, args.max_games, args.seed)


if __name__ == "__main__":
    main()
