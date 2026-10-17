"""DQN ajanını pencere açmadan eğitir. Çalıştırmak için:
    python train.py                    # varsayılan süre sınırı: config.TRAIN_MINUTES
    python train.py --minutes 5        # kısa bir deneme

Her EVAL_EVERY_GAMES oyunda bir, ajan rastgelelik kapalı olarak EVAL_GAMES oyun
oynatılıp ölçülür; ortalaması o ana kadarki en iyisiyse models/best.pth'e kaydedilir.
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


def train(minutes, max_games, seed):
    random.seed(seed)
    torch.manual_seed(seed)  # Ağın başlangıç ağırlıkları her seferinde aynı olsun

    env = SnakeEnv(rng=random.Random(seed))
    agent = DQNAgent(seed=seed)
    baseline = average(evaluate(RuleBasedAgent(), num_games=100, seed=0))

    scores = []
    eval_games = []
    eval_scores = []
    best_eval = None
    start_time = time.time()
    deadline = start_time + minutes * 60

    print(f"Eğitim başladı: en fazla {minutes} dakika / {max_games} oyun. "
          f"Kural tabanlı ajan ortalaması: {baseline:.2f}", flush=True)

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
                agent.save(config.BEST_MODEL_PATH)
            agent.save(config.LAST_MODEL_PATH)
            save_training_curve(scores, eval_games, eval_scores, baseline, config.TRAINING_CURVE_PATH)

            elapsed = (time.time() - start_time) / 60
            recent = average(scores[-config.EVAL_EVERY_GAMES:])
            print(f"[{elapsed:5.1f} dk] oyun {agent.games_played:5d} | epsilon {agent.epsilon():.3f} | "
                  f"son {config.EVAL_EVERY_GAMES} oyun ort. {recent:6.2f} | ölçüm ort. {eval_average:6.2f}"
                  f"{'  <- en iyi, kaydedildi' if is_best else ''}", flush=True)

    agent.save(config.LAST_MODEL_PATH)
    save_training_curve(scores, eval_games, eval_scores, baseline, config.TRAINING_CURVE_PATH)
    elapsed = (time.time() - start_time) / 60
    print(f"Eğitim bitti: {agent.games_played} oyun, {elapsed:.1f} dakika. "
          f"En iyi ölçüm ortalaması: {best_eval}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="DQN ajanını eğitir.")
    parser.add_argument("--minutes", type=float, default=config.TRAIN_MINUTES)
    parser.add_argument("--max-games", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    # argparse, "--max-games" adını Python'a uygun "max_games" adına çevirir.
    train(args.minutes, args.max_games, args.seed)


if __name__ == "__main__":
    main()
