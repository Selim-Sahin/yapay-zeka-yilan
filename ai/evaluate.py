"""Bir ajanı pencere açmadan çok sayıda oyunda oynatıp skorlarını ölçer.

Çalıştırmak için:  python -m ai.evaluate --agent rule --games 100
(`-m` ile çalıştırmak, Python'un proje klasörünü tanımasını ve
`from snake ...` satırlarının çalışmasını sağlar.)
"""

import argparse
import random

from ai.environment import SnakeEnv
from ai.rule_based_agent import RuleBasedAgent


def make_agent(name):
    """İsmi verilen ajanı oluşturur."""
    if name == "rule":
        return RuleBasedAgent()
    raise ValueError(f"Bilinmeyen ajan: {name}")


def play_episode(env, agent):
    """Tek bir oyunu baştan sona oynatır ve skoru döner."""
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        state, reward, done = env.step(action)
    return env.game.score


def evaluate(agent, num_games, seed=0):
    """Ajanı num_games oyun oynatır, skorların listesini döner.

    Sabit seed sayesinde iki ajan aynı yem dizisiyle karşılaştırılabilir.
    """
    env = SnakeEnv(rng=random.Random(seed))
    scores = []
    for _ in range(num_games):
        scores.append(play_episode(env, agent))
    return scores


def main():
    # argparse, komut satırı argümanlarını okur. C'deki argc/argv'yi elle
    # ayrıştırmanın hazır ve kolay hali.
    parser = argparse.ArgumentParser(description="Ajanın ortalama skorunu ölçer.")
    parser.add_argument("--agent", default="rule", help="rule")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    agent = make_agent(args.agent)
    scores = evaluate(agent, args.games, args.seed)
    average = sum(scores) / len(scores)
    print(f"Ajan: {args.agent} | Oyun: {len(scores)} | Ortalama: {average:.2f} "
          f"| En yüksek: {max(scores)} | En düşük: {min(scores)}")


if __name__ == "__main__":
    main()
