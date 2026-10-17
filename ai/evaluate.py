"""Bir ajanı pencere açmadan çok sayıda oyunda oynatıp skorlarını ölçer.

Çalıştırmak için:  python -m ai.evaluate --agent rule            (varsayılan: 500 oyun, seed 0)
                   python -m ai.evaluate --agent dqn --model models/best.pth
(`-m` ile çalıştırmak, Python'un proje klasörünü tanımasını ve
`from snake ...` satırlarının çalışmasını sağlar.)
"""

import argparse
import math
import random
import statistics

from ai.environment import SnakeEnv
from ai.rule_based_agent import RuleBasedAgent
from snake import config


def make_agent(name, model_path=config.BEST_MODEL_PATH):
    """İsmi verilen ajanı oluşturur. "dqn" için eğitilmiş model dosyadan yüklenir."""
    if name == "rule":
        return RuleBasedAgent()
    if name == "dqn":
        # torch'u sadece gerektiğinde içe aktarıyoruz; kural tabanlı ajan torch'suz da çalışsın.
        from ai.agent import load_agent
        return load_agent(model_path)
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
    Ortam, ajanın beklediği durum sürümünü üretir.
    """
    env = SnakeEnv(rng=random.Random(seed), state_version=agent.state_version)
    scores = []
    for _ in range(num_games):
        scores.append(play_episode(env, agent))
    return scores


def summarize(scores):
    """Skor listesinin özet istatistiklerini bir sözlük olarak döner.

    Ortalamanın standart hatası (SEM) = standart sapma / karekök(oyun sayısı).
    "Aynı ölçümü tekrar yapsak ortalama ne kadar oynar?" sorusunun cevabıdır.
    Oyun sayısı 4 katına çıkınca SEM yarıya iner; 500 oyunun 100 oyundan
    daha güvenilir olmasının sebebi bu.
    """
    std = statistics.stdev(scores)
    return {
        "games": len(scores),
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "std": std,
        "sem": std / math.sqrt(len(scores)),
        "max": max(scores),
        "min": min(scores),
    }


def main():
    # argparse, komut satırı argümanlarını okur. C'deki argc/argv'yi elle
    # ayrıştırmanın hazır ve kolay hali.
    parser = argparse.ArgumentParser(description="Ajanın ortalama skorunu ölçer.")
    parser.add_argument("--agent", default="rule", help="rule veya dqn")
    parser.add_argument("--model", default=config.BEST_MODEL_PATH, help="dqn için model dosyası")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    agent = make_agent(args.agent, args.model)
    stats = summarize(evaluate(agent, args.games, args.seed))
    # `**stats` sözlüğü isimli parametrelere açar; format() içinde {mean:.2f} gibi kullanılır.
    print("Ajan: {agent} | Oyun: {games} | Ortalama: {mean:.2f} | Medyan: {median} | "
          "Std. sapma: {std:.2f} | Std. hata: {sem:.2f} | En yüksek: {max} | En düşük: {min}"
          .format(agent=args.agent, **stats))


if __name__ == "__main__":
    main()
