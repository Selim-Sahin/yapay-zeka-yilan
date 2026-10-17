"""Eğitim sonuçlarının grafiğini çizip PNG dosyası olarak kaydeder."""

import os

import matplotlib

# "Agg" arka ucu pencere açmadan doğrudan dosyaya çizer. Bu satır,
# pyplot içe aktarılmadan önce gelmeli.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (bilerek use() satırından sonra)


def moving_average(values, window):
    """Her nokta için son `window` değerin ortalaması; grafikteki dalgalanmayı yumuşatır."""
    averages = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        recent = values[start:i + 1]
        averages.append(sum(recent) / len(recent))
    return averages


def save_training_curve(scores, eval_games, eval_scores, baseline, path, window=50):
    """Eğitim skorlarını, hareketli ortalamayı ve ölçüm sonuçlarını tek grafikte çizer.

    scores:      her eğitim oyununun skoru
    eval_games:  ölçüm yapılan oyun numaraları
    eval_scores: o ölçümlerdeki ortalama skor (rastgelelik kapalı)
    baseline:    kural tabanlı ajanın ortalaması (yatay çizgi), None ise çizilmez
    """
    game_numbers = range(1, len(scores) + 1)

    figure, axes = plt.subplots(figsize=(10, 5))
    axes.plot(game_numbers, scores, color="#b0c4de", linewidth=0.8, label="Eğitim oyunu skoru")
    axes.plot(game_numbers, moving_average(scores, window), color="#1f5fa8", linewidth=2,
              label=f"Son {window} oyunun ortalaması")
    if eval_games:
        axes.plot(eval_games, eval_scores, "o-", color="#d9822b", linewidth=1.5, markersize=4,
                  label="Ölçüm ortalaması (rastgelelik kapalı)")
    if baseline is not None:
        axes.axhline(baseline, color="#555555", linestyle="--", linewidth=1.2,
                     label=f"Kural tabanlı ajan ({baseline:.1f})")

    axes.set_title("DQN ajanının öğrenme eğrisi")
    axes.set_xlabel("Oyun sayısı")
    axes.set_ylabel("Skor")
    axes.grid(alpha=0.3)
    axes.legend(loc="upper left")
    figure.tight_layout()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)  # Belleği serbest bırak (C'deki free() gibi)
