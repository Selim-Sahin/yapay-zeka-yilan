"""Ajanın "beyni": 11 sayılık durumu alıp 3 eylemin her biri için bir puan veren sinir ağı.

Bu puanlara "Q değeri" denir: "bu durumda bu eylemi yaparsam, bundan sonra toplam
ne kadar ödül beklerim?" sorusunun tahmini. Ajan en yüksek Q değerli eylemi seçer.

PyTorch'ta veriler "tensor" denen çok boyutlu dizilerle tutulur. C'deki
float states[N][11] dizisini düşün; tensor bunun boyutu, tipi ve üzerindeki
matematik işlemleri hazır gelen hali.
"""

from torch import nn

from ai.environment import NUM_ACTIONS, STATE_SIZE
from snake import config


class QNetwork(nn.Module):
    """state_size giriş (sürüm 1'de 11) -> 256 gizli nöron -> 3 çıkış.

    `class QNetwork(nn.Module)` "kalıtım" (inheritance) demek: QNetwork,
    PyTorch'un nn.Module sınıfının bütün özelliklerini devralır ve üstüne
    kendi katmanlarını ekler. C'de bir struct'ın ilk alanı olarak başka bir
    struct koyup onu genişletmeye benzer.
    """

    def __init__(self, state_size=STATE_SIZE, hidden_size=config.HIDDEN_SIZE):
        super().__init__()  # Önce nn.Module'ün kendi __init__'i çalışmalı
        # nn.Sequential, katmanları sırayla birbirine bağlar.
        # Linear: y = W·x + b (matris çarpımı + toplama).
        # ReLU: negatif değerleri 0 yapar; ağın düz bir çizgiden fazlasını öğrenmesini sağlar.
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, NUM_ACTIONS),
        )

    def forward(self, states):
        """Durumları alır, her eylem için Q değerlerini döner.

        `model(states)` yazınca PyTorch bu fonksiyonu kendisi çağırır.
        """
        return self.layers(states)
