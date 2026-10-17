"""Deneyimlerinden öğrenen ajan (Deep Q-Learning, kısaca DQN).

Fikir: Ajan oynarken her hamleyi (durum, eylem, ödül, yeni durum, bitti mi)
olarak hafızaya yazar. Sonra şu kurala göre sinir ağını düzeltir:

    Q(durum, eylem)  ≈  ödül + GAMMA * (yeni durumdaki en yüksek Q değeri)

Yani "bu hamlenin değeri = hemen aldığım ödül + sonrasında alabileceğim
en iyi ödülün biraz indirimli hali". Ağın tahmini ile bu hedef arasındaki fark
(hata) küçüldükçe ajan daha iyi oynar.
"""

import os
import random
from collections import deque

import torch
from torch import nn

from ai.environment import NUM_ACTIONS, STATE_SIZES
from ai.model import QNetwork
from snake import config


def load_agent(path):
    """Kaydedilmiş bir modeli, doğru durum sürümüyle birlikte yükler.

    Yeni dosyalar {"state_version": ..., "weights": ...} sözlüğü olarak kaydedilir.
    İlk eğitimden kalan models/best.pth ise sadece ağırlıkları içerir; onu sürüm 1 sayarız.
    """
    checkpoint = torch.load(path, weights_only=True)
    if "state_version" in checkpoint:
        agent = DQNAgent(state_version=checkpoint["state_version"])
        agent.model.load_state_dict(checkpoint["weights"])
    else:
        agent = DQNAgent(state_version=1)
        agent.model.load_state_dict(checkpoint)
    return agent


class DQNAgent:
    def __init__(self, seed=None, state_version=1):
        # Ajan hangi durum sürümünü görüyorsa ortam da onu üretmeli (bkz. evaluate.py).
        self.state_version = state_version
        self.model = QNetwork(STATE_SIZES[state_version])
        # Adam, ağırlıkları hatayı azaltacak yönde güncelleyen yöntem ("optimizer").
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.loss_function = nn.MSELoss()  # Hata = (tahmin - hedef)^2 ortalaması

        # deque, maxlen dolunca en eski elemanı kendiliğinden silen bir liste.
        # C'de elle yazdığın "dairesel tampon" (ring buffer) ile aynı mantık.
        self.memory = deque(maxlen=config.MEMORY_SIZE)

        self.rng = random.Random(seed)
        self.games_played = 0

    # --- Hamle seçimi ---

    def epsilon(self):
        """Şu anki rastgele hamle olasılığı: oyun sayısı arttıkça doğrusal azalır."""
        progress = self.games_played / config.EPSILON_DECAY_GAMES
        value = config.EPSILON_START - progress * (config.EPSILON_START - config.EPSILON_MIN)
        return max(config.EPSILON_MIN, value)

    def act(self, state):
        """Eğitim sırasında hamle seçer: bazen rastgele (keşif), çoğunlukla en iyisi."""
        if self.rng.random() < self.epsilon():
            return self.rng.randrange(NUM_ACTIONS)
        return self.choose_action(state)

    def choose_action(self, state):
        """Hiç rastgelelik olmadan, ağa göre en iyi eylemi seçer.

        Kural tabanlı ajanla aynı isim ve parametreye sahip; böylece evaluate.py
        ve play_ai.py iki ajanı da aynı şekilde kullanabiliyor.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # `with` bloğu, içindeki kod bitince ayarı otomatik geri alır.
        # no_grad: "burada öğrenme yok, sadece tahmin" der; daha hızlıdır.
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(torch.argmax(q_values))  # En yüksek değerin indeksi = seçilen eylem

    # --- Öğrenme ---

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short(self, state, action, reward, next_state, done):
        """Az önce yapılan tek hamleden öğrenir (her adımda çağrılır)."""
        return self.train_step([state], [action], [reward], [next_state], [done])

    def train_long(self):
        """Hafızadan rastgele bir grup deneyim seçip onlardan öğrenir (her oyun sonunda).

        Rastgele seçmek önemli: art arda gelen hamleler birbirine çok benzer;
        karışık örneklerle öğrenmek ağın daha dengeli öğrenmesini sağlar.
        """
        if not self.memory:
            return None
        sample_size = min(config.BATCH_SIZE, len(self.memory))
        batch = self.rng.sample(self.memory, sample_size)
        # zip(*batch), [(s, a, r, s2, d), ...] listesini sütunlara ayırır:
        # bütün s'ler bir yerde, bütün a'lar bir yerde... (bir tablonun transpozu gibi)
        states, actions, rewards, next_states, dones = zip(*batch)
        return self.train_step(states, actions, rewards, next_states, dones)

    def train_step(self, states, actions, rewards, next_states, dones):
        """Bir grup deneyim için Q-learning güncellemesini yapar, hatayı döner."""
        states = torch.tensor(states, dtype=torch.float32)            # (N, 11)
        actions = torch.tensor(actions, dtype=torch.int64)            # (N,)
        rewards = torch.tensor(rewards, dtype=torch.float32)          # (N,)
        next_states = torch.tensor(next_states, dtype=torch.float32)  # (N, 11)
        dones = torch.tensor(dones, dtype=torch.float32)              # (N,) 1: oyun bitti

        # Ağın, gerçekten yapılmış eylemler için verdiği Q değerleri.
        # gather: her satırdan, `actions` içindeki indeksteki elemanı seçer.
        predicted = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Hedef: ödül + GAMMA * sonraki durumdaki en iyi Q değeri.
        # Oyun bittiyse "sonrası" yoktur; (1 - dones) çarpanı o kısmı sıfırlar.
        with torch.no_grad():
            next_best = self.model(next_states).max(dim=1).values
            target = rewards + config.GAMMA * next_best * (1 - dones)

        loss = self.loss_function(predicted, target)
        self.optimizer.zero_grad()  # Önceki adımdan kalan türevleri temizle
        loss.backward()             # Hatanın her ağırlığa göre türevini hesapla
        self.optimizer.step()       # Ağırlıkları hatayı azaltacak yönde güncelle
        return loss.item()

    # --- Kaydetme ---

    def save(self, path):
        """Ağırlıkları ve durum sürümünü dosyaya yazar. Klasör yoksa önce oluşturur."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {"state_version": self.state_version, "weights": self.model.state_dict()}
        torch.save(checkpoint, path)
