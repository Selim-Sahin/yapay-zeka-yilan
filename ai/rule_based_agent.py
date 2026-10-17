"""Kural tabanlı ajan: "tehlikeden kaç, yeme doğru git".

Hiçbir şey öğrenmez; kuralları biz yazdık. Öğrenen ajanla (Aşama 4) adil bir
karşılaştırma olsun diye, o da sadece aynı 11 sayılık durum listesini görür.
"""

from ai.environment import (
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STRAIGHT,
    DIRECTION_START,
    FOOD_START,
    STATE_DIRECTION_ORDER,
    turn,
)


class RuleBasedAgent:
    # Sınıf değişkeni: her RuleBasedAgent nesnesi için aynıdır (C'deki static gibi).
    # Kural tabanlı ajan her zaman sürüm 1 durumunu (11 sayı) kullanır.
    state_version = 1

    def choose_action(self, state):
        """Durum listesine bakıp bir eylem seçer."""
        # Durum listesinin ilk üç elemanı (önde/sağda/solda tehlike) ile eylem
        # numaraları (düz=0, sağ=1, sol=2) aynı sırada. Bu yüzden state[action]
        # "bu eylemi yaparsam ölür müyüm?" sorusunun cevabıdır.
        candidate_actions = [ACTION_STRAIGHT, ACTION_RIGHT, ACTION_LEFT]
        safe_actions = [action for action in candidate_actions if state[action] == 0]

        if not safe_actions:
            return ACTION_STRAIGHT  # Her yön tehlikeli: ne yapsak ölüyoruz

        direction = self.current_direction(state)
        for action in safe_actions:
            if self.moves_toward_food(state, turn(direction, action)):
                return action

        # Güvenli ama yeme yaklaştırmayan bir hamle: öncelik düz gitmekte.
        return safe_actions[0]

    def current_direction(self, state):
        """Durum listesindeki yön bölümünden (tek bir 1 değeri) mevcut yönü bulur."""
        for i, direction in enumerate(STATE_DIRECTION_ORDER):
            # enumerate, döngüde hem sırayı (i) hem elemanı verir:
            # C'deki for (i = 0; i < n; i++) { direction = order[i]; ... } gibi.
            if state[DIRECTION_START + i] == 1:
                return direction
        raise ValueError("Durum listesinde yön bilgisi yok")

    def moves_toward_food(self, state, direction):
        """Bu yöne gitmek yemi yaklaştırıyor mu?"""
        i = STATE_DIRECTION_ORDER.index(direction)
        return state[FOOD_START + i] == 1
