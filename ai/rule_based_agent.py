"""Kural tabanlı ajanlar. Hiçbir şey öğrenmezler; kuralları biz yazdık.

- RuleBasedAgent: "tehlikeden kaç, yeme doğru git". 11 sayılık durumu (sürüm 1) görür.
- FloodFillRuleAgent: Aynı fikir, ama DQN v2 ile aynı 14 sayılık durumu (sürüm 2) görür
  ve en geniş alana açılan hamleyi tercih eder. DQN v2 ile eşit bilgiyle karşılaştırma için.
"""

from ai.environment import (
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STRAIGHT,
    AREA_START,
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


class FloodFillRuleAgent(RuleBasedAgent):
    """En geniş alana açılan hamleler arasından yeme doğru gideni seçen ajan.

    `(RuleBasedAgent)` kalıtım demek: current_direction ve moves_toward_food
    metotlarını olduğu gibi devralır, sadece choose_action'ı yeniden yazar.
    """

    state_version = 2  # DQN v2 ile aynı 14 sayılık durum

    def choose_action(self, state):
        # Alan değerleri de eylem numaralarıyla aynı sırada: düz, sağ, sol.
        # Değer 0 ise o hamle ölüm demek; 1.0 ise önünde en az boyu kadar yer var.
        areas = state[AREA_START:AREA_START + 3]
        best_area = max(areas)
        if best_area == 0:
            return ACTION_STRAIGHT  # Her yön ölüm: ne yapsak aynı

        # Sadece en geniş alanı veren hamleler aday. Açık alanda hepsi 1.0 olduğu için
        # bu ajan eski ajan gibi davranır; fark, bir cebe girme tehlikesi olduğunda çıkar.
        candidate_actions = [ACTION_STRAIGHT, ACTION_RIGHT, ACTION_LEFT]
        best_actions = [action for action in candidate_actions if areas[action] == best_area]

        direction = self.current_direction(state)
        for action in best_actions:
            if self.moves_toward_food(state, turn(direction, action)):
                return action
        return best_actions[0]
