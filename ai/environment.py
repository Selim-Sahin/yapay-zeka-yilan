"""Yapay zekanın oyunu oynadığı "ortam" (environment). Pygame KULLANMAZ.

Pekiştirmeli öğrenmede (reinforcement learning) ajan ile oyun şöyle konuşur:
    state = env.reset()                          # yeni oyun, ilk durum
    state, reward, done = env.step(action)       # bir hamle yap, sonucu al
Ajan sadece `state` (sayı listesi) görür ve `action` (0, 1 veya 2) seçer.
Ekran, piksel, klavye yoktur; bu yüzden eğitim pencere açmadan ve çok hızlı yapılır.
"""

from collections import deque

from snake import config
from snake.game_logic import SnakeGame


# Eylemler. Mutlak yön (yukarı/aşağı...) yerine yılana göre göreli yön kullanıyoruz:
# böylece ters yöne dönmek zaten imkansız olur ve seçenek sayısı 4'ten 3'e iner.
ACTION_STRAIGHT = 0
ACTION_RIGHT = 1
ACTION_LEFT = 2
NUM_ACTIONS = 3

STATE_SIZE = 11  # Sürüm 1'in boyu (kural tabanlı ajan ve ilk DQN modeli)

# Durum sürümleri ve boyları. Sürüm 2 = sürüm 1'in 11 sayısı + 3 adet "ulaşılabilir alan" değeri.
STATE_SIZES = {1: STATE_SIZE, 2: STATE_SIZE + 3}

# Durum listesinin bölümlerinin başladığı indeksler (bkz. get_state).
# Yön ve yem bölümleri aynı sırayı kullanır: sol, sağ, yukarı, aşağı.
DIRECTION_START = 3
FOOD_START = 7
AREA_START = 11  # Sadece sürüm 2'de: düz / sağ / sol için ulaşılabilir alan
STATE_DIRECTION_ORDER = [config.LEFT, config.RIGHT, config.UP, config.DOWN]

# Saat yönünde sıralı yönler. Sağa dönmek listede bir ileri, sola dönmek bir geri gitmektir.
CLOCKWISE = [config.RIGHT, config.DOWN, config.LEFT, config.UP]


def turn(direction, action):
    """Mevcut yöne göreli eylemi uygular ve yeni mutlak yönü döner."""
    index = CLOCKWISE.index(direction)  # Listede bu yönün kaçıncı sırada olduğu
    if action == ACTION_RIGHT:
        index = index + 1
    elif action == ACTION_LEFT:
        index = index - 1
    # `%` mod alma (C'deki gibi). 4'ü geçince ya da -1 olunca liste başa/sona sarar.
    # Not: Python'da -1 % 4 == 3 olur; C'de ise -1 çıkar. Burada Python'un davranışı işimize yarıyor.
    return CLOCKWISE[index % 4]


class SnakeEnv:
    """SnakeGame'i yapay zekanın kullanabileceği bir ortama saran sınıf."""

    def __init__(self, width=config.GRID_WIDTH, height=config.GRID_HEIGHT, rng=None, state_version=1):
        self.game = SnakeGame(width, height, rng)
        self.steps_since_food = 0
        self.state_version = state_version
        self.state_size = STATE_SIZES[state_version]

    def reset(self):
        """Yeni bir oyun başlatır ve ilk durumu döner."""
        self.game.reset()
        self.steps_since_food = 0
        return self.get_state()

    def step(self, action):
        """Bir eylem uygular. (yeni_durum, ödül, oyun_bitti_mi) üçlüsünü döner.

        Python'da bir fonksiyon birden çok değeri tuple olarak dönebilir.
        C'de bunun için bir struct döner ya da işaretçi parametreler kullanırdın.
        """
        game = self.game
        score_before = game.score

        game.change_direction(turn(game.direction, action))
        game.step()
        self.steps_since_food += 1

        if game.game_over:
            return self.get_state(), config.REWARD_DEATH, True

        if game.score > score_before:
            self.steps_since_food = 0
            return self.get_state(), config.REWARD_FOOD, False

        # Çok uzun süre yem yemeden dolaştıysa oyunu bitir (sonsuz döngü koruması).
        if self.steps_since_food > config.MAX_STEPS_PER_LENGTH * len(game.snake):
            game.game_over = True
            return self.get_state(), config.REWARD_DEATH, True

        return self.get_state(), config.REWARD_STEP, False

    def next_head(self, action):
        """Bu eylem yapılırsa başın gireceği kareyi döner (hamleyi yapmadan)."""
        dx, dy = turn(self.game.direction, action)
        head_x, head_y = self.game.snake[0]
        return (head_x + dx, head_y + dy)

    def get_state(self):
        """Ajanın gördüğü durum listesini döner; içeriği state_version'a bağlıdır."""
        state = self.basic_state()
        if self.state_version >= 2:
            state = state + self.reachable_area_features()  # İki listeyi uç uca ekler
        return state

    def reachable_area_features(self):
        """Düz / sağ / sol hamlelerinin her biri için ulaşılabilir alan, 0 ile 1 arasında.

        1.0: "o yöne gidersem önümde en az yılanın boyu kadar boş alan var", yani güvenli.
        0.5: "sadece boyumun yarısı kadar alan var", yani muhtemelen kendimi bir cebe kapatıyorum.
        0.0: "o kare zaten duvar ya da gövde".
        """
        limit = len(self.game.snake)
        features = []
        for action in (ACTION_STRAIGHT, ACTION_RIGHT, ACTION_LEFT):
            area = self.reachable_area(self.next_head(action), limit)
            features.append(area / limit)
        return features

    def reachable_area(self, start, limit):
        """`start` karesine girildikten sonra kaç boş kareye ulaşılabilir? (En fazla `limit` sayılır.)

        "Flood fill" (taşma doldurma): boya kovası gibi, başlangıç karesinden komşulara yayılıp
        duvar ve gövdeye çarpmadan kaç kareye varılabildiğini sayar. Sayı `limit`'e ulaşınca
        durur; böylece açık alanda bütün tahtayı gezmeye gerek kalmaz ve hızlı çalışır.
        Hamleden sonra kuyruk bir kare çekileceği için kuyruğun son karesi boş sayılır
        (is_collision ile aynı kural).
        """
        game = self.game
        if game.is_collision(start):
            return 0

        # set (küme): "bu kare içinde mi?" sorusunu listedeki gibi tek tek bakmadan,
        # çok hızlı cevaplar. C'de bunun için bir hash tablosu yazman gerekirdi.
        blocked = set(game.snake[:-1])
        visited = {start}
        # deque burada kuyruk (queue) olarak kullanılıyor: sondan ekle, baştan al.
        # Bu sıra "genişlik öncelikli arama" (BFS) demek.
        queue = deque([start])
        while queue and len(visited) < limit:
            x, y = queue.popleft()
            for dx, dy in CLOCKWISE:
                neighbor = (x + dx, y + dy)
                if neighbor in visited or neighbor in blocked or game.hits_wall(neighbor):
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        return min(len(visited), limit)

    def basic_state(self):
        """Oyunun durumunu 11 adet 0/1 değerinden oluşan bir listeye çevirir (sürüm 1).

        [0]  önde tehlike var mı      [3]  sola gidiyor     [7]  yem solda
        [1]  sağda tehlike var mı     [4]  sağa gidiyor     [8]  yem sağda
        [2]  solda tehlike var mı     [5]  yukarı gidiyor   [9]  yem yukarıda
                                      [6]  aşağı gidiyor    [10] yem aşağıda
        """
        game = self.game
        head_x, head_y = game.snake[0]
        direction = game.direction

        danger_straight = game.is_collision(self.next_head(ACTION_STRAIGHT))
        danger_right = game.is_collision(self.next_head(ACTION_RIGHT))
        danger_left = game.is_collision(self.next_head(ACTION_LEFT))

        # Tahta doluysa yem yoktur (None); bu durumda dört yem bilgisi de 0 olur.
        if game.food is None:
            food_left = food_right = food_up = food_down = False
        else:
            food_x, food_y = game.food
            food_left = food_x < head_x
            food_right = food_x > head_x
            food_up = food_y < head_y
            food_down = food_y > head_y

        # Yön ve yem bölümlerinin sırası STATE_DIRECTION_ORDER ile aynı olmalı.
        state = [
            danger_straight,
            danger_right,
            danger_left,
            direction == config.LEFT,
            direction == config.RIGHT,
            direction == config.UP,
            direction == config.DOWN,
            food_left,
            food_right,
            food_up,
            food_down,
        ]
        # True/False değerlerini 1/0'a çeviriyoruz. C'de de "doğru" zaten 1'dir.
        return [int(value) for value in state]
