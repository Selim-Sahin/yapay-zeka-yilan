"""Yılan oyununun kuralları. Bu dosya pygame KULLANMAZ.

Neden? Aşama 3-4'teki yapay zeka oyunu binlerce kez, pencere açmadan
oynayacak. Kurallar çizimden bağımsız olursa bu mümkün olur; ayrıca kuralları
pytest ile kolayca test edebiliriz. C'de kuralları game.c, çizimi render.c
dosyasına koymaya benzer.
"""

import random

from snake import config


class SnakeGame:
    """Oyunun durumunu ve kurallarını tutan sınıf.

    Sınıf, C'deki bir struct ile o struct üzerinde çalışan fonksiyonların tek
    pakette toplanmış halidir. `self`, C'deki `struct Game *game` işaretçisinin
    karşılığıdır; `game.step()` yazmak `step(&game)` çağrısıyla aynı şeydir.
    """

    def __init__(self, width=config.GRID_WIDTH, height=config.GRID_HEIGHT, rng=None):
        # Parametrelerdeki `=` varsayılan değerdir: SnakeGame() yazınca config'deki
        # boyutlar kullanılır, testlerde ise küçük bir tahta verebiliriz.
        # C'de bu özellik yoktur; her parametreyi her çağrıda vermek gerekir.
        self.width = width
        self.height = height

        # rng: rastgele sayı üreteci. Testlerde sabit bir "seed" ile oluşturulursa
        # yem her seferinde aynı yerlere düşer (C'deki srand(42) gibi).
        if rng is None:
            rng = random.Random()
        self.rng = rng

        self.high_score = 0  # reset() bunu sıfırlamaz; rekor oyunlar arasında korunur
        self.reset()

    def reset(self):
        """Oyunu başlangıç durumuna getirir. Rekor skor korunur."""
        center_x = self.width // 2   # `//` tam sayı bölmesi (C'de int / int gibi)
        center_y = self.height // 2

        # Yılan, (x, y) tuple'larından oluşan bir liste; ilk eleman baştır.
        # Bu satır bir "list comprehension", C'deki şu döngünün kısa yazımı:
        #   for (i = 0; i < START_LENGTH; i++) snake[i] = (center_x - i, center_y);
        self.snake = [(center_x - i, center_y) for i in range(config.START_LENGTH)]

        self.direction = config.RIGHT       # Son adımda gerçekten gidilen yön
        self.next_direction = config.RIGHT  # Bir sonraki adımda gidilecek yön
        self.score = 0
        self.game_over = False
        self.food = self.place_food()

    def place_food(self):
        """Yılanın olmadığı rastgele bir kareyi döner; yer kalmadıysa None döner."""
        empty_cells = []
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) not in self.snake:
                    empty_cells.append((x, y))

        if not empty_cells:
            return None  # None, C'deki NULL'a benzer: "değer yok" demek
        return self.rng.choice(empty_cells)

    def change_direction(self, new_direction):
        """Yönü değiştirir; yılanın kendi üstüne geri dönmesine izin vermez."""
        # Ters yönün iki bileşeni de işaret olarak terstir: RIGHT=(1, 0), LEFT=(-1, 0).
        # Karşılaştırmayı `next_direction` ile değil, son adımda gerçekten gidilen
        # `direction` ile yapıyoruz. Yoksa sağa giderken aynı adım içinde hızlıca
        # önce yukarı, sonra sola basınca yılan kendi üstüne dönebilirdi.
        opposite = (-self.direction[0], -self.direction[1])
        if new_direction != opposite:
            self.next_direction = new_direction

    def step(self):
        """Yılanı bir kare ilerletir ve oyunun kurallarını uygular."""
        if self.game_over:
            return

        self.direction = self.next_direction
        head_x, head_y = self.snake[0]   # Tuple'ı iki değişkene "açma"
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        if self.hits_wall(new_head):
            self.game_over = True
            return

        # Başa yeni bir kare ekle, sondan bir kare sil: yılan ilerler.
        # Yem yendiyse sondan silmeyiz; böylece yılan bir kare uzar.
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.high_score = max(self.high_score, self.score)
            self.food = self.place_food()
        else:
            self.snake.pop()

        # `self.snake[1:]` bir "slice": baş hariç listenin geri kalanı.
        # Kontrolü kuyruk silindikten sonra yapıyoruz; böylece yılan, o an
        # boşalan kuyruk karesine güvenle girebiliyor.
        if new_head in self.snake[1:]:
            self.game_over = True

    def hits_wall(self, position):
        """Verilen kare tahtanın dışındaysa True döner."""
        x, y = position
        return x < 0 or x >= self.width or y < 0 or y >= self.height
