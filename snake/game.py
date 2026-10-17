"""Aşama 1: Yılan oyununun tamamı (kurallar + çizim) bu dosyada.

Aşama 2'de kuralları `game_logic.py`, çizimi `renderer.py` dosyasına ayıracağız.
"""

import random

import pygame

from snake import config


# Hangi tuşun hangi yöne karşılık geldiği.
# Bu bir "dict" (sözlük): anahtar -> değer eşlemesi. C'de bunu bir
# switch-case ile yapardın; burada KEY_TO_DIRECTION[tuş] yazmak yeterli.
KEY_TO_DIRECTION = {
    pygame.K_UP: config.UP,
    pygame.K_DOWN: config.DOWN,
    pygame.K_LEFT: config.LEFT,
    pygame.K_RIGHT: config.RIGHT,
}


class SnakeGame:
    """Oyunun durumunu ve kurallarını bir arada tutan sınıf.

    Sınıfı, C'deki bir struct ile o struct üzerinde çalışan fonksiyonların
    tek pakette toplanmış hali olarak düşünebilirsin. Her metodun ilk
    parametresi olan `self`, C'de fonksiyona verdiğin `struct Game *game`
    işaretçisinin karşılığıdır. Farkı şu: Python onu otomatik geçirir;
    `game.step()` yazmak, C'deki `step(&game)` çağrısıyla aynı şeydir.
    """

    def __init__(self):
        # __init__, nesne oluşturulurken bir kez çalışır (C'deki init_game()
        # gibi). `self.x = ...` yazmak, struct'a bir alan eklemek demektir.
        pygame.init()
        self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # Yılan, (x, y) koordinatlarından oluşan bir liste; ilk eleman baştır.
        # - Python listesi, C'deki dizinin boyu değişebilen halidir.
        # - (x, y) bir "tuple": değiştirilemeyen küçük bir paket,
        #   C'deki struct { int x; int y; } gibi.
        # - `//` tam sayı bölmesidir (C'de iki int'i `/` ile bölmek gibi).
        center_x = config.GRID_WIDTH // 2
        center_y = config.GRID_HEIGHT // 2

        # Aşağıdaki satır bir "list comprehension". C'deki şu döngünün kısa yazımı:
        #   for (i = 0; i < START_LENGTH; i++) snake[i] = (center_x - i, center_y);
        # Sonuç: baş ortada, gövde onun solunda. Yılan sağa bakarak başlıyor.
        self.snake = [(center_x - i, center_y) for i in range(config.START_LENGTH)]

        self.direction = config.RIGHT       # Son adımda gidilen yön
        self.next_direction = config.RIGHT  # Bir sonraki adımda gidilecek yön
        self.score = 0
        self.game_over = False
        self.food = self.place_food()
        self.update_title()

    def place_food(self):
        """Yemi, yılanın olmadığı rastgele bir kareye koyar."""
        empty_cells = []
        for x in range(config.GRID_WIDTH):
            for y in range(config.GRID_HEIGHT):
                # `in`, listede arama yapar. C'de bunun için elle döngü yazardın.
                if (x, y) not in self.snake:
                    empty_cells.append((x, y))

        if not empty_cells:
            return None  # Yılan bütün tahtayı kapladı, yem koyacak yer yok
        return random.choice(empty_cells)

    def handle_events(self):
        """Klavye ve pencere olaylarını işler. Pencere kapatılırsa False döner."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key in KEY_TO_DIRECTION:
                self.change_direction(KEY_TO_DIRECTION[event.key])
        return True

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
        self.direction = self.next_direction

        # Tuple'ı iki değişkene "açma": head_x = snake[0].x; head_y = snake[0].y gibi.
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        if self.hits_wall(new_head):
            self.end_game()
            return

        # Yılanı ilerletmenin hilesi: başa yeni bir kare ekle, sondan bir kare sil.
        # Yem yendiyse sondan silmeyiz; böylece yılan bir kare uzar.
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.food = self.place_food()
            self.update_title()
        else:
            self.snake.pop()

        # Baş, gövdenin geri kalanına çarptı mı? `self.snake[1:]` bir "slice":
        # listenin 1. elemanından sonuna kadar olan kısmı (yani baş hariç her şey).
        # Kontrolü kuyruk silindikten sonra yapıyoruz; böylece yılan, o an
        # boşalan kuyruk karesine güvenle girebiliyor.
        if new_head in self.snake[1:]:
            self.end_game()

    def hits_wall(self, position):
        """Verilen kare tahtanın dışındaysa True döner."""
        x, y = position
        # Python'da `or`, C'deki `||` operatörüdür.
        return x < 0 or x >= config.GRID_WIDTH or y < 0 or y >= config.GRID_HEIGHT

    def end_game(self):
        self.game_over = True
        self.update_title()

    def update_title(self):
        """Skoru ve oyun durumunu pencere başlığına yazar."""
        # f"..." bir "f-string": süslü parantez içindeki değer metne yerleşir.
        # C'deki sprintf(title, "Skor: %d", score) ile aynı işi yapar.
        if self.game_over:
            title = f"{config.WINDOW_TITLE} | Oyun bitti! Skor: {self.score} | Çıkmak için pencereyi kapat"
        else:
            title = f"{config.WINDOW_TITLE} | Skor: {self.score}"
        pygame.display.set_caption(title)

    def draw(self):
        """Ekranı temizler, yemi ve yılanı çizer."""
        self.screen.fill(config.BACKGROUND_COLOR)

        if self.food is not None:  # `is not None`, C'deki `!= NULL` kontrolü gibi
            self.draw_cell(self.food, config.FOOD_COLOR)

        for cell in self.snake[1:]:
            self.draw_cell(cell, config.SNAKE_BODY_COLOR)
        self.draw_cell(self.snake[0], config.SNAKE_HEAD_COLOR)

        # Her şey önce görünmeyen bir arka tamponda çizilir, flip() ile tek
        # seferde ekrana basılır. Bu sayede ekran titremez (double buffering).
        pygame.display.flip()

    def draw_cell(self, cell, color):
        """Izgaradaki bir kareyi (x, y) verilen renkle boyar."""
        x, y = cell
        rect = pygame.Rect(x * config.CELL_SIZE, y * config.CELL_SIZE, config.CELL_SIZE, config.CELL_SIZE)
        pygame.draw.rect(self.screen, color, rect)

    def run(self):
        """Ana oyun döngüsü: olayları oku, oyunu ilerlet, çiz, bekle."""
        running = True
        while running:
            running = self.handle_events()
            if not self.game_over:
                self.step()
            self.draw()
            # Döngüyü saniyede en fazla FPS kez çalışacak şekilde bekletir.
            # C'de her turun sonuna bir sleep() koymaya benzer.
            self.clock.tick(config.FPS)
        pygame.quit()
