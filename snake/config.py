"""Oyunun tüm sabit ayarları burada durur.

C'deki #define satırlarının karşılığı gibi düşünebilirsin. Python'da gerçek
"sabit" yoktur; BÜYÜK_HARFLE yazılan isimler "bunu değiştirme" anlamına gelen
bir gelenektir.
"""

# Izgara (oyun tahtası) ayarları
CELL_SIZE = 20      # Bir karenin kenarı, piksel cinsinden
GRID_WIDTH = 30     # Yatayda kaç kare var
GRID_HEIGHT = 20    # Dikeyde kaç kare var

WINDOW_WIDTH = CELL_SIZE * GRID_WIDTH
WINDOW_HEIGHT = CELL_SIZE * GRID_HEIGHT
WINDOW_TITLE = "Yılan"

FPS = 10            # Saniyedeki adım sayısı, yani yılanın hızı
START_LENGTH = 3    # Yılanın başlangıçtaki boyu

# Renkler: (kırmızı, yeşil, mavi), her biri 0-255 arası
BACKGROUND_COLOR = (30, 30, 30)
SNAKE_HEAD_COLOR = (0, 220, 0)
SNAKE_BODY_COLOR = (0, 150, 0)
FOOD_COLOR = (220, 50, 50)
TEXT_COLOR = (240, 240, 240)
# Dördüncü sayı "alpha", yani saydamlık: 0 tamamen saydam, 255 tamamen opak.
OVERLAY_COLOR = (0, 0, 0, 170)

# Game Over ekranındaki yazılar
TITLE_FONT_SIZE = 64
TEXT_FONT_SIZE = 32
TEXT_LINE_SPACING = 50  # Satırlar arasındaki dikey mesafe, piksel

# Yönler: (dx, dy). Dikkat: ekranda y aşağı doğru artar, bu yüzden UP = -1.
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# --- Yapay zeka ortamı ayarları (ai/environment.py) ---

# Ödüller: ajan bunları en büyük yapmaya çalışır.
REWARD_FOOD = 10
REWARD_DEATH = -10
REWARD_STEP = 0

# Yem yemeden en fazla (MAX_STEPS_PER_LENGTH * yılan boyu) adım atılabilir.
# Bu sınır olmazsa ajan sonsuza kadar daire çizebilir ve oyun hiç bitmez.
MAX_STEPS_PER_LENGTH = 100

AI_FPS = 20  # play_ai.py'de ajanı izlerken oyun hızı
