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

# --- Öğrenen ajan (DQN) ayarları (ai/model.py, ai/agent.py, train.py) ---

HIDDEN_SIZE = 256         # Sinir ağının gizli katmanındaki nöron sayısı
LEARNING_RATE = 0.001     # Her eğitim adımında ağırlıkların ne kadar değişeceği
GAMMA = 0.9               # Gelecekteki ödüllerin bugünkü değeri (0: sadece şimdi, 1: hepsi eşit)
MEMORY_SIZE = 100_000     # Hatırlanan en fazla deneyim sayısı (`_` sadece okunabilirlik için)
BATCH_SIZE = 1000         # Her oyun sonunda hafızadan rastgele seçilip öğrenilen deneyim sayısı

# Keşif (exploration): ajan başta çoğunlukla rastgele oynar, zamanla öğrendiğine güvenir.
EPSILON_START = 1.0       # İlk oyunda hamlelerin %100'ü rastgele
EPSILON_MIN = 0.01        # En sonunda hamlelerin %1'i rastgele
EPSILON_DECAY_GAMES = 200 # Bu kadar oyun boyunca rastgelelik düzenli olarak azalır

TRAIN_MINUTES = 40        # Eğitimin süre sınırı
EVAL_EVERY_GAMES = 50     # Kaç oyunda bir ajanın gerçek başarısı ölçülsün
EVAL_GAMES = 20           # Her ölçümde kaç oyun oynansın
EVAL_SEED = 1000          # Ölçüm oyunlarının seed'i (son karşılaştırmada kullanılan 0'dan farklı)

MODELS_DIR = "models"
BEST_MODEL_PATH = "models/best.pth"
LAST_MODEL_PATH = "models/last.pth"
TRAINING_CURVE_PATH = "assets/training_curve.png"
