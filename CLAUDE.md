# Yapay Zeka Yılan — Proje Kuralları

## Proje hakkında
Pygame ile yazılmış bir Yılan oyunu ve bu oyunu Deep Q-Learning ile kendi kendine
oynamayı öğrenen bir yapay zeka ajanı. Amaç: GitHub/LinkedIn portfolyo projesi.

## Geliştirici profili
- C biliyor, Python'da yeni. Python'a özgü yapıları (list comprehension, sınıflar,
  `__init__`, `self`, dekoratörler, `with` vb.) ilk kullanıldıkları yerde kısaca açıkla.
  Mümkünse C'deki karşılığıyla kıyasla.
- Açıklamalar Türkçe ve sade olsun. Her satırı değil, önemli yerleri açıkla.

## Kod kuralları
- Dil: Python 3.12
- Değişken, fonksiyon, sınıf ve dosya adları **İngilizce** (`snake_body`, `move_snake()`).
- Yorum satırları ve docstring'ler **Türkçe**.
- PEP 8 stiline uy: `snake_case` fonksiyon/değişken, `PascalCase` sınıf, `UPPER_CASE` sabit.
- Sabitler (renkler, boyutlar, hız) `snake/config.py` içinde durur; kodun içine sayı gömme.
- Fonksiyonlar kısa ve tek işli olsun. Gereksiz soyutlama ve "akıllıca" tek satırlıklar yazma.

## Mimari kuralı (önemli)
- `snake/game_logic.py` **asla** `pygame` import etmez. Oyun mantığı saf Python'dur.
- Çizim sadece `snake/renderer.py` içindedir.
- Yapay zeka (`ai/`) oyuna sadece `game_logic` üzerinden erişir; pencere açmadan çalışabilmelidir.

## Test
- Test aracı: `pytest`. Çalıştırma: `python -m pytest`
- `game_logic` ve `environment` için birim testleri zorunludur.
- Bir aşama bitmeden önce tüm testler geçmelidir.

## Çalışma şekli
- Proje aşamalarla ilerler. Bir sonraki aşamaya geçmeden önce kullanıcının onayını al.
- Her aşamanın sonunda çalışan bir şey olmalı.
- Her aşama sonunda Conventional Commits formatında **İngilizce** bir commit mesajı öner
  (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`). Commit'i kullanıcı onaylarsa at.

## Komutlar
- Sanal ortam: `python -m venv .venv` ve `.venv\Scripts\Activate.ps1`
- Kurulum: `pip install -r requirements.txt`
- Oyna: `python main.py`
- Eğit: `python train.py`
- Ajanı izle: `python play_ai.py`
- Test: `python -m pytest`

## Git'e girmeyecekler
`.venv/`, `__pycache__/`, `models/*.pth` (sadece `best.pth` hariç), geçici grafikler.
