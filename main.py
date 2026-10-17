"""Oyunu başlatan dosya. Çalıştırmak için: python main.py

Burası üç parçayı birleştirir: klavyeyi okur, kurallara (SnakeGame) iletir,
sonucu çizdirir (Renderer).
"""

import pygame

from snake import config
from snake.game_logic import SnakeGame
from snake.renderer import Renderer


# Hangi tuşun hangi yöne karşılık geldiği. Bu bir "dict" (sözlük): anahtar -> değer
# eşlemesi. C'de bunu bir switch-case ile yapardın.
KEY_TO_DIRECTION = {
    pygame.K_UP: config.UP,
    pygame.K_DOWN: config.DOWN,
    pygame.K_LEFT: config.LEFT,
    pygame.K_RIGHT: config.RIGHT,
}


def handle_events(game):
    """Klavye ve pencere olaylarını işler. Pencere kapatılırsa False döner."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key in KEY_TO_DIRECTION:
                game.change_direction(KEY_TO_DIRECTION[event.key])
            elif event.key == pygame.K_r and game.game_over:
                game.reset()
    return True


def main():
    pygame.init()
    game = SnakeGame()
    renderer = Renderer()
    clock = pygame.time.Clock()

    # Ana oyun döngüsü: olayları oku, oyunu ilerlet, çiz, bekle.
    running = True
    while running:
        running = handle_events(game)
        game.step()
        renderer.draw(game)
        # Döngüyü saniyede en fazla FPS kez çalışacak şekilde bekletir (C'deki sleep() gibi).
        clock.tick(config.FPS)

    pygame.quit()


# Bu dosya doğrudan çalıştırılırsa (python main.py) main() çağrılır; başka bir
# dosya bunu import ederse çağrılmaz. C'deki main()'in giriş noktası olmasına benzer.
if __name__ == "__main__":
    main()
