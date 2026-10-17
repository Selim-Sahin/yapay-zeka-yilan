"""Oyunu ekrana çizen kod. Kurallara dokunmaz, sadece SnakeGame'in durumunu okur."""

import pygame

from snake import config


class Renderer:
    """Pencereyi, fontları ve çizim işlerini yöneten sınıf."""

    def __init__(self):
        self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        # Font(None, ...) pygame'in yerleşik fontunu kullanır; Türkçe harfleri destekler.
        self.title_font = pygame.font.Font(None, config.TITLE_FONT_SIZE)
        self.text_font = pygame.font.Font(None, config.TEXT_FONT_SIZE)

        # Game Over ekranında oyunun üstüne serilen yarı saydam siyah katman.
        # SRCALPHA, yüzeyin saydamlık bilgisi taşıyabilmesini sağlar.
        self.overlay = pygame.Surface((config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.SRCALPHA)
        self.overlay.fill(config.OVERLAY_COLOR)

    def draw(self, game):
        """Oyunun o anki halini çizer ve ekrana basar."""
        self.screen.fill(config.BACKGROUND_COLOR)

        if game.food is not None:  # `is not None`, C'deki `!= NULL` kontrolü gibi
            self.draw_cell(game.food, config.FOOD_COLOR)

        for cell in game.snake[1:]:
            self.draw_cell(cell, config.SNAKE_BODY_COLOR)
        self.draw_cell(game.snake[0], config.SNAKE_HEAD_COLOR)

        if game.game_over:
            self.draw_game_over(game)

        self.update_title(game)

        # Her şey önce görünmeyen bir arka tamponda çizilir, flip() ile tek
        # seferde ekrana basılır. Bu sayede ekran titremez (double buffering).
        pygame.display.flip()

    def draw_cell(self, cell, color):
        """Izgaradaki bir kareyi (x, y) verilen renkle boyar."""
        x, y = cell
        rect = pygame.Rect(x * config.CELL_SIZE, y * config.CELL_SIZE, config.CELL_SIZE, config.CELL_SIZE)
        pygame.draw.rect(self.screen, color, rect)

    def draw_game_over(self, game):
        """Oyunun üstüne karartma katmanı ve Game Over yazılarını çizer."""
        # blit: bir yüzeyi başka bir yüzeyin üstüne kopyalar (C'deki memcpy'nin
        # görüntüler için olanı gibi düşünebilirsin).
        self.screen.blit(self.overlay, (0, 0))

        center_y = config.WINDOW_HEIGHT // 2
        self.draw_centered_text("Oyun Bitti", self.title_font, center_y - config.TEXT_LINE_SPACING)
        self.draw_centered_text(f"Skor: {game.score}   Rekor: {game.high_score}", self.text_font, center_y)
        self.draw_centered_text("Yeniden başlamak için R tuşuna bas", self.text_font, center_y + config.TEXT_LINE_SPACING)

    def draw_centered_text(self, text, font, center_y):
        """Yazıyı yatayda ortalayarak, verilen yüksekliğe çizer."""
        # render(), yazıyı bir resme (Surface) dönüştürür. True: kenar yumuşatma açık.
        text_surface = font.render(text, True, config.TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(config.WINDOW_WIDTH // 2, center_y))
        self.screen.blit(text_surface, text_rect)

    def update_title(self, game):
        """Skoru, rekoru ve oyun durumunu pencere başlığına yazar."""
        # f"..." bir "f-string": C'deki sprintf(title, "Skor: %d", score) gibi.
        if game.game_over:
            title = f"{config.WINDOW_TITLE} | Oyun bitti! Skor: {game.score} | Rekor: {game.high_score}"
        else:
            title = f"{config.WINDOW_TITLE} | Skor: {game.score} | Rekor: {game.high_score}"
        pygame.display.set_caption(title)
