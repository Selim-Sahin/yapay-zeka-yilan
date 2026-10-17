"""snake/game_logic.py için birim testleri. Çalıştırmak için: python -m pytest

pytest, adı `test_` ile başlayan fonksiyonları kendisi bulup çalıştırır.
`assert koşul` yanlışsa test başarısız olur (C'deki assert.h'ye benzer, ama
program durmaz; pytest hatayı raporlar ve sonraki teste geçer).
"""

import random

import pytest

from snake import config
from snake.game_logic import SnakeGame


def make_game(width=10, height=10):
    """Testler için küçük ve her seferinde aynı davranan bir oyun oluşturur."""
    return SnakeGame(width, height, rng=random.Random(0))


def test_initial_state():
    game = make_game()
    assert len(game.snake) == config.START_LENGTH
    assert game.score == 0
    assert game.high_score == 0
    assert not game.game_over
    assert game.food not in game.snake


def test_snake_moves_one_cell_forward():
    game = make_game()
    game.food = (0, 0)  # Yemi yoldan çekiyoruz
    head_x, head_y = game.snake[0]

    game.step()

    assert game.snake[0] == (head_x + 1, head_y)
    assert len(game.snake) == config.START_LENGTH


def test_eating_food_grows_snake_and_increases_score():
    game = make_game()
    head_x, head_y = game.snake[0]
    game.food = (head_x + 1, head_y)  # Yemi tam başın önüne koyuyoruz

    game.step()

    assert len(game.snake) == config.START_LENGTH + 1
    assert game.score == 1
    assert game.food not in game.snake


def test_food_never_spawns_on_snake():
    game = make_game(width=4, height=4)
    # Tahtanın 16 karesinden 15'ini yılanla dolduruyoruz; tek boş kare (3, 3).
    game.snake = [(x, y) for y in range(4) for x in range(4) if (x, y) != (3, 3)]

    for _ in range(50):  # `_`, kullanılmayan döngü değişkeni için gelenektir
        assert game.place_food() == (3, 3)


def test_place_food_returns_none_when_board_is_full():
    game = make_game(width=2, height=2)
    game.snake = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert game.place_food() is None


def test_reverse_direction_is_ignored():
    game = make_game()
    game.change_direction(config.LEFT)  # Sağa giderken sola basmak
    game.step()
    assert game.direction == config.RIGHT
    assert not game.game_over


def test_quick_double_turn_cannot_reverse():
    game = make_game()
    game.food = (0, 0)
    # Aynı adım içinde hızlıca önce yukarı, sonra sola basılıyor.
    game.change_direction(config.UP)
    game.change_direction(config.LEFT)
    game.step()
    assert game.direction == config.UP
    assert not game.game_over


# parametrize, aynı testi farklı girdilerle tekrar tekrar çalıştırır: burada
# dört yön için dört ayrı test oluşur. `@...` satırı bir "dekoratör": altındaki
# fonksiyona ek davranış ekler.
@pytest.mark.parametrize("direction", [config.UP, config.DOWN, config.LEFT, config.RIGHT])
def test_hitting_wall_ends_game(direction):
    game = make_game(width=5, height=5)
    game.snake = [(2, 2)]
    game.food = None
    game.direction = direction
    game.next_direction = direction

    game.step()  # (2, 2) -> kenar karesi
    game.step()  # kenar karesi -> son kare
    assert not game.game_over
    game.step()  # tahtanın dışı
    assert game.game_over


def test_hitting_own_body_ends_game():
    game = make_game()
    # Baş (5, 5)'te, sağa gidiyor. Aşağısındaki (5, 6) gövdenin bir parçası.
    game.snake = [(5, 5), (4, 5), (4, 6), (5, 6), (6, 6), (6, 5)]
    game.food = (0, 0)
    game.change_direction(config.DOWN)
    game.step()
    assert game.game_over


def test_moving_into_tail_cell_is_allowed():
    game = make_game()
    # Yılan yukarı gidiyor ve küçük bir kare çiziyor. Sola dönünce baş,
    # o an boşalacak olan kuyruk karesine (4, 5) giriyor.
    game.snake = [(5, 5), (5, 6), (4, 6), (4, 5)]
    game.food = (0, 0)
    game.direction = config.UP
    game.next_direction = config.LEFT
    game.step()
    assert not game.game_over
    assert game.snake[0] == (4, 5)


def test_step_does_nothing_after_game_over():
    game = make_game()
    game.game_over = True
    snake_before = list(game.snake)  # list(...) bir kopya oluşturur
    game.step()
    assert game.snake == snake_before


def test_high_score_follows_best_score():
    game = make_game()
    head_x, head_y = game.snake[0]
    game.food = (head_x + 1, head_y)
    game.step()
    assert game.high_score == 1


def test_reset_restarts_game_but_keeps_high_score():
    game = make_game()
    game.score = 7
    game.high_score = 7
    game.game_over = True

    game.reset()

    assert game.score == 0
    assert not game.game_over
    assert len(game.snake) == config.START_LENGTH
    assert game.direction == config.RIGHT
    assert game.high_score == 7
