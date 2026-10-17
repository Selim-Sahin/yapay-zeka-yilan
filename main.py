"""Oyunu başlatan dosya. Çalıştırmak için: python main.py"""

from snake.game import SnakeGame


def main():
    game = SnakeGame()
    game.run()


# Bu dosya doğrudan çalıştırılırsa (python main.py) main() çağrılır.
# Başka bir dosya bunu import ederse çağrılmaz. C'deki main()'in
# programın giriş noktası olmasına benzer.
if __name__ == "__main__":
    main()
