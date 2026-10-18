"""Ajanın oynayışını pencere açmadan bir GIF dosyasına kaydeder. Çalıştırmak için:
    python record_demo.py                                  # DQN v2, assets/demo.gif
    python record_demo.py --agent rule_ff --output plots/rule_ff.gif

play_ai.py ile aynı çizim kodunu (Renderer) kullanır; tek fark, kareler ekrana
değil bir listeye gider ve sonunda Pillow ile GIF olarak kaydedilir.
"""

import argparse
import os
import random

# SDL'e "gerçek pencere açma, her şeyi bellekte çiz" diyoruz. Bu satır pygame
# içe aktarılmadan ÖNCE çalışmalı; pygame ayarı ilk yüklendiğinde okur.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
from PIL import Image

from ai.environment import SnakeEnv
from ai.evaluate import make_agent
from snake import config
from snake.renderer import Renderer


def capture_frame(surface):
    """Pygame yüzeyini (Surface) bir Pillow resmine kopyalar."""
    # tobytes(): pikselleri ham bayt dizisi olarak verir (C'deki bir unsigned char
    # dizisi gibi). frombytes() aynı baytları Pillow'un resim nesnesine çevirir.
    data = pygame.image.tobytes(surface, "RGB")
    return Image.frombytes("RGB", surface.get_size(), data)


def record_frames(agent, env, renderer, max_frames):
    """Tek bir oyunu oynatır ve her adımın görüntüsünü bir listeye ekler."""
    state = env.reset()
    frames = []
    while len(frames) < max_frames and not env.game.game_over:
        state, reward, done = env.step(agent.choose_action(state))
        renderer.draw(env.game)
        frames.append(capture_frame(renderer.screen))
    if env.game.game_over:
        # Oyun Bitti ekranı hemen kaybolmasın diye son kareyi birkaç kez tekrarla.
        frames.extend([frames[-1]] * config.DEMO_GAME_OVER_FRAMES)
    return frames


def save_gif(frames, path, fps):
    """Kareleri döngüyle tekrar eden bir GIF olarak kaydeder."""
    frames[0].save(
        path,
        save_all=True,             # Sadece ilk kareyi değil, hepsini kaydet
        append_images=frames[1:],
        duration=1000 // fps,      # Her karenin ekranda kalma süresi, milisaniye
        loop=0,                    # 0: sonsuz döngü
        optimize=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Ajanın oynayışını GIF olarak kaydet.")
    parser.add_argument("--agent", default="dqn", help="rule, rule_ff veya dqn")
    parser.add_argument("--model", default=config.DEMO_MODEL_PATH, help="dqn için model dosyası")
    parser.add_argument("--seed", type=int, default=0, help="yem dizisinin seed'i")
    parser.add_argument("--max-frames", type=int, default=config.DEMO_MAX_FRAMES)
    parser.add_argument("--fps", type=int, default=config.DEMO_FPS)
    parser.add_argument("--output", default=config.DEMO_OUTPUT_PATH)
    args = parser.parse_args()

    agent = make_agent(args.agent, args.model)
    env = SnakeEnv(rng=random.Random(args.seed), state_version=agent.state_version)

    pygame.init()
    renderer = Renderer()
    frames = record_frames(agent, env, renderer, args.max_frames)
    pygame.quit()

    save_gif(frames, args.output, args.fps)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"{args.output}: {len(frames)} kare, skor {env.game.score}, {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
