"""Ajanın oyunu oynamasını pencerede izlemek için. Çalıştırmak için:
    python play_ai.py --agent rule
    python play_ai.py --agent dqn --model models/best.pth

Oyun bitince R ile yeni oyun başlar, pencereyi kapatınca program biter.
"""

import argparse

import pygame

from ai.environment import SnakeEnv
from ai.evaluate import make_agent
from snake import config
from snake.renderer import Renderer


def handle_events(env):
    """Pencere kapatılırsa False döner; oyun bittiyse R ile yeniden başlatır."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r and env.game.game_over:
            env.reset()
    return True


def main():
    parser = argparse.ArgumentParser(description="Ajanın oynayışını izle.")
    parser.add_argument("--agent", default="rule", help="rule veya dqn")
    parser.add_argument("--model", default=config.BEST_MODEL_PATH, help="dqn için model dosyası")
    parser.add_argument("--fps", type=int, default=config.AI_FPS, help="oyun hızı")
    args = parser.parse_args()

    agent = make_agent(args.agent, args.model)
    env = SnakeEnv(state_version=agent.state_version)

    pygame.init()
    renderer = Renderer()
    clock = pygame.time.Clock()

    state = env.reset()
    running = True
    while running:
        running = handle_events(env)
        if env.game.game_over:
            state = env.get_state()  # R ile yeni oyun başlarsa güncel durumu al
        else:
            # Tuşlar yerine ajan karar veriyor; gerisi main.py ile aynı.
            state, reward, done = env.step(agent.choose_action(state))
        renderer.draw(env.game)
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
