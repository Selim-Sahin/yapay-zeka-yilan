"""record_demo.py için testler. Pencere açılmaz (SDL_VIDEODRIVER=dummy)."""

import random

import pygame

import record_demo
from ai.environment import SnakeEnv
from ai.rule_based_agent import RuleBasedAgent
from snake import config
from snake.renderer import Renderer


def test_record_frames_returns_window_sized_images():
    pygame.init()
    try:
        renderer = Renderer()
        env = SnakeEnv(rng=random.Random(0))
        frames = record_demo.record_frames(RuleBasedAgent(), env, renderer, max_frames=5)
    finally:
        pygame.quit()

    assert len(frames) == 5  # Kısa kayıtta oyun bitmez; tam max_frames kare gelir
    assert frames[0].size == (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
