import sys
import time

import numpy as np
import pygame

from snake_env import SnakeEnv
from agent import DQNAgent


# Colors
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 130, 0)
RED = (220, 30, 30)
WHITE = (255, 255, 255)
GRAY = (60, 60, 60)

CELL_SIZE = 30
GRID_SIZE = 20
WINDOW_SIZE = CELL_SIZE * GRID_SIZE
FPS = 10


def main():
    model_path = "checkpoints/final_model.pth"

    env = SnakeEnv(grid_size=GRID_SIZE)
    agent = DQNAgent()
    agent.load(model_path)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Snake DQN")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 24, bold=True)

    high_score = 0
    paused = False
    state = env.reset()
    done = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    state = env.reset()
                    done = False

        if paused or done:
            if done:
                # Show death briefly then restart
                pygame.time.wait(1000)
                state = env.reset()
                done = False
            clock.tick(FPS)
            continue

        action = agent.get_action(state)
        state, _, done, info = env.step(action)
        score = info["score"]
        high_score = max(high_score, score)

        # Draw
        screen.fill(BLACK)

        # Grid lines
        for i in range(GRID_SIZE + 1):
            pygame.draw.line(screen, GRAY, (i * CELL_SIZE, 0), (i * CELL_SIZE, WINDOW_SIZE))
            pygame.draw.line(screen, GRAY, (0, i * CELL_SIZE), (WINDOW_SIZE, i * CELL_SIZE))

        # Food
        fx, fy = env.food
        pygame.draw.rect(screen, RED, (fx * CELL_SIZE, fy * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Snake
        for i, (sx, sy) in enumerate(env.snake):
            color = DARK_GREEN if i == 0 else GREEN
            pygame.draw.rect(screen, color,
                             (sx * CELL_SIZE + 1, sy * CELL_SIZE + 1,
                              CELL_SIZE - 2, CELL_SIZE - 2))

        # Score
        score_text = font.render(f"Score: {score}  Best: {high_score}", True, WHITE)
        screen.blit(score_text, (10, 5))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
