import sys
import time

import numpy as np
import pygame

from snake_env import SnakeEnv, Direction, DIR_VECTORS
from agent import DQNAgent


# Colors
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 130, 0)
LIGHT_GREEN = (100, 230, 100)
RED = (220, 30, 30)
RED_DARK = (180, 20, 20)
WHITE = (255, 255, 255)
GRAY = (60, 60, 60)
EYE_WHITE = (240, 240, 240)
EYE_BLACK = (20, 20, 20)
TONGUE = (220, 50, 80)

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

        # Food (apple with highlight)
        fx, fy = env.food
        food_cx = fx * CELL_SIZE + CELL_SIZE // 2
        food_cy = fy * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, RED, (food_cx, food_cy), CELL_SIZE // 2 - 1)
        pygame.draw.circle(screen, RED_DARK, (food_cx, food_cy), CELL_SIZE // 2 - 1, 2)
        # Apple highlight
        pygame.draw.circle(screen, (255, 100, 100), (food_cx - 3, food_cy - 3), 4)

        # Snake body
        for i, (sx, sy) in enumerate(env.snake):
            if i == 0:
                continue  # draw head separately
            # Rounded body segments with subtle gradient
            body_rect = pygame.Rect(sx * CELL_SIZE + 1, sy * CELL_SIZE + 1,
                                    CELL_SIZE - 2, CELL_SIZE - 2)
            pygame.draw.rect(screen, GREEN, body_rect, border_radius=6)
            # Inner highlight
            inner = pygame.Rect(sx * CELL_SIZE + 3, sy * CELL_SIZE + 3,
                                CELL_SIZE - 6, CELL_SIZE - 6)
            pygame.draw.rect(screen, LIGHT_GREEN, inner, border_radius=4)

        # Snake head with face
        hx, hy = env.snake[0]
        head_rect = pygame.Rect(hx * CELL_SIZE + 1, hy * CELL_SIZE + 1,
                                CELL_SIZE - 2, CELL_SIZE - 2)
        pygame.draw.rect(screen, DARK_GREEN, head_rect, border_radius=8)

        # Eyes and tongue based on direction
        cx = hx * CELL_SIZE + CELL_SIZE // 2
        cy = hy * CELL_SIZE + CELL_SIZE // 2
        dx, dy = DIR_VECTORS[env.direction]
        eye_r = 4
        pupil_r = 2

        # Perpendicular to direction for eye placement
        perp_x, perp_y = -dy, dx
        eye_offset = 6
        eye_fwd = 4

        for side in (-1, 1):
            ex = cx + dx * eye_fwd + perp_x * eye_offset * side
            ey = cy + dy * eye_fwd + perp_y * eye_offset * side
            pygame.draw.circle(screen, EYE_WHITE, (int(ex), int(ey)), eye_r)
            pygame.draw.circle(screen, EYE_BLACK, (int(ex + dx * 1), int(ey + dy * 1)), pupil_r)

        # Tongue (flickers out in front)
        tongue_base_x = cx + dx * (CELL_SIZE // 2)
        tongue_base_y = cy + dy * (CELL_SIZE // 2)
        tongue_tip_x = tongue_base_x + dx * 8
        tongue_tip_y = tongue_base_y + dy * 8
        pygame.draw.line(screen, TONGUE,
                         (int(tongue_base_x), int(tongue_base_y)),
                         (int(tongue_tip_x), int(tongue_tip_y)), 2)
        # Forked tongue tips
        fork = 3
        pygame.draw.line(screen, TONGUE,
                         (int(tongue_tip_x), int(tongue_tip_y)),
                         (int(tongue_tip_x + dx * 3 + perp_x * fork),
                          int(tongue_tip_y + dy * 3 + perp_y * fork)), 2)
        pygame.draw.line(screen, TONGUE,
                         (int(tongue_tip_x), int(tongue_tip_y)),
                         (int(tongue_tip_x + dx * 3 - perp_x * fork),
                          int(tongue_tip_y + dy * 3 - perp_y * fork)), 2)

        # Score
        score_text = font.render(f"Score: {score}  Best: {high_score}", True, WHITE)
        screen.blit(score_text, (10, 5))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
