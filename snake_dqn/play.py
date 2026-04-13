import sys
import math

import numpy as np
import pygame

from snake_env import SnakeEnv, Direction, DIR_VECTORS
from agent import DQNAgent


# Colors
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 130, 0)
LIGHT_GREEN = (100, 230, 100)
WHITE = (255, 255, 255)
GRAY = (60, 60, 60)
EYE_WHITE = (240, 240, 240)
EYE_BLACK = (20, 20, 20)
TONGUE = (220, 50, 80)
YELLOW = (255, 220, 50)

# Apple colors
APPLE_RED = (220, 30, 30)
APPLE_DARK = (160, 15, 15)
APPLE_SHINE = (255, 130, 130)
APPLE_STEM = (90, 60, 30)
APPLE_LEAF = (50, 180, 50)

CELL_SIZE = 30
GRID_SIZE = 20
WINDOW_SIZE = CELL_SIZE * GRID_SIZE
FPS = 10


def draw_apple(screen, fx, fy):
    """Draw a nice apple with stem, leaf, and shine."""
    cx = fx * CELL_SIZE + CELL_SIZE // 2
    cy = fy * CELL_SIZE + CELL_SIZE // 2
    r = CELL_SIZE // 2 - 2

    # Main apple body
    pygame.draw.circle(screen, APPLE_RED, (cx, cy + 1), r)
    # Darker bottom edge
    pygame.draw.arc(screen, APPLE_DARK,
                    (cx - r, cy + 1 - r, r * 2, r * 2),
                    math.pi + 0.3, 2 * math.pi - 0.3, 2)
    # Shine highlight
    pygame.draw.circle(screen, APPLE_SHINE, (cx - 4, cy - 4), 4)
    pygame.draw.circle(screen, (255, 180, 180), (cx - 2, cy - 6), 2)

    # Stem
    stem_base = (cx, cy - r + 2)
    stem_top = (cx + 1, cy - r - 4)
    pygame.draw.line(screen, APPLE_STEM, stem_base, stem_top, 2)

    # Leaf
    leaf_pts = [
        (cx + 1, cy - r - 2),
        (cx + 7, cy - r - 6),
        (cx + 4, cy - r - 1),
    ]
    pygame.draw.polygon(screen, APPLE_LEAF, leaf_pts)


def draw_death_face(screen, hx, hy, dx, dy):
    """Draw X eyes and dizzy stars on the dead snake head."""
    cx = hx * CELL_SIZE + CELL_SIZE // 2
    cy = hy * CELL_SIZE + CELL_SIZE // 2

    # Head (slightly reddened)
    head_rect = pygame.Rect(hx * CELL_SIZE + 1, hy * CELL_SIZE + 1,
                            CELL_SIZE - 2, CELL_SIZE - 2)
    pygame.draw.rect(screen, (100, 100, 0), head_rect, border_radius=8)

    # X eyes
    perp_x, perp_y = -dy, dx
    eye_offset = 6
    eye_fwd = 4
    x_size = 4

    for side in (-1, 1):
        ex = int(cx + dx * eye_fwd + perp_x * eye_offset * side)
        ey = int(cy + dy * eye_fwd + perp_y * eye_offset * side)
        # Draw X
        pygame.draw.line(screen, EYE_BLACK,
                         (ex - x_size, ey - x_size), (ex + x_size, ey + x_size), 2)
        pygame.draw.line(screen, EYE_BLACK,
                         (ex - x_size, ey + x_size), (ex + x_size, ey - x_size), 2)

    # Dizzy stars around head
    star_positions = [
        (cx - 14, cy - 14),
        (cx + 12, cy - 12),
        (cx - 10, cy + 13),
    ]
    for sx, sy in star_positions:
        star_size = 3
        pygame.draw.line(screen, YELLOW, (sx - star_size, sy), (sx + star_size, sy), 1)
        pygame.draw.line(screen, YELLOW, (sx, sy - star_size), (sx, sy + star_size), 1)
        pygame.draw.line(screen, YELLOW,
                         (sx - 2, sy - 2), (sx + 2, sy + 2), 1)
        pygame.draw.line(screen, YELLOW,
                         (sx - 2, sy + 2), (sx + 2, sy - 2), 1)

    # Open mouth (shocked)
    mouth_x = int(cx + dx * 2)
    mouth_y = int(cy + dy * 2 + 3)
    pygame.draw.circle(screen, (50, 50, 0), (mouth_x, mouth_y), 3)


def draw_alive_head(screen, hx, hy, direction, show_tongue=False, eating=False):
    """Draw the snake head. Tongue only when near food, open mouth when eating."""
    cx = hx * CELL_SIZE + CELL_SIZE // 2
    cy = hy * CELL_SIZE + CELL_SIZE // 2
    dx, dy = DIR_VECTORS[direction]

    # Head
    head_rect = pygame.Rect(hx * CELL_SIZE + 1, hy * CELL_SIZE + 1,
                            CELL_SIZE - 2, CELL_SIZE - 2)
    pygame.draw.rect(screen, DARK_GREEN, head_rect, border_radius=8)

    # Perpendicular for eye placement
    perp_x, perp_y = -dy, dx
    eye_offset = 6
    eye_fwd = 4
    eye_r = 4
    pupil_r = 2

    if eating:
        # Happy squint eyes when eating
        for side in (-1, 1):
            ex = int(cx + dx * eye_fwd + perp_x * eye_offset * side)
            ey = int(cy + dy * eye_fwd + perp_y * eye_offset * side)
            # Upward arc (happy eyes)
            pygame.draw.arc(screen, EYE_BLACK,
                            (ex - 4, ey - 3, 8, 6),
                            0.3, math.pi - 0.3, 2)
        # Open mouth eating the apple
        mouth_x = int(cx + dx * 8)
        mouth_y = int(cy + dy * 8)
        pygame.draw.circle(screen, (40, 100, 40), (mouth_x, mouth_y), 5)
        # Small red crumb to suggest apple being eaten
        pygame.draw.circle(screen, APPLE_RED, (mouth_x + int(dx * 2), mouth_y + int(dy * 2)), 2)
    else:
        # Normal eyes
        for side in (-1, 1):
            ex = cx + dx * eye_fwd + perp_x * eye_offset * side
            ey = cy + dy * eye_fwd + perp_y * eye_offset * side
            pygame.draw.circle(screen, EYE_WHITE, (int(ex), int(ey)), eye_r)
            pygame.draw.circle(screen, EYE_BLACK, (int(ex + dx * 1), int(ey + dy * 1)), pupil_r)

    # Tongue — only when near food (and not eating)
    if show_tongue and not eating:
        tongue_base_x = cx + dx * (CELL_SIZE // 2)
        tongue_base_y = cy + dy * (CELL_SIZE // 2)
        tongue_tip_x = tongue_base_x + dx * 8
        tongue_tip_y = tongue_base_y + dy * 8
        pygame.draw.line(screen, TONGUE,
                         (int(tongue_base_x), int(tongue_base_y)),
                         (int(tongue_tip_x), int(tongue_tip_y)), 2)
        fork = 3
        pygame.draw.line(screen, TONGUE,
                         (int(tongue_tip_x), int(tongue_tip_y)),
                         (int(tongue_tip_x + dx * 3 + perp_x * fork),
                          int(tongue_tip_y + dy * 3 + perp_y * fork)), 2)
        pygame.draw.line(screen, TONGUE,
                         (int(tongue_tip_x), int(tongue_tip_y)),
                         (int(tongue_tip_x + dx * 3 - perp_x * fork),
                          int(tongue_tip_y + dy * 3 - perp_y * fork)), 2)


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
    death_font = pygame.font.SysFont("monospace", 18, bold=True)

    high_score = 0
    paused = False
    state = env.reset()
    done = False
    death_timer = 0
    death_direction = (1, 0)
    prev_score = 0
    eat_timer = 0

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
                    death_timer = 0

        if paused:
            clock.tick(FPS)
            continue

        if done:
            death_timer += 1
            if death_timer > FPS * 2:  # 2 seconds of death animation
                state = env.reset()
                done = False
                death_timer = 0
                clock.tick(FPS)
                continue

            # Draw the death frame
            screen.fill(BLACK)

            # Grid
            for i in range(GRID_SIZE + 1):
                pygame.draw.line(screen, GRAY, (i * CELL_SIZE, 0), (i * CELL_SIZE, WINDOW_SIZE))
                pygame.draw.line(screen, GRAY, (0, i * CELL_SIZE), (WINDOW_SIZE, i * CELL_SIZE))

            # Food
            fx, fy = env.food
            draw_apple(screen, fx, fy)

            # Body (faded)
            for i, (sx, sy) in enumerate(env.snake):
                if i == 0:
                    continue
                body_rect = pygame.Rect(sx * CELL_SIZE + 1, sy * CELL_SIZE + 1,
                                        CELL_SIZE - 2, CELL_SIZE - 2)
                pygame.draw.rect(screen, (0, 140, 0), body_rect, border_radius=6)
                inner = pygame.Rect(sx * CELL_SIZE + 3, sy * CELL_SIZE + 3,
                                    CELL_SIZE - 6, CELL_SIZE - 6)
                pygame.draw.rect(screen, (70, 170, 70), inner, border_radius=4)

            # Dead head — bounced back
            hx, hy = env.snake[0]
            ddx, ddy = death_direction
            bounce = max(0, 4 - death_timer)  # bounce back a few pixels
            offset_x = -ddx * bounce
            offset_y = -ddy * bounce
            # Shift head by offset for bounce effect
            orig_hx_px = hx * CELL_SIZE
            orig_hy_px = hy * CELL_SIZE
            screen_hx = orig_hx_px + offset_x
            screen_hy = orig_hy_px + offset_y
            head_rect = pygame.Rect(screen_hx + 1, screen_hy + 1,
                                    CELL_SIZE - 2, CELL_SIZE - 2)
            pygame.draw.rect(screen, (100, 100, 0), head_rect, border_radius=8)

            # Death face at offset position
            # Use pixel-level drawing for the offset head
            face_cx = screen_hx + CELL_SIZE // 2
            face_cy = screen_hy + CELL_SIZE // 2
            perp_x, perp_y = -ddy, ddx
            eye_offset = 6
            eye_fwd = 4
            x_size = 4
            for side in (-1, 1):
                ex = int(face_cx + ddx * eye_fwd + perp_x * eye_offset * side)
                ey = int(face_cy + ddy * eye_fwd + perp_y * eye_offset * side)
                pygame.draw.line(screen, EYE_BLACK,
                                 (ex - x_size, ey - x_size), (ex + x_size, ey + x_size), 2)
                pygame.draw.line(screen, EYE_BLACK,
                                 (ex - x_size, ey + x_size), (ex + x_size, ey - x_size), 2)

            # Stars
            if death_timer % 3 != 0:  # flicker
                star_positions = [
                    (int(face_cx - 14), int(face_cy - 14)),
                    (int(face_cx + 12), int(face_cy - 12)),
                    (int(face_cx - 10), int(face_cy + 13)),
                ]
                for stx, sty in star_positions:
                    s = 3
                    pygame.draw.line(screen, YELLOW, (stx - s, sty), (stx + s, sty), 1)
                    pygame.draw.line(screen, YELLOW, (stx, sty - s), (stx, sty + s), 1)
                    pygame.draw.line(screen, YELLOW, (stx - 2, sty - 2), (stx + 2, sty + 2), 1)
                    pygame.draw.line(screen, YELLOW, (stx - 2, sty + 2), (stx + 2, sty - 2), 1)

            # Shocked mouth
            mouth_x = int(face_cx + ddx * 2)
            mouth_y = int(face_cy + ddy * 2 + 3)
            pygame.draw.circle(screen, (50, 50, 0), (mouth_x, mouth_y), 3)

            # Score + death text
            score_text = font.render(f"Score: {score}  Best: {high_score}", True, WHITE)
            screen.blit(score_text, (10, 5))
            if death_timer % 6 < 3:  # blink
                death_text = death_font.render("BONK!", True, YELLOW)
                screen.blit(death_text, (hx * CELL_SIZE - 10, hy * CELL_SIZE - 22))

            pygame.display.flip()
            clock.tick(FPS)
            continue

        action = agent.get_action(state)
        prev_direction = env.direction
        state, _, done, info = env.step(action)
        score = info["score"]
        high_score = max(high_score, score)

        # Detect eating
        if score > prev_score:
            eat_timer = 3  # show eating face for 3 frames
        prev_score = score
        eating = eat_timer > 0
        if eat_timer > 0:
            eat_timer -= 1

        if done:
            death_direction = DIR_VECTORS[prev_direction]
            death_timer = 0

        # Check if tongue should show (food within 3 Manhattan distance)
        hx, hy = env.snake[0]
        fx, fy = env.food
        food_dist = abs(hx - fx) + abs(hy - fy)
        show_tongue = food_dist <= 3

        # Draw
        screen.fill(BLACK)

        # Grid lines
        for i in range(GRID_SIZE + 1):
            pygame.draw.line(screen, GRAY, (i * CELL_SIZE, 0), (i * CELL_SIZE, WINDOW_SIZE))
            pygame.draw.line(screen, GRAY, (0, i * CELL_SIZE), (WINDOW_SIZE, i * CELL_SIZE))

        # Food (don't draw during eating animation — it looks like it was eaten)
        if not eating:
            draw_apple(screen, fx, fy)

        # Snake body
        for i, (sx, sy) in enumerate(env.snake):
            if i == 0:
                continue
            body_rect = pygame.Rect(sx * CELL_SIZE + 1, sy * CELL_SIZE + 1,
                                    CELL_SIZE - 2, CELL_SIZE - 2)
            pygame.draw.rect(screen, GREEN, body_rect, border_radius=6)
            inner = pygame.Rect(sx * CELL_SIZE + 3, sy * CELL_SIZE + 3,
                                CELL_SIZE - 6, CELL_SIZE - 6)
            pygame.draw.rect(screen, LIGHT_GREEN, inner, border_radius=4)

        # Snake head
        draw_alive_head(screen, hx, hy, env.direction,
                        show_tongue=show_tongue, eating=eating)

        # Score
        score_text = font.render(f"Score: {score}  Best: {high_score}", True, WHITE)
        screen.blit(score_text, (10, 5))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
