import sys
import math
import os

import numpy as np
import pygame

from snake_env import SnakeEnv, Direction, DIR_VECTORS
from agent import DQNAgent


# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (60, 60, 60)
YELLOW = (255, 220, 50)
BODY_COLOR = (70, 130, 220)
BODY_LIGHT = (110, 165, 240)
BODY_DARK = (50, 100, 180)

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

# Direction to rotation angle (image faces UP by default after crop)
# RIGHT=0, DOWN=1, LEFT=2, UP=3
DIR_TO_ANGLE = {
    Direction.RIGHT: 270,
    Direction.DOWN: 180,
    Direction.LEFT: 90,
    Direction.UP: 0,
}


def load_face_sprites(path, size):
    """Load the face photo, crop to circle, create rotated versions per direction."""
    raw = pygame.image.load(path)

    # Crop to square from center
    w, h = raw.get_size()
    side = min(w, h)
    crop_x = (w - side) // 2
    crop_y = (h - side) // 2
    square = pygame.Surface((side, side), pygame.SRCALPHA)
    square.blit(raw, (0, 0), (crop_x, crop_y, side, side))

    # Scale to cell size
    scaled = pygame.transform.smoothscale(square, (size, size))

    # Circular mask
    circle_surf = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(circle_surf, (255, 255, 255, 255), (size // 2, size // 2), size // 2)

    masked = pygame.Surface((size, size), pygame.SRCALPHA)
    for x in range(size):
        for y in range(size):
            if circle_surf.get_at((x, y)).a > 0:
                masked.set_at((x, y), scaled.get_at((x, y)))

    # Create rotated sprites for each direction
    sprites = {}
    for direction, angle in DIR_TO_ANGLE.items():
        rotated = pygame.transform.rotate(masked, angle)
        # Rotation may change size slightly — re-center
        rot_rect = rotated.get_rect(center=(size // 2, size // 2))
        final = pygame.Surface((size, size), pygame.SRCALPHA)
        final.blit(rotated, rot_rect)
        sprites[direction] = final

    # Death sprite — tinted red, no rotation needed (use last direction)
    death_sprite = masked.copy()
    red_overlay = pygame.Surface((size, size), pygame.SRCALPHA)
    red_overlay.fill((255, 0, 0, 80))
    death_sprite.blit(red_overlay, (0, 0))
    sprites["death"] = death_sprite

    return sprites


def draw_apple(screen, fx, fy):
    """Draw a nice apple with stem, leaf, and shine."""
    cx = fx * CELL_SIZE + CELL_SIZE // 2
    cy = fy * CELL_SIZE + CELL_SIZE // 2
    r = CELL_SIZE // 2 - 2

    pygame.draw.circle(screen, APPLE_RED, (cx, cy + 1), r)
    pygame.draw.arc(screen, APPLE_DARK,
                    (cx - r, cy + 1 - r, r * 2, r * 2),
                    math.pi + 0.3, 2 * math.pi - 0.3, 2)
    pygame.draw.circle(screen, APPLE_SHINE, (cx - 4, cy - 4), 4)
    pygame.draw.circle(screen, (255, 180, 180), (cx - 2, cy - 6), 2)

    stem_base = (cx, cy - r + 2)
    stem_top = (cx + 1, cy - r - 4)
    pygame.draw.line(screen, APPLE_STEM, stem_base, stem_top, 2)

    leaf_pts = [
        (cx + 1, cy - r - 2),
        (cx + 7, cy - r - 6),
        (cx + 4, cy - r - 1),
    ]
    pygame.draw.polygon(screen, APPLE_LEAF, leaf_pts)


def draw_body_segment(screen, sx, sy, is_tail=False):
    """Draw a body segment as a rounded blue blob."""
    cx = sx * CELL_SIZE + CELL_SIZE // 2
    cy = sy * CELL_SIZE + CELL_SIZE // 2
    r = CELL_SIZE // 2 - 2 if not is_tail else CELL_SIZE // 2 - 4

    # Outer circle
    pygame.draw.circle(screen, BODY_COLOR, (cx, cy), r)
    # Inner highlight
    pygame.draw.circle(screen, BODY_LIGHT, (cx - 2, cy - 2), r - 4)
    # Dark outline
    pygame.draw.circle(screen, BODY_DARK, (cx, cy), r, 2)


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

    # Load face sprites
    face_path = os.path.join(os.path.dirname(__file__), "face.jpg")
    face_sprites = load_face_sprites(face_path, CELL_SIZE)

    high_score = 0
    paused = False
    state = env.reset()
    done = False
    death_timer = 0
    death_direction = (1, 0)
    last_direction = Direction.RIGHT
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
            if death_timer > FPS * 2:
                state = env.reset()
                done = False
                death_timer = 0
                prev_score = 0
                clock.tick(FPS)
                continue

            # --- Death frame ---
            screen.fill(BLACK)

            # Grid
            for i in range(GRID_SIZE + 1):
                pygame.draw.line(screen, GRAY, (i * CELL_SIZE, 0), (i * CELL_SIZE, WINDOW_SIZE))
                pygame.draw.line(screen, GRAY, (0, i * CELL_SIZE), (WINDOW_SIZE, i * CELL_SIZE))

            # Food
            draw_apple(screen, env.food[0], env.food[1])

            # Body (faded)
            snake_list = list(env.snake)
            for i, (sx, sy) in enumerate(snake_list):
                if i == 0:
                    continue
                draw_body_segment(screen, sx, sy, is_tail=(i == len(snake_list) - 1))

            # Dead head — bounced back with death sprite
            hx, hy = env.snake[0]
            ddx, ddy = death_direction
            bounce = max(0, 4 - death_timer)
            px = hx * CELL_SIZE + int(-ddx * bounce)
            py = hy * CELL_SIZE + int(-ddy * bounce)

            # Draw the death-tinted face
            death_face = face_sprites.get("death", face_sprites[last_direction])
            screen.blit(death_face, (px, py))

            # Dizzy stars
            face_cx = px + CELL_SIZE // 2
            face_cy = py + CELL_SIZE // 2
            if death_timer % 3 != 0:
                for (ox, oy) in [(-14, -16), (12, -14), (-10, 15)]:
                    stx, sty = int(face_cx + ox), int(face_cy + oy)
                    s = 3
                    pygame.draw.line(screen, YELLOW, (stx - s, sty), (stx + s, sty), 1)
                    pygame.draw.line(screen, YELLOW, (stx, sty - s), (stx, sty + s), 1)
                    pygame.draw.line(screen, YELLOW, (stx - 2, sty - 2), (stx + 2, sty + 2), 1)
                    pygame.draw.line(screen, YELLOW, (stx - 2, sty + 2), (stx + 2, sty - 2), 1)

            # Score + BONK
            score_text = font.render(f"Score: {score}  Best: {high_score}", True, WHITE)
            screen.blit(score_text, (10, 5))
            if death_timer % 6 < 3:
                death_text = death_font.render("BONK!", True, YELLOW)
                screen.blit(death_text, (hx * CELL_SIZE - 10, hy * CELL_SIZE - 22))

            pygame.display.flip()
            clock.tick(FPS)
            continue

        # --- Alive frame ---
        action = agent.get_action(state)
        prev_direction = env.direction
        state, _, done, info = env.step(action)
        score = info["score"]
        high_score = max(high_score, score)

        # Eating detection
        if score > prev_score:
            eat_timer = 3
        prev_score = score
        eating = eat_timer > 0
        if eat_timer > 0:
            eat_timer -= 1

        if done:
            death_direction = DIR_VECTORS[prev_direction]
            last_direction = prev_direction
            death_timer = 0

        # Draw
        screen.fill(BLACK)

        # Grid lines
        for i in range(GRID_SIZE + 1):
            pygame.draw.line(screen, GRAY, (i * CELL_SIZE, 0), (i * CELL_SIZE, WINDOW_SIZE))
            pygame.draw.line(screen, GRAY, (0, i * CELL_SIZE), (WINDOW_SIZE, i * CELL_SIZE))

        # Food (hide briefly when eating)
        if not eating:
            draw_apple(screen, env.food[0], env.food[1])

        # Body segments
        snake_list = list(env.snake)
        for i, (sx, sy) in enumerate(snake_list):
            if i == 0:
                continue
            draw_body_segment(screen, sx, sy, is_tail=(i == len(snake_list) - 1))

        # Head — the face photo
        hx, hy = env.snake[0]
        head_sprite = face_sprites[env.direction]
        screen.blit(head_sprite, (hx * CELL_SIZE, hy * CELL_SIZE))

        # Eating effect — small apple crumbs near mouth
        if eating:
            dx, dy = DIR_VECTORS[env.direction]
            crumb_x = hx * CELL_SIZE + CELL_SIZE // 2 + int(dx * 12)
            crumb_y = hy * CELL_SIZE + CELL_SIZE // 2 + int(dy * 12)
            pygame.draw.circle(screen, APPLE_RED, (crumb_x, crumb_y), 3)
            pygame.draw.circle(screen, APPLE_RED, (crumb_x + 4, crumb_y - 3), 2)

        # Score
        score_text = font.render(f"Score: {score}  Best: {high_score}", True, WHITE)
        screen.blit(score_text, (10, 5))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
