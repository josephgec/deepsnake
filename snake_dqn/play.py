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
    """Load the face photo, crop tight to face, circle-mask, create rotated versions."""
    raw = pygame.image.load(path)

    # Zoom in on the face — crop to center 55% of image height, centered
    w, h = raw.get_size()
    face_h = int(h * 0.55)
    face_w = face_h  # square
    crop_x = (w - face_w) // 2
    crop_y = int(h * 0.15)  # face starts ~15% from top
    square = pygame.Surface((face_w, face_h), pygame.SRCALPHA)
    square.blit(raw, (0, 0), (crop_x, crop_y, face_w, face_h))

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


SKIN_COLOR = (210, 170, 130)
SKIN_LIGHT = (230, 195, 160)
SKIN_DARK = (180, 140, 100)
SHOE_COLOR = (200, 60, 60)
SHOE_DARK = (150, 40, 40)
LEG_COLOR = (180, 145, 110)
HAND_COLOR = (220, 180, 140)


def draw_body_segment(screen, sx, sy, is_tail=False, is_neck=False,
                      frame=0, seg_index=0, direction=None, snake_length=3):
    """Draw a body segment. Neck gets arms, tail gets legs that grow."""
    cx = sx * CELL_SIZE + CELL_SIZE // 2
    cy = sy * CELL_SIZE + CELL_SIZE // 2

    # Long smooth body
    body_rect = pygame.Rect(sx * CELL_SIZE + 2, sy * CELL_SIZE + 2,
                            CELL_SIZE - 4, CELL_SIZE - 4)
    pygame.draw.rect(screen, SKIN_COLOR, body_rect, border_radius=8)
    inner = pygame.Rect(sx * CELL_SIZE + 4, sy * CELL_SIZE + 4,
                        CELL_SIZE - 8, CELL_SIZE - 8)
    pygame.draw.rect(screen, SKIN_LIGHT, inner, border_radius=6)
    pygame.draw.rect(screen, SKIN_DARK, body_rect, 1, border_radius=8)

    if direction is None:
        return

    dx, dy = DIR_VECTORS[direction]
    perp_x, perp_y = -dy, dx

    # Arms on the neck segment (first body segment behind head)
    if is_neck:
        phase = frame % 2
        arm_len = 10

        for side in (-1, 1):
            # Arm base at the side of the body
            base_x = cx + perp_x * (CELL_SIZE // 2 - 2) * side
            base_y = cy + perp_y * (CELL_SIZE // 2 - 2) * side

            # Arms swing forward/back while running
            swing = arm_len if (phase == 0) == (side == 1) else -arm_len
            elbow_x = base_x + perp_x * 6 * side + dx * (swing * 0.5)
            elbow_y = base_y + perp_y * 6 * side + dy * (swing * 0.5)
            hand_x = elbow_x + dx * swing * 0.5 + perp_x * 2 * side
            hand_y = elbow_y + dy * swing * 0.5 + perp_y * 2 * side

            # Upper arm
            pygame.draw.line(screen, LEG_COLOR,
                             (int(base_x), int(base_y)),
                             (int(elbow_x), int(elbow_y)), 2)
            # Forearm
            pygame.draw.line(screen, LEG_COLOR,
                             (int(elbow_x), int(elbow_y)),
                             (int(hand_x), int(hand_y)), 2)
            # Hand
            pygame.draw.circle(screen, HAND_COLOR, (int(hand_x), int(hand_y)), 3)

    # Legs on the tail — length grows with snake size
    if is_tail:
        phase = frame % 2
        # Legs grow: start at 6, add 0.5 per body segment, cap at 20
        leg_len = min(6 + snake_length * 0.5, 20)
        leg_thickness = min(2 + snake_length // 10, 4)
        foot_r = min(3 + snake_length // 15, 5)

        for side in (-1, 1):
            base_x = cx + perp_x * (CELL_SIZE // 2 - 3) * side
            base_y = cy + perp_y * (CELL_SIZE // 2 - 3) * side

            # Knee bend — legs have two segments
            swing = leg_len if (phase == 0) == (side == 1) else -leg_len
            knee_x = base_x + perp_x * (leg_len * 0.4) * side + dx * (swing * 0.3)
            knee_y = base_y + perp_y * (leg_len * 0.4) * side + dy * (swing * 0.3)
            tip_x = knee_x + perp_x * (leg_len * 0.3) * side + dx * (swing * 0.7)
            tip_y = knee_y + perp_y * (leg_len * 0.3) * side + dy * (swing * 0.7)

            # Thigh
            pygame.draw.line(screen, LEG_COLOR,
                             (int(base_x), int(base_y)),
                             (int(knee_x), int(knee_y)), leg_thickness)
            # Shin
            pygame.draw.line(screen, LEG_COLOR,
                             (int(knee_x), int(knee_y)),
                             (int(tip_x), int(tip_y)), leg_thickness)
            # Shoe
            pygame.draw.circle(screen, SHOE_COLOR, (int(tip_x), int(tip_y)), foot_r)
            pygame.draw.circle(screen, SHOE_DARK, (int(tip_x), int(tip_y)), foot_r, 1)


def draw_motion_lines(screen, tx, ty, direction, frame):
    """Draw speed lines behind the tail."""
    dx, dy = DIR_VECTORS[direction]
    cx = tx * CELL_SIZE + CELL_SIZE // 2
    cy = ty * CELL_SIZE + CELL_SIZE // 2

    # Lines trail behind (opposite of direction)
    for i in range(3):
        offset = 8 + i * 6 + (frame % 2) * 3
        length = 6 - i * 2
        lx = cx - dx * offset
        ly = cy - dy * offset
        # Perpendicular spread
        perp_x, perp_y = -dy, dx
        spread = (i - 1) * 5
        sx_pos = int(lx + perp_x * spread)
        sy_pos = int(ly + perp_y * spread)
        ex = int(sx_pos - dx * length)
        ey = int(sy_pos - dy * length)
        alpha = 180 - i * 50
        pygame.draw.line(screen, (alpha, alpha, alpha),
                         (sx_pos, sy_pos), (ex, ey), 1)


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
    frame_count = 0

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

            # Body (frozen pose)
            snake_list = list(env.snake)
            slen = len(snake_list)
            for i, (sx, sy) in enumerate(snake_list):
                if i == 0:
                    continue
                draw_body_segment(screen, sx, sy,
                                  is_tail=(i == slen - 1),
                                  is_neck=(i == 1),
                                  frame=0, seg_index=i,
                                  direction=last_direction, snake_length=slen)

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

        # Body segments with arms and legs
        snake_list = list(env.snake)
        slen = len(snake_list)
        for i, (sx, sy) in enumerate(snake_list):
            if i == 0:
                continue
            draw_body_segment(screen, sx, sy,
                              is_tail=(i == slen - 1),
                              is_neck=(i == 1),
                              frame=frame_count, seg_index=i,
                              direction=env.direction, snake_length=slen)

        # Motion lines behind the tail
        if len(snake_list) > 1:
            tail_x, tail_y = snake_list[-1]
            draw_motion_lines(screen, tail_x, tail_y, env.direction, frame_count)

        # Head — the face photo
        hx, hy = env.snake[0]
        head_sprite = face_sprites[env.direction]
        screen.blit(head_sprite, (hx * CELL_SIZE, hy * CELL_SIZE))

        frame_count += 1

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
