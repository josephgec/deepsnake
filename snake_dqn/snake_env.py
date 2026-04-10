import numpy as np
from enum import IntEnum
from collections import deque


class Direction(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


# Relative actions
STRAIGHT = 0
TURN_RIGHT = 1
TURN_LEFT = 2

# Direction vectors: RIGHT, DOWN, LEFT, UP
DIR_VECTORS = {
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.UP: (0, -1),
}


class SnakeEnv:
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.reset()

    def reset(self) -> np.ndarray:
        mid = self.grid_size // 2
        # Snake starts at center, length 3, moving right
        # Head is first element
        self.snake = deque([
            (mid, mid),
            (mid - 1, mid),
            (mid - 2, mid),
        ])
        self.direction = Direction.RIGHT
        self.score = 0
        self.steps = 0
        self._place_food()
        return self.get_state()

    def _place_food(self):
        snake_set = set(self.snake)
        while True:
            pos = (
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size),
            )
            if pos not in snake_set:
                self.food = pos
                return

    def _is_collision(self, x, y) -> bool:
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        # Check body collision (exclude head at index 0)
        for i, seg in enumerate(self.snake):
            if i == 0:
                continue
            if seg == (x, y):
                return True
        return False

    def _get_local_grid(self, head_x, head_y, radius=4):
        """Return a 3-channel (2*radius+1)x(2*radius+1) grid centered on the head.

        Channel 0: body segments (excluding head)
        Channel 1: food
        Channel 2: walls (out-of-bounds cells)
        """
        size = 2 * radius + 1
        grid = np.zeros((3, size, size), dtype=np.float32)
        snake_set = set(self.snake)

        for ddy in range(-radius, radius + 1):
            for ddx in range(-radius, radius + 1):
                gx = head_x + ddx
                gy = head_y + ddy
                lx = ddx + radius
                ly = ddy + radius

                if gx < 0 or gx >= self.grid_size or gy < 0 or gy >= self.grid_size:
                    grid[2][ly][lx] = 1.0
                    continue

                if (gx, gy) in snake_set and (gx, gy) != (head_x, head_y):
                    grid[0][ly][lx] = 1.0

                if (gx, gy) == self.food:
                    grid[1][ly][lx] = 1.0

        return grid

    def _ray_distance(self, start_x, start_y, dx, dy) -> float:
        """Cast a ray and return normalized distance to nearest obstacle."""
        dist = 0
        x, y = start_x, start_y
        while True:
            x += dx
            y += dy
            dist += 1
            if self._is_collision(x, y):
                return dist / self.grid_size

    def step(self, action: int) -> tuple:
        self.steps += 1

        # Update direction based on relative action
        if action == TURN_RIGHT:
            self.direction = Direction((self.direction + 1) % 4)
        elif action == TURN_LEFT:
            self.direction = Direction((self.direction - 1) % 4)
        # STRAIGHT: no direction change

        # Move head
        dx, dy = DIR_VECTORS[self.direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        # Check death
        max_steps = 100 * len(self.snake)
        if self._is_collision(*new_head) or self.steps > max_steps:
            reward = -10
            done = True
            info = {"score": self.score, "length": len(self.snake)}
            return self.get_state(), reward, done, info

        # Distance-based reward
        old_dist = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        self.snake.appendleft(new_head)

        # Check food
        if new_head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            reward = 1 if new_dist < old_dist else -1

        done = False
        info = {"score": self.score, "length": len(self.snake)}
        return self.get_state(), reward, done, info

    def get_state(self) -> np.ndarray:
        head_x, head_y = self.snake[0]
        dx, dy = DIR_VECTORS[self.direction]

        # Compute the directions for right and left relative turns
        dir_right = Direction((self.direction + 1) % 4)
        dir_left = Direction((self.direction - 1) % 4)
        dx_r, dy_r = DIR_VECTORS[dir_right]
        dx_l, dy_l = DIR_VECTORS[dir_left]

        # Danger checks
        danger_straight = self._is_collision(head_x + dx, head_y + dy)
        danger_right = self._is_collision(head_x + dx_r, head_y + dy_r)
        danger_left = self._is_collision(head_x + dx_l, head_y + dy_l)

        # Direction one-hot
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        # Food direction (binary)
        food_left = self.food[0] < head_x
        food_right = self.food[0] > head_x
        food_up = self.food[1] < head_y
        food_down = self.food[1] > head_y

        # Additional features for state disambiguation
        gs = self.grid_size
        head_x_norm = head_x / gs
        head_y_norm = head_y / gs
        food_dx = (self.food[0] - head_x) / gs
        food_dy = (self.food[1] - head_y) / gs
        length_norm = len(self.snake) / (gs * gs)

        # Raycasting: distance to nearest obstacle in 8 directions
        # Relative to current direction: forward, forward-right, right, back-right,
        # back, back-left, left, forward-left
        ray_dirs = [
            (dx, dy),                   # forward
            (dx + dx_r, dy + dy_r),     # forward-right (diagonal)
            (dx_r, dy_r),               # right
            (-dx + dx_r, -dy + dy_r),   # back-right
            (-dx, -dy),                 # back
            (-dx + dx_l, -dy + dy_l),   # back-left
            (dx_l, dy_l),               # left
            (dx + dx_l, dy + dy_l),     # forward-left
        ]
        rays = [self._ray_distance(head_x, head_y, rdx, rdy)
                for rdx, rdy in ray_dirs]

        scalar_state = np.array([
            danger_straight,
            danger_right,
            danger_left,
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            food_left,
            food_right,
            food_up,
            food_down,
            head_x_norm,
            head_y_norm,
            food_dx,
            food_dy,
            length_norm,
            *rays,
        ], dtype=np.float32)

        # Append flattened 7x7 local grid (3 channels = 147 values)
        local_grid = self._get_local_grid(head_x, head_y).flatten()
        return np.concatenate([scalar_state, local_grid])


if __name__ == "__main__":
    env = SnakeEnv()
    scores = []
    for _ in range(100):
        env.reset()
        done = False
        while not done:
            action = np.random.randint(0, 3)
            _, _, done, info = env.step(action)
        scores.append(info["score"])
    print(f"Random agent over 100 games: avg={np.mean(scores):.2f}, "
          f"min={np.min(scores)}, max={np.max(scores)}")
