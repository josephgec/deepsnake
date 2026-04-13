# DeepSnake - Snake DQN Reinforcement Learning

A Snake game agent trained with Deep Q-Learning (DQN) using PyTorch. The agent learns from scratch to play Snake, averaging ~53 points over 200 evaluation games. Hyperparameters were tuned via automated experiment sweeps (autoresearch), then further improved with a hybrid scalar-MLP / local-CNN architecture and extended training (5000 episodes).

## Results

- **Average score (200 games):** ~53 (varies 50–56 across seeds)
- **Best single game:** 87
- **Games scoring 30+:** ~95%

*Improved from baseline avg 33.18 through autoresearch + a hybrid scalar-MLP / local-CNN architecture with 9×9 local view.*

![Training Plot](snake_dqn/training_plot.png)

## Tech Stack

- Python 3.10+
- PyTorch (neural network)
- Pygame (visualization)
- NumPy
- Matplotlib (training plots)

## Project Structure

```
snake_dqn/
├── snake_env.py      # Headless Snake game environment (READ-ONLY)
├── model.py          # Hybrid DQN: scalar MLP + local-grid CNN
├── agent.py          # Double DQN agent with replay buffer
├── train.py          # Training loop (5000 episodes)
├── evaluate.py       # Headless evaluation over 200 games
├── play.py           # Pygame visualization of trained agent
├── plot_training.py  # Plot training curves
├── training_log.csv  # Per-episode training log
├── training_plot.png # Training curves plot
└── checkpoints/      # Saved model weights
results.tsv           # Full autoresearch experiment log (48 experiments)
```

## Quick Start

### Install dependencies

```bash
pip install torch numpy pygame matplotlib
```

### Watch the trained agent play

```bash
cd snake_dqn
python play.py
```

**Controls:**
- `SPACE` - pause/unpause
- `R` - reset game
- `Q` - quit

The visualization features custom sprite-based graphics: the snake has a dark-green rounded head with directional eyes and a forked red tongue that flicks out when food is within 3 cells, and a green rounded-segment body. The apple has a 3D-shaded look with stem, leaf, and highlight. Eating triggers a happy-squint-eyes + open-mouth animation (with a red crumb), and dying plays a bounce-back death effect with X-eyes, dizzy stars, and a flashing "BONK!" label.

### Train from scratch

```bash
cd snake_dqn
python train.py
```

Training runs for 5000 episodes and saves checkpoints to `checkpoints/`.

### Generate training plots

```bash
cd snake_dqn
python plot_training.py
```

## How It Works

### Environment

- 20x20 grid, snake starts at center (length 3, moving right)
- Actions are relative: straight, turn right, turn left
- Reward: +10 eat food, -10 die, +1/-1 move closer/farther from food

### State Representation (24 scalar features + 9×9 local grid)

**Scalar features (24 values):**
- Immediate danger in 3 directions (straight, right, left)
- Current direction (one-hot, 4 values)
- Food direction (4 binary values)
- Normalized head position (x, y)
- Normalized food offset (dx, dy)
- Snake length (normalized)
- Distance to nearest obstacle in 8 directions (raycasting)

**Local grid (3 × 9 × 9 = 243 values):**
- Centered on the snake's head, radius 4
- Channel 0: nearby body segments
- Channel 1: food (if within window)
- Channel 2: walls (out-of-bounds cells)

The grid lets the CNN see the local body topology that scalar features can't encode (e.g., box-loop traps, narrow corridors).

### DQN Improvements

| Technique | Purpose |
|---|---|
| Double DQN | Reduces Q-value overestimation |
| Huber loss (SmoothL1) | Stabilizes training vs MSE |
| Soft target updates (tau=0.005) | Smoother learning than hard copies |
| Gradient clipping (max_norm=1.0) | Prevents gradient explosion |
| Linear epsilon decay (over 70% of episodes) | Better exploration schedule than multiplicative decay |
| Larger batch size (64) | More stable gradient estimates |
| Train every 4 steps | Reduces compute; decorrelates updates |

### Network Architecture (Hybrid)

The model has two parallel input paths that merge before the output layer:

```
                  ┌─────────────────────┐
                  │   24 scalar features │
                  └──────────┬──────────┘
                             │
                      Linear(24 → 128) → ReLU
                             │
                      128-dim ─────────────────┐
                                                │
  ┌─────────────────────┐                      │
  │  3 × 9 × 9 grid     │                      │
  │  (local view)       │                      │
  └──────────┬──────────┘                      │
             │                                  │
     Conv2d(3→16, 3×3) → ReLU                 │
     Conv2d(16→16, 3×3) → ReLU                │
     Flatten (400)                             │
             │                                  │
      400-dim ─────────────────────────────────┤
                                                │
                                    Concat (528)
                                        │
                                Linear(528 → 128) → ReLU
                                Linear(128 → 3)
                                        │
                                Q-values (3 actions)
```

**~74K parameters total.** The scalar path bootstraps food-finding (proven from baseline); the CNN path adds local body-topology awareness for trap avoidance.

### Autoresearch Results

48 experiments were run automatically using the [Karpathy autoresearch](https://github.com/karpathy/autoresearch) pattern: modify code, train, evaluate over 200 greedy games, keep improvements, discard failures, repeat. Each experiment was committed, evaluated, and either kept or reverted.

**Improvements kept (cumulative):**

| # | Experiment | avg_score | Delta |
|---|---|---|---|
| 0 | Baseline (original code) | 33.18 | -- |
| 1 | Linear epsilon decay (80%) | 39.10 | +5.92 |
| 2 | Batch size 64 → 128 | 40.99 | +1.89 |
| 3 | Wider network (512-256-128) | 42.66 | +1.67 |
| 4 | Epsilon decay tuned to 70% | 43.66 | +1.00 |
| 5 | 1000 episodes + correct schedule | 44.36 | +0.70 |

**Total improvement: +11.18 avg_score (+34%)**

After the autoresearch sweep, two manual changes pushed the score further: replacing the flat MLP with the hybrid scalar-MLP + local-CNN architecture (v5), and expanding the local grid from 7×7 to 9×9 with 5000 training episodes (v6), reaching the current avg ~53.

Notable failed experiments include: dueling DQN, prioritized experience replay, n-step returns, reward clipping/scaling, cosine LR annealing, LayerNorm, dropout, GELU/SiLU activations, and various learning rate reductions. See `results.tsv` for the full log of all 48 experiments.

### Evaluate a trained model

```bash
cd snake_dqn
python evaluate.py
```

Runs 200 headless games and prints avg_score, median, best, and percentage scoring 30+.
