"""Headless evaluation of trained DQN agent over 200 games."""
import numpy as np
import torch
from snake_env import SnakeEnv
from model import DQN

def evaluate(model_path="checkpoints/best_avg_model.pth", num_games=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    env = SnakeEnv()
    scores = []
    for _ in range(num_games):
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = model(state_t).argmax(dim=1).item()
            state, _, done, info = env.step(action)
        scores.append(info["score"])

    scores = np.array(scores)
    over_30 = int(np.sum(scores >= 30))
    print("---")
    print(f"avg_score:        {scores.mean():.2f}")
    print(f"median_score:     {np.median(scores):.2f}")
    print(f"best_score:       {scores.max()}")
    print(f"games_over_30:    {over_30}")
    print(f"pct_over_30:      {over_30 / num_games * 100:.1f}")
    print(f"total_games:      {num_games}")
    print("---")

if __name__ == "__main__":
    evaluate()
