import csv

import matplotlib.pyplot as plt
import numpy as np


def plot_training(log_path="training_log.csv", output_path="training_plot.png"):
    episodes, scores, epsilons = [], [], []

    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            scores.append(int(row["score"]))
            epsilons.append(float(row["epsilon"]))

    episodes = np.array(episodes)
    scores = np.array(scores)
    epsilons = np.array(epsilons)

    # Rolling average
    window = 100
    rolling_avg = np.convolve(scores, np.ones(window) / window, mode="valid")
    rolling_x = episodes[window - 1:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Score plot
    ax1.scatter(episodes, scores, alpha=0.15, s=5, color="steelblue", label="Score")
    ax1.plot(rolling_x, rolling_avg, color="orange", linewidth=2,
             label=f"Rolling avg ({window})")
    ax1.axhline(y=30, color="red", linestyle="--", alpha=0.5, label="Target (30)")
    ax1.set_ylabel("Score")
    ax1.set_title("Snake DQN Training")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Epsilon plot
    ax2.plot(episodes, epsilons, color="green", linewidth=1.5)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Epsilon")
    ax2.set_title("Epsilon Decay")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_training()
