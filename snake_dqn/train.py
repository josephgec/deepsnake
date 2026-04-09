import csv
import os
from collections import deque

from snake_env import SnakeEnv
from agent import DQNAgent


def train(num_episodes=1000):
    env = SnakeEnv()
    agent = DQNAgent()

    os.makedirs("checkpoints", exist_ok=True)

    scores = deque(maxlen=100)
    best_score = 0
    best_avg = 0.0
    log_rows = []

    step_count = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_loss = 0.0
        loss_count = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            step_count += 1
            if step_count % 4 == 0:
                loss = agent.train_step(batch_size=64)
                if loss > 0:
                    total_loss += loss
                    loss_count += 1
            # Soft target update every step (cheap)
            agent.update_target_network()
            state = next_state

        score = info["score"]
        scores.append(score)
        agent.decay_epsilon(total_episodes=num_episodes)

        avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
        avg_score = sum(scores) / len(scores)

        log_rows.append({
            "episode": episode,
            "score": score,
            "epsilon": agent.epsilon,
            "avg_loss": avg_loss,
        })

        if score > best_score:
            best_score = score
            agent.save("checkpoints/best_model.pth")

        if avg_score > best_avg and len(scores) == 100:
            best_avg = avg_score
            agent.save("checkpoints/best_avg_model.pth")

        if episode % 50 == 0:
            print(f"Ep {episode:5d} | Score: {score:3d} | "
                  f"Avg(100): {avg_score:6.2f} | "
                  f"Best: {best_score:3d} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}", flush=True)

    # Save final model and log
    agent.save("checkpoints/final_model.pth")

    with open("training_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "score", "epsilon", "avg_loss"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nTraining complete. Final avg(100): {sum(scores)/len(scores):.2f}, "
          f"Best: {best_score}")


if __name__ == "__main__":
    train()
