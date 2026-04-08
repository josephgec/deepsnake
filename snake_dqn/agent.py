import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN


class DQNAgent:
    def __init__(self, state_size=24, action_size=3, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

        # Replay buffer
        self.memory = deque(maxlen=100_000)
        self.min_replay_size = 1_000

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.epsilon_step = 0  # for linear decay

    def get_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, batch_size=128) -> float:
        if len(self.memory) < self.min_replay_size:
            return 0.0

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Current Q values
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Double DQN: policy net selects actions, target net evaluates them
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            next_q[dones] = 0.0
            target_q = rewards + self.gamma * next_q

        loss = self.criterion(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self, total_episodes=500):
        self.epsilon_step += 1
        decay_episodes = int(total_episodes * 0.7)
        self.epsilon = max(self.epsilon_min, 1.0 - self.epsilon_step / decay_episodes)

    def update_target_network(self, tau=0.005):
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1.0 - tau) * target_param.data
            )

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        self.policy_net.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.update_target_network()
        self.epsilon = 0.0  # No exploration when playing


if __name__ == "__main__":
    agent = DQNAgent()
    # Test storing and training
    dummy_state = np.random.randn(24).astype(np.float32)
    for _ in range(1100):
        action = agent.get_action(dummy_state)
        agent.store_transition(dummy_state, action, 1.0, dummy_state, False)
    loss = agent.train_step()
    print(f"Train step loss: {loss:.4f}")
    print("Agent OK!")
