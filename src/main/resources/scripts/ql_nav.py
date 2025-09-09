#!/usr/bin/env python3
import json
import math
import os
import random
import sys
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

STATE_DIM = 4  # dx, dz, vx, vz
ACTION_DIM = 5  # forward, back, left, right, idle
CHECKPOINT_INTERVAL = 10

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_DIM)
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self):
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.batch_size = 64
        self.memory = deque(maxlen=5000)
        if torch:
            self.model = DQN()
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            self.loss_fn = nn.MSELoss()
        self.last_state = None
        self.last_action = None
        self.last_distance = None

    def save(self, path):
        if not torch:
            return
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "memory": list(self.memory)
        }, path)

    def load(self, path):
        if not torch or not os.path.exists(path):
            return False
        data = torch.load(path)
        self.model.load_state_dict(data.get("model", {}))
        self.optimizer.load_state_dict(data.get("optimizer", {}))
        self.epsilon = data.get("epsilon", self.epsilon)
        self.memory = deque(data.get("memory", []), maxlen=5000)
        return True

    def normalize(self, data):
        dx = (data["gx"] - data["px"]) / 50.0
        dz = (data["gz"] - data["pz"]) / 50.0
        vx = data["vx"] / 5.0
        vz = data["vz"] / 5.0
        state = torch.tensor([dx, dz, vx, vz], dtype=torch.float32) if torch else [dx, dz, vx, vz]
        dist = math.hypot(dx, dz)
        return state, dist

    def act(self, state):
        if torch and random.random() > self.epsilon:
            with torch.no_grad():
                q = self.model(state)
                return int(torch.argmax(q).item())
        return random.randrange(ACTION_DIM)

    def remember(self, s, a, r, ns, done):
        if torch:
            self.memory.append((s, a, r, ns, done))

    def replay(self):
        if not torch or len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([b[0] for b in batch])
        actions = torch.tensor([b[1] for b in batch])
        rewards = torch.tensor([b[2] for b in batch])
        next_states = torch.stack([b[3] for b in batch])
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q = self.model(next_states).max(1)[0]
        target = rewards + self.gamma * next_q * (1 - dones)
        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "ql_nav_checkpoint.pth"
    agent = Agent()
    if torch and agent.load(checkpoint_path):
        print(f"Loaded checkpoint from {checkpoint_path}", file=sys.stderr)
    else:
        print("No checkpoint found, starting fresh.", file=sys.stderr)
    episode_rewards = []
    episode_reward = 0.0
    episode_count = 0
    if plt:
        plt.ion()
        fig, ax = plt.subplots()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        data = json.loads(line)
        state, dist = agent.normalize(data)
        done = data.get("done", False)
        reached = data.get("reached", False)

        if agent.last_distance is None:
            agent.last_distance = dist

        if torch:
            dx, dz, vx, vz = state.tolist()
        else:
            dx, dz, vx, vz = state

        progress = agent.last_distance - dist
        alignment = dx * vx + dz * vz
        reward = progress * 10 + alignment * 5 - 0.05

        if done:
            if reached:
                reward += 100.0
            else:
                reward -= 25.0

        if agent.last_state is not None:
            agent.remember(agent.last_state, agent.last_action, reward, state, done)
            agent.replay()

        episode_reward += reward

        action = agent.act(state)
        sys.stdout.write(str(action) + "\n")
        sys.stdout.flush()

        if done:
            episode_rewards.append(episode_reward)
            if plt:
                ax.clear()
                ax.plot(episode_rewards)
                ax.set_xlabel("Episode")
                ax.set_ylabel("Total Reward")
                ax.set_title("QLearningNavigator Reward Progress")
                plt.pause(0.001)
            episode_reward = 0.0
            episode_count += 1
            agent.last_state = None
            agent.last_action = None
            agent.last_distance = None
            agent.decay()
            if torch and episode_count % CHECKPOINT_INTERVAL == 0:
                agent.save(checkpoint_path)
        else:
            agent.last_state = state
            agent.last_action = action
            agent.last_distance = dist

if __name__ == "__main__":
    main()
