#!/usr/bin/env python3
import json
import math
import random
import sys
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None

STATE_DIM = 4  # dx, dz, vx, vz
ACTION_DIM = 5  # forward, back, left, right, idle

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
    agent = Agent()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        data = json.loads(line)
        state, dist = agent.normalize(data)
        reward = data.get("reward", 0.0)
        done = data.get("done", False)

        if agent.last_distance is None:
            agent.last_distance = dist

        if reward == 0.0:
            reward = agent.last_distance - dist - 0.01

        if done:
            if dist < 0.02:
                reward += 100.0
            else:
                reward -= 10.0

        if agent.last_state is not None:
            agent.remember(agent.last_state, agent.last_action, reward, state, done)
            agent.replay()

        action = agent.act(state)
        sys.stdout.write(str(action) + "\n")
        sys.stdout.flush()

        if done:
            agent.last_state = None
            agent.last_action = None
            agent.last_distance = None
            agent.decay()
        else:
            agent.last_state = state
            agent.last_action = action
            agent.last_distance = dist

if __name__ == "__main__":
    main()
