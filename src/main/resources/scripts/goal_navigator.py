import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["forward", "back", "left", "right"]

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)

net = QNet()
optimizer = optim.Adam(net.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

gamma = 0.9
epsilon = 0.1

prev_state = None
prev_action = None
prev_distance = None

for line in sys.stdin:
    parts = line.strip().split(',')
    if len(parts) != 7:
        continue
    x, z, vx, vz, gx, gz = map(float, parts[:6])
    done = int(parts[6]) == 1
    state = torch.tensor([x, z, vx, vz, gx, gz], dtype=torch.float32).unsqueeze(0)
    distance = math.sqrt((gx - x) ** 2 + (gz - z) ** 2)

    if prev_state is not None:
        reward = prev_distance - distance
        with torch.no_grad():
            target_value = reward
            if not done:
                target_value += gamma * net(state).max().item()
        q_values = net(prev_state)
        target = q_values.clone().detach()
        target[0, prev_action] = target_value
        loss = loss_fn(q_values, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if torch.rand(1).item() < epsilon:
        action = torch.randint(0, 4, (1,)).item()
    else:
        with torch.no_grad():
            action = net(state).argmax().item()

    sys.stdout.write(ACTIONS[action] + "\n")
    sys.stdout.flush()

    prev_state = None if done else state
    prev_action = None if done else action
    prev_distance = None if done else distance
