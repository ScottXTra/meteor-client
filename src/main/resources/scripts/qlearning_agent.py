import sys
import json
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure we use CUDA 11.8 build if available
# This script expects torch installed with CUDA 11.8 support.
# Example: pip install torch --index-url https://download.pytorch.org/whl/cu118

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

net = DQN().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.01)
criterion = nn.MSELoss()

epsilon = 0.2
gamma = 0.9

prev_state = None
prev_action = None
prev_distance = None
last_reset_time = None

actions = [
    "forward",
    "back",
    "left",
    "right",
    "sprint-forward",
    "look-left",
    "look-right",
    "look-up",
    "look-down",
]

for line in sys.stdin:
    data = json.loads(line)
    reset = data.get("reset", False)
    px, py, pz = data["player"]["x"], data["player"]["y"], data["player"]["z"]
    yaw, pitch = data["player"]["yaw"], data["player"]["pitch"]
    gx, gy, gz = data["goal"]["x"], data["goal"]["y"], data["goal"]["z"]

    dx = gx - px
    dz = gz - pz
    state = torch.tensor([[dx, dz, yaw, pitch]], dtype=torch.float32, device=device)
    distance = math.sqrt(dx * dx + dz * dz)

    reward = None
    if prev_state is not None and not reset:
        reward = prev_distance - distance - 0.01
        target = net(prev_state).detach().clone()
        next_q = net(state).max().detach()
        target[0, prev_action] = reward + gamma * next_q
        output = net(prev_state)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if reset:
        if last_reset_time is not None:
            elapsed = time.time() - last_reset_time
            print(f"Reached goal in {elapsed:.2f}s", file=sys.stderr, flush=True)
        last_reset_time = time.time()
        prev_state = None
        prev_action = None
        prev_distance = None
        print(f"Updated goal: {gx:.2f}, {gy:.2f}, {gz:.2f}", file=sys.stderr, flush=True)

    if torch.rand(1).item() < epsilon:
        action = torch.randint(0, len(actions), (1,)).item()
    else:
        with torch.no_grad():
            action = net(state).argmax().item()

    sys.stdout.write(actions[action] + "\n")
    sys.stdout.flush()

    prev_state = state
    prev_action = action
    prev_distance = distance
