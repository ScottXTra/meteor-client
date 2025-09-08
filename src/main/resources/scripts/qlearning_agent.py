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
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

net = DQN().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.MSELoss()

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
gamma = 0.95

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
    dy = gy - py
    dz = gz - pz
    horiz_dist = math.sqrt(dx * dx + dz * dz)
    distance = math.sqrt(horiz_dist * horiz_dist + dy * dy)

    desired_yaw = math.degrees(math.atan2(dz, dx))
    yaw_diff = ((desired_yaw - yaw + 180) % 360) - 180
    desired_pitch = -math.degrees(math.atan2(dy, horiz_dist)) if horiz_dist > 0 else 0.0
    pitch_diff = desired_pitch - pitch

    state = torch.tensor([[dx / 100, dz / 100, yaw_diff / 180, pitch_diff / 180]], dtype=torch.float32, device=device)

    if prev_state is not None:
        reward = prev_distance - distance - 0.01
        if reset:
            reward += 1.0
        target = net(prev_state).detach().clone()
        next_q = 0.0 if reset else net(state).max().detach()
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
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
