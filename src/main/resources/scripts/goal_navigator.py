import sys, math, random, collections
import torch
import torch.nn as nn
import torch.optim as optim

# CPU-only, keep PyTorch light
torch.set_num_threads(1)
device = torch.device("cpu")

ACTIONS = ["forward", "back", "left", "right", "none"]  # add "none" to reduce jitter
N_ACTIONS = len(ACTIONS)

# ----- Model -----
class QNet(nn.Module):
    def __init__(self, in_dim=6, hid=64, out_dim=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, out_dim),
        )
    def forward(self, x):
        return self.net(x)

policy = QNet().to(device)
target = QNet().to(device)
target.load_state_dict(policy.state_dict())
target.eval()

optimizer = optim.AdamW(policy.parameters(), lr=3e-4, betas=(0.9, 0.99), weight_decay=1e-4)
loss_fn = nn.SmoothL1Loss()  # Huber

# ----- Replay Buffer -----
Transition = collections.namedtuple("Transition", "s a r s1 done")
buffer = collections.deque(maxlen=4096)
BATCH = 64
GAMMA = 0.98
TARGET_SYNC = 200         # steps
GRAD_CLIP = 1.0

# ε schedule
EPS_START = 0.20
EPS_END   = 0.02
EPS_STEPS = 5000
global_step = 0

prev_s = None
prev_a = None
prev_dist = None

def epsilon():
    return max(EPS_END, EPS_START - (EPS_START - EPS_END) * (global_step / EPS_STEPS))

def to_state(x, z, vx, vz, gx, gz):
    # Use relative, well-scaled features (still 6-D to match the wire)
    dx, dz = gx - x, gz - z
    dist = math.hypot(dx, dz)
    speed = math.hypot(vx, vz)
    # normalize-ish
    nx = dx / (dist + 1e-6)
    nz = dz / (dist + 1e-6)
    s = torch.tensor([nx, nz, vx, vz, dist, speed], dtype=torch.float32)
    return s

def select_action(s):
    if random.random() < epsilon():
        return random.randrange(N_ACTIONS)
    with torch.no_grad():
        q = policy(s.unsqueeze(0).to(device))
        return int(torch.argmax(q, dim=1).item())

def train_step():
    if len(buffer) < BATCH:
        return
    batch = random.sample(buffer, BATCH)
    s  = torch.stack([t.s for t in batch]).to(device)
    a  = torch.tensor([t.a for t in batch], dtype=torch.int64, device=device).unsqueeze(1)
    r  = torch.tensor([t.r for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
    d  = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device).unsqueeze(1)
    s1 = torch.stack([t.s1 for t in batch]).to(device)

    # Double DQN target
    with torch.no_grad():
        next_actions = policy(s1).argmax(dim=1, keepdim=True)          # a* = argmax_a Q(s', a; policy)
        next_q = target(s1).gather(1, next_actions)                    # Q(s', a*; target)
        y = r + (1.0 - d) * GAMMA * next_q

    q = policy(s).gather(1, a)
    loss = loss_fn(q, y)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
    optimizer.step()

# ----- Main loop -----
for raw in sys.stdin:
    parts = raw.strip().split(',')
    if len(parts) != 7:
        continue

    x, z, vx, vz, gx, gz = map(float, parts[:6])
    done_flag = int(parts[6]) == 1

    s = to_state(x, z, vx, vz, gx, gz)
    dist = s[4].item()

    # --- reward shaping (based on previous step -> now) ---
    reward = 0.0
    if prev_s is not None:
        # progress reward: decrease in distance
        reward += (prev_dist - dist) * 3.0
        # time penalty (prefer faster solutions)
        reward += -0.01
        # alignment reward: velocity aligned to goal direction
        # prev_s[0:2] is normalized (nx, nz), so dot with velocity direction
        vx_prev, vz_prev = prev_s[2].item(), prev_s[3].item()
        sp = math.hypot(vx_prev, vz_prev)
        if sp > 1e-5:
            align = (prev_s[0].item() * (vx_prev / sp) + prev_s[1].item() * (vz_prev / sp))
            reward += 0.05 * align
        # success bonus if episode ended by proximity (host also signals done at timeout)
        # we infer success if current distance is small
        if dist <= 0.8:
            reward += 1.0

        buffer.append(Transition(prev_s, prev_a, reward, s, 1.0 if done_flag else 0.0))
        train_step()

    # act
    a = select_action(s)

    # output action immediately
    sys.stdout.write(ACTIONS[a] + "\n")
    sys.stdout.flush()

    prev_s = None if done_flag else s
    prev_a = None if done_flag else a
    prev_dist = None if done_flag else dist

    global_step += 1
    if global_step % TARGET_SYNC == 0:
        target.load_state_dict(policy.state_dict())
