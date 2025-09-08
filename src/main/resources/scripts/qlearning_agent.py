# PYTHON BACKEND (qlearning_agent.py) — drop-in replacement
# DQN with replay buffer, target network, and dense rewards.
# Updated to support start-relative goals (goal_rel) and one action per input line (tick).

import sys
import json
import os
import math
import time
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "qlearning_checkpoint.pt"

MAX_GOAL_DIST = 200.0   # keep in sync with Java
REPLAY_CAPACITY = 50_000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
TARGET_SYNC_EVERY = 1000   # steps
LEARN_START = 1000         # don't learn until we have some data
LEARN_EVERY = 1            # learn each step after warmup
GRAD_CLIP = 1.0

# Epsilon-greedy (starts high, decays to a floor)
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 50_000   # linear decay steps

# Reward shaping (time pressure encourages speed)
STEP_PENALTY = -0.005
SUCCESS_BONUS = 1.0
FAIL_PENALTY = -0.5
PROGRESS_SCALE = 0.05      # (prev_dist - dist) * scale per step
ORIENT_SCALE = 0.01        # small reward for reducing |yaw_diff|

ACTIONS = [
    "forward",
    "back",
    "left",
    "right",
    "jump",
    "turn_left",
    "turn_right",
    "none",
]
NUM_ACTIONS = len(ACTIONS)

def angle_wrap(a: float) -> float:
    # wrap to [-pi, pi]
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a

# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

class DQN(nn.Module):
    # Inputs: [dx_norm, dz_norm, cos(yaw_diff), sin(yaw_diff)]
    def __init__(self, input_dim=4, hidden=128, outputs=NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, outputs),
        )

    def forward(self, x):
        return self.net(x)

# --------------------------------------------------------------------------------------
# Replay Buffer
# --------------------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.as_tensor(s, dtype=torch.float32, device=device),
            torch.as_tensor(a, dtype=torch.long, device=device),
            torch.as_tensor(r, dtype=torch.float32, device=device),
            torch.as_tensor(s2, dtype=torch.float32, device=device),
            torch.as_tensor(d, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buf)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def build_state(px, pz, yaw_deg, gx, gz):
    dx = gx - px
    dz = gz - pz

    # normalize distances
    dxn = max(-1.0, min(1.0, dx / MAX_GOAL_DIST))
    dzn = max(-1.0, min(1.0, dz / MAX_GOAL_DIST))

    # orientation: angle to goal vs player yaw
    target_bearing = math.atan2(dz, dx)  # [-pi, pi]
    yaw_rad = math.radians(yaw_deg)
    yaw_diff = angle_wrap(target_bearing - yaw_rad)

    return [dxn, dzn, math.cos(yaw_diff), math.sin(yaw_diff)], math.hypot(dx, dz), abs(yaw_diff)

def choose_action(qnet, state_tensor, epsilon):
    if random.random() < epsilon:
        return random.randrange(NUM_ACTIONS)
    with torch.no_grad():
        q = qnet(state_tensor)
        return int(torch.argmax(q, dim=1).item())

def compute_loss(qnet, target_net, batch):
    s, a, r, s2, d = batch
    q_vals = qnet(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(s2).max(1).values
        target = r + (1.0 - d) * GAMMA * next_q
    loss = nn.functional.smooth_l1_loss(q_vals, target)
    return loss

# --------------------------------------------------------------------------------------
# Init
# --------------------------------------------------------------------------------------

qnet = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(qnet.state_dict())
optimizer = optim.Adam(qnet.parameters(), lr=LR)
replay = ReplayBuffer()

global_step = 0
epsilon = EPS_START

# load checkpoint if exists
if os.path.exists(checkpoint_path):
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        qnet.load_state_dict(ckpt["qnet"])
        target_net.load_state_dict(ckpt.get("target_net", ckpt["qnet"]))
        optimizer.load_state_dict(ckpt["opt"])
        global_step = ckpt.get("global_step", 0)
        epsilon = ckpt.get("epsilon", epsilon)
        print(f"Loaded checkpoint from {checkpoint_path}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"Failed to load checkpoint ({e}), starting fresh", file=sys.stderr, flush=True)
else:
    print("No checkpoint found, starting fresh", file=sys.stderr, flush=True)

# --------------------------------------------------------------------------------------
# Stream loop (one line per tick)
# --------------------------------------------------------------------------------------

prev_state_vec = None
prev_action = None
prev_dist = None
prev_yaw_abs = None
episode_return = 0.0
episode_steps = 0
episodes = 0
last_reset_time = None

def save_ckpt():
    tmp = checkpoint_path + ".tmp"
    torch.save(
        {
            "qnet": qnet.state_dict(),
            "target_net": target_net.state_dict(),
            "opt": optimizer.state_dict(),
            "epsilon": epsilon,
            "global_step": global_step,
        },
        tmp,
    )
    os.replace(tmp, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}", file=sys.stderr, flush=True)

for line in sys.stdin:
    data = json.loads(line)

    # explicit save request from client on deactivate
    if data.get("save"):
        save_ckpt()
        break

    reset = bool(data.get("reset", False))
    success = bool(data.get("success", False))
    fail = bool(data.get("fail", False))

    # Player pose
    px = float(data["player"]["x"])
    pz = float(data["player"]["z"])
    yaw = float(data["player"]["yaw"])

    # Determine absolute goal from start-relative coords when present
    if "goal_rel" in data and "start" in data:
        gx = float(data["start"]["x"]) + float(data["goal_rel"]["dx"])
        gz = float(data["start"]["z"]) + float(data["goal_rel"]["dz"])
    else:
        gx = float(data["goal"]["x"])
        gz = float(data["goal"]["z"])

    state_vec, dist, yaw_abs = build_state(px, pz, yaw, gx, gz)
    state_t = torch.tensor([state_vec], dtype=torch.float32, device=device)

    # ---------------- reward shaping for the transition that JUST happened ----------------
    if prev_state_vec is not None and prev_action is not None and prev_dist is not None:
        progress = (prev_dist - dist) * PROGRESS_SCALE
        orient_improve = 0.0
        if prev_yaw_abs is not None:
            orient_improve = (prev_yaw_abs - yaw_abs) * ORIENT_SCALE

        r = STEP_PENALTY + progress + orient_improve

        done = False
        if reset:
            if success:
                r += SUCCESS_BONUS
            elif fail:
                r += FAIL_PENALTY
            done = True

        # Store transition in replay
        replay.push(prev_state_vec, prev_action, r, state_vec, float(done))

        episode_return += r
        episode_steps += 1

        # Learn
        if len(replay) >= LEARN_START and (global_step % LEARN_EVERY == 0):
            batch = replay.sample(BATCH_SIZE)
            loss = compute_loss(qnet, target_net, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(qnet.parameters(), GRAD_CLIP)
            optimizer.step()

        # Target sync
        if global_step % TARGET_SYNC_EVERY == 0:
            target_net.load_state_dict(qnet.state_dict())

        global_step += 1

        # Episode accounting
        if reset:
            episodes += 1
            elapsed = 0.0
            if last_reset_time is not None:
                elapsed = time.time() - last_reset_time
            print(
                f"Episode {episodes}: return={episode_return:.3f} steps={episode_steps} "
                f"eps={epsilon:.3f} ({'SUCCESS' if success else 'FAIL' if fail else 'RESET'}) in {elapsed:.2f}s",
                file=sys.stderr,
                flush=True,
            )
            # occasional autosave
            if episodes % 50 == 0:
                save_ckpt()

            last_reset_time = time.time()
            episode_return = 0.0
            episode_steps = 0

    # ---------------- action selection for the NEXT tick ----------------
    # epsilon linear decay
    epsilon = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (global_step / EPS_DECAY_STEPS))

    action_id = choose_action(qnet, state_t, epsilon)
    sys.stdout.write(ACTIONS[action_id] + "\n")
    sys.stdout.flush()

    prev_state_vec = state_vec
    prev_action = action_id
    prev_dist = dist
    prev_yaw_abs = yaw_abs
