import sys, math
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(1)
device = torch.device("cpu")

ACTIONS = ["forward","back","left","right","turn_left","turn_right","jump","none"]
N_ACTIONS = len(ACTIONS)

# in_dim:
# nx, nz (unit vector to goal), vx, vz, dist, speed, sin(dYaw), cos(dYaw), collide
IN_DIM = 9

class ActorCritic(nn.Module):
    def __init__(self, in_dim=IN_DIM, hid=96, out_dim=N_ACTIONS):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
        )
        self.pi = nn.Linear(hid, out_dim)
        self.v  = nn.Linear(hid, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)

policy = ActorCritic().to(device)
optimizer = optim.AdamW(policy.parameters(), lr=3e-4, betas=(0.9, 0.99), weight_decay=1e-4)
value_loss_fn = nn.SmoothL1Loss()

GAMMA = 0.98
ENTROPY_BETA = 0.015
VALUE_BETA   = 0.5
GRAD_CLIP    = 1.0

TAU_START = 1.00
TAU_END   = 0.20
TAU_STEPS = 6000
global_step = 0

def temperature():
    t = max(TAU_END, TAU_START - (TAU_START - TAU_END) * (global_step / TAU_STEPS))
    return t

prev_s = None
prev_a = None
prev_dist = None
prev_abs_dyaw = None

def to_state(x, z, vx, vz, gx, gz, dyaw_deg, collide):
    dx, dz = gx - x, gz - z
    dist = math.hypot(dx, dz)
    speed = math.hypot(vx, vz)
    if dist < 1e-6:
        nx, nz = 0.0, 0.0
    else:
        nx, nz = dx / dist, dz / dist
    dyaw_rad = math.radians(dyaw_deg)
    s = torch.tensor([nx, nz, vx, vz, dist, speed, math.sin(dyaw_rad), math.cos(dyaw_rad), float(collide)],
                     dtype=torch.float32)
    return s, dist, abs(dyaw_deg)

@torch.no_grad()
def select_action(s):
    logits, _ = policy(s.unsqueeze(0).to(device))
    tau = temperature()
    probs = torch.softmax(logits / tau, dim=1)
    return torch.distributions.Categorical(probs=probs).sample().item()

def a2c_update(prev_s, prev_a, reward, s1, done_flag):
    s0 = prev_s.unsqueeze(0).to(device)
    s1 = s1.unsqueeze(0).to(device)

    with torch.no_grad():
        _, v1 = policy(s1)
        target_v = reward + (0.0 if done_flag else GAMMA * v1.item())

    logits0, v0 = policy(s0)
    logp0 = torch.log_softmax(logits0, dim=1)[0, prev_a]
    p0 = torch.softmax(logits0, dim=1)
    entropy = -(p0 * torch.log_softmax(logits0, dim=1)).sum(dim=1).mean()

    target_v_t = torch.tensor(target_v, dtype=torch.float32, device=device)
    advantage = (target_v_t - v0.squeeze(0)).detach()

    policy_loss = -(logp0 * advantage)
    value_loss  = value_loss_fn(v0.squeeze(0), target_v_t)
    loss = policy_loss + VALUE_BETA * value_loss - ENTROPY_BETA * entropy

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
    optimizer.step()

    return float(loss.item()), float(entropy.item()), float(advantage.item())

for raw in sys.stdin:
    parts = raw.strip().split(',')
    if len(parts) != 9:
        # old format? ignore line
        continue

    x, z, vx, vz, gx, gz = map(float, parts[:6])
    dyaw = float(parts[6])
    collide = int(parts[7]) == 1
    done_flag = int(parts[8]) == 1

    s, dist, abs_dyaw = to_state(x, z, vx, vz, gx, gz, dyaw, collide)

    reward = 0.0
    loss_val = None

    if prev_s is not None:
        # progress toward goal
        reward += (prev_dist - dist) * 3.0
        # gentle penalty for large misalignment (prefer facing the goal)
        reward += -0.002 * prev_abs_dyaw
        # time penalty to encourage faster solutions
        reward += -0.01
        # if colliding, encourage jump/turn by small negative (discourage ramming)
        if collide:
            reward += -0.02
        # success bonus
        if dist <= 0.8:
            reward += 1.0

        loss_val, ent_val, adv_val = a2c_update(prev_s, prev_a, reward, s, done_flag)

    a = select_action(s)

    msg = ""
    if loss_val is not None and global_step % 120 == 0:
        msg = f"step {global_step} loss {loss_val:.4f} tau {temperature():.2f}"

    out = ACTIONS[a]
    if msg:
        out += "|" + msg
    sys.stdout.write(out + "\n")
    sys.stdout.flush()

    prev_s = None if done_flag else s
    prev_a = None if done_flag else a
    prev_dist = None if done_flag else dist
    prev_abs_dyaw = None if done_flag else abs_dyaw

    global_step += 1
