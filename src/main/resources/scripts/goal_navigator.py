import sys, math, random
import torch
import torch.nn as nn
import torch.optim as optim

# CPU-only, keep PyTorch light
torch.set_num_threads(1)
device = torch.device("cpu")

ACTIONS = ["forward", "back", "left", "right", "none"]
N_ACTIONS = len(ACTIONS)

# ----- Model: shared torso + policy & value heads (A2C) -----
class ActorCritic(nn.Module):
    def __init__(self, in_dim=6, hid=64, out_dim=N_ACTIONS):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hid, out_dim)   # policy logits
        self.v  = nn.Linear(hid, 1)         # state value

    def forward(self, x):
        h = self.body(x)
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value

policy = ActorCritic().to(device)

optimizer = optim.AdamW(policy.parameters(), lr=3e-4, betas=(0.9, 0.99), weight_decay=1e-4)
value_loss_fn = nn.SmoothL1Loss()  # Huber for value

# ----- A2C hyperparams -----
GAMMA = 0.98
ENTROPY_BETA = 0.01
VALUE_BETA   = 0.5
GRAD_CLIP    = 1.0

# Temperature schedule for Boltzmann exploration (anneal over time)
TAU_START = 1.00
TAU_END   = 0.20
TAU_STEPS = 5000
global_step = 0

def temperature():
    t = max(TAU_END, TAU_START - (TAU_START - TAU_END) * (global_step / TAU_STEPS))
    return t

prev_s = None
prev_a = None
prev_dist = None

def to_state(x, z, vx, vz, gx, gz):
    # Same 6-D features as before (relative & well-scaled)
    dx, dz = gx - x, gz - z
    dist = math.hypot(dx, dz)
    speed = math.hypot(vx, vz)
    nx = dx / (dist + 1e-6)
    nz = dz / (dist + 1e-6)
    s = torch.tensor([nx, nz, vx, vz, dist, speed], dtype=torch.float32)
    return s

@torch.no_grad()
def select_action(s):
    # Boltzmann policy with annealed temperature
    logits, _ = policy(s.unsqueeze(0).to(device))
    tau = temperature()
    probs = torch.softmax(logits / tau, dim=1)
    a = torch.distributions.Categorical(probs=probs).sample().item()
    return a

def a2c_update(prev_s, prev_a, reward, s1, done_flag):
    # One-step A2C: bootstrap with V(s1)
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

# ----- Main loop (same I/O contract) -----
for raw in sys.stdin:
    parts = raw.strip().split(',')
    if len(parts) != 7:
        continue

    x, z, vx, vz, gx, gz = map(float, parts[:6])
    done_flag = int(parts[6]) == 1

    s = to_state(x, z, vx, vz, gx, gz)
    dist = s[4].item()

    # --- reward shaping (unchanged) ---
    reward = 0.0
    loss_val = None
    if prev_s is not None:
        # progress reward
        reward += (prev_dist - dist) * 3.0
        # time penalty (prefer faster solutions)
        reward += -0.01
        # alignment reward: velocity aligned to goal direction (from previous step)
        vx_prev, vz_prev = prev_s[2].item(), prev_s[3].item()
        sp = math.hypot(vx_prev, vz_prev)
        if sp > 1e-5:
            align = (prev_s[0].item() * (vx_prev / sp) + prev_s[1].item() * (vz_prev / sp))
            reward += 0.05 * align
        # success bonus if close to goal
        if dist <= 0.8:
            reward += 1.0

        # A2C one-step update
        loss_val, ent_val, adv_val = a2c_update(prev_s, prev_a, reward, s, done_flag)

    # act from current state
    a = select_action(s)

    # optional training info every 100 steps
    msg = ""
    if loss_val is not None and global_step % 100 == 0:
        msg = f"step {global_step} loss {loss_val:.4f} tau {temperature():.2f}"

    # output action (and optional message separated by '|') immediately
    out = ACTIONS[a]
    if msg:
        out += "|" + msg
    sys.stdout.write(out + "\n")
    sys.stdout.flush()

    prev_s   = None if done_flag else s
    prev_a   = None if done_flag else a
    prev_dist= None if done_flag else dist

    global_step += 1
