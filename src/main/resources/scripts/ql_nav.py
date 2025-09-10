#!/usr/bin/env python3
import json, math, os, random, sys, threading
from collections import deque
from queue import Queue, Empty

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

# -----------------------------
# Config
# -----------------------------
# State: [dist_norm, f_err_norm, l_err_norm, v_f/5, v_l/5, cos(yaw_err), sin(yaw_err)]
STATE_DIM = 7

# Discrete heads (kept small, but composite at runtime)
MOVE_DIM   = 3  # idle, forward, back
STRAFE_DIM = 3  # idle, left, right
YAW_DIM    = 3  # none, left, right
PITCH_DIM  = 3  # none, up, down
JUMP_DIM   = 2  # no, yes

ACTION_REPEAT = 3
CHECKPOINT_INTERVAL = 10
PLOT_EVERY = 5

GOAL_RANGE_NORM = 20.0  # distance normalization

if torch:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Network
# -----------------------------
if torch:
    class MultiHeadDuelingDQN(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = 128
            self.features = nn.Sequential(
                nn.Linear(STATE_DIM, hidden),
                nn.ReLU(),
            )
            # Value stream shared
            self.val = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1)
            )
            # Advantage streams per head
            self.adv_move   = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, MOVE_DIM))
            self.adv_strafe = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, STRAFE_DIM))
            self.adv_yaw    = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, YAW_DIM))
            self.adv_pitch  = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, PITCH_DIM))
            self.adv_jump   = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, JUMP_DIM))

        def forward(self, x):
            f = self.features(x)
            v = self.val(f)

            def dueling(adv):
                a = adv(f)
                return v + a - a.mean(dim=1, keepdim=True)

            return {
                "move":   dueling(self.adv_move),
                "strafe": dueling(self.adv_strafe),
                "yaw":    dueling(self.adv_yaw),
                "pitch":  dueling(self.adv_pitch),
                "jump":   dueling(self.adv_jump),
            }

# -----------------------------
# Agent
# -----------------------------
class Agent:
    def __init__(self):
        # Faster ε schedule since heads are small
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay_steps = 10_000
        self.global_step = 0
        self.eval_mode = False

        self.gamma = 0.99
        self.batch_size = 128
        self.memory = deque(maxlen=100_000)
        self.update_every = 4
        self.replay_min = 512
        self.tau = 0.005

        self.action_repeat = ACTION_REPEAT
        self.repeat_left = 0
        self.cached_action = None  # cached composite for repeat

        if torch:
            self.model = MultiHeadDuelingDQN().to(DEVICE)
            self.target_model = MultiHeadDuelingDQN().to(DEVICE)
            self.target_model.load_state_dict(self.model.state_dict())
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            self.loss_fn = nn.SmoothL1Loss()
            self.learn_step = 0

        self.last_state = None
        self.last_action = None  # dict with ints
        self.last_dist_raw = None
        self.prev_yaw_err = None
        self._save_thread = None

    def current_epsilon(self):
        if self.eval_mode:
            return 0.0
        if self.global_step >= self.epsilon_decay_steps:
            return self.epsilon_end
        frac = self.global_step / float(self.epsilon_decay_steps)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac

    def save(self, path):
        if not torch:
            return
        if self._save_thread and self._save_thread.is_alive():
            return

        def _save():
            data = {
                "model": self.model.state_dict(),
                "target_model": self.target_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "memory": list(self.memory),
                "global_step": self.global_step,
                "eps_params": (self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps),
                "meta": {
                    "state_dim": STATE_DIM,
                    "heads": [MOVE_DIM, STRAFE_DIM, YAW_DIM, PITCH_DIM, JUMP_DIM]
                }
            }
            torch.save(data, path)

        self._save_thread = threading.Thread(target=_save, daemon=True)
        self._save_thread.start()

    def load(self, path):
        if not torch or not os.path.exists(path):
            return False
        try:
            data = torch.load(path, map_location=DEVICE)
        except Exception:
            return False
        meta = data.get("meta", {})
        if meta.get("state_dim") != STATE_DIM or meta.get("heads") != [MOVE_DIM, STRAFE_DIM, YAW_DIM, PITCH_DIM, JUMP_DIM]:
            return False

        if "model" in data:
            self.model.load_state_dict(data["model"])
        if "target_model" in data:
            self.target_model.load_state_dict(data["target_model"])
        else:
            self.target_model.load_state_dict(self.model.state_dict())
        if "optimizer" in data:
            try: self.optimizer.load_state_dict(data["optimizer"])
            except Exception: pass
        if "memory" in data:
            self.memory = deque(data["memory"], maxlen=100_000)
        self.global_step = data.get("global_step", 0)
        eps = data.get("eps_params", None)
        if eps:
            self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps = eps
        return True

    # ----- State -----
    def _build_state(self, data):
        dx = (data["gx"] - data["px"])
        dz = (data["gz"] - data["pz"])

        yaw = math.radians(data["yaw"])
        cy, sy = math.cos(yaw), math.sin(yaw)

        f_err =  cy * dx + sy * dz
        l_err = -sy * dx + cy * dz
        dist_raw = math.hypot(dx, dz)

        vx, vz = data["vx"], data["vz"]
        v_f =  cy * vx + sy * vz
        v_l = -sy * vx + cy * vz

        goal_bearing = math.atan2(dz, dx)
        yaw_err = (yaw - goal_bearing + math.pi) % (2 * math.pi) - math.pi

        dist_norm = min(dist_raw / GOAL_RANGE_NORM, 1.0)
        state_list = [
            dist_norm,
            max(-1.0, min(1.0, f_err / GOAL_RANGE_NORM)),
            max(-1.0, min(1.0, l_err / GOAL_RANGE_NORM)),
            v_f / 5.0,
            v_l / 5.0,
            math.cos(yaw_err),
            math.sin(yaw_err),
        ]
        if torch:
            state = torch.tensor(state_list, dtype=torch.float32, device=DEVICE)
        else:
            state = state_list
        return state, dist_raw, yaw_err, v_f

    # ----- Policy (ε-greedy per head) -----
    def select_actions(self, q_heads):
        eps = self.current_epsilon()
        actions = {}
        for name, q in q_heads.items():
            dim = q.shape[1]
            if random.random() < eps:
                actions[name] = random.randrange(dim)
            else:
                actions[name] = int(torch.argmax(q, dim=1).item())
        return actions

    # ----- Replay memory -----
    def remember(self, s, a_dict, r, ns, done):
        if torch:
            # store compact as tuple for speed
            a = (a_dict["move"], a_dict["strafe"], a_dict["yaw"], a_dict["pitch"], a_dict["jump"])
            self.memory.append((s, a, r, ns, done))

    def replay(self):
        if not torch or len(self.memory) < self.replay_min:
            return
        if self.global_step % self.update_every != 0:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([b[0] for b in batch]).to(DEVICE)
        actions = torch.tensor([b[1] for b in batch], device=DEVICE, dtype=torch.long)  # shape [B, 5]
        rewards = torch.tensor([b[2] for b in batch], device=DEVICE, dtype=torch.float32)
        next_states = torch.stack([b[3] for b in batch]).to(DEVICE)
        dones = torch.tensor([b[4] for b in batch], device=DEVICE, dtype=torch.float32)

        # Current Q(s,a): sum across heads of the chosen action's Q
        q_heads = self.model(states)
        q_move   = q_heads["move"].gather(1, actions[:,0:1]).squeeze(1)
        q_strafe = q_heads["strafe"].gather(1, actions[:,1:2]).squeeze(1)
        q_yaw    = q_heads["yaw"].gather(1, actions[:,2:3]).squeeze(1)
        q_pitch  = q_heads["pitch"].gather(1, actions[:,3:4]).squeeze(1)
        q_jump   = q_heads["jump"].gather(1, actions[:,4:5]).squeeze(1)
        q_sum = q_move + q_strafe + q_yaw + q_pitch + q_jump

        # Next Q target: greedy per head on online net, evaluate on target net, sum
        with torch.no_grad():
            next_q_heads_online = self.model(next_states)
            greedy = {
                "move":   torch.argmax(next_q_heads_online["move"], dim=1),
                "strafe": torch.argmax(next_q_heads_online["strafe"], dim=1),
                "yaw":    torch.argmax(next_q_heads_online["yaw"], dim=1),
                "pitch":  torch.argmax(next_q_heads_online["pitch"], dim=1),
                "jump":   torch.argmax(next_q_heads_online["jump"], dim=1),
            }
            next_q_heads_target = self.target_model(next_states)
            nq_move   = next_q_heads_target["move"  ].gather(1, greedy["move"  ].unsqueeze(1)).squeeze(1)
            nq_strafe = next_q_heads_target["strafe"].gather(1, greedy["strafe"].unsqueeze(1)).squeeze(1)
            nq_yaw    = next_q_heads_target["yaw"   ].gather(1, greedy["yaw"   ].unsqueeze(1)).squeeze(1)
            nq_pitch  = next_q_heads_target["pitch" ].gather(1, greedy["pitch" ].unsqueeze(1)).squeeze(1)
            nq_jump   = next_q_heads_target["jump"  ].gather(1, greedy["jump"  ].unsqueeze(1)).squeeze(1)
            next_q_sum = nq_move + nq_strafe + nq_yaw + nq_pitch + nq_jump

            target = rewards + self.gamma * next_q_sum * (1.0 - dones)

        loss = self.loss_fn(q_sum, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        with torch.no_grad():
            for tp, p in zip(self.target_model.parameters(), self.model.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        self.learn_step += 1

# -----------------------------
# Main loop
# -----------------------------
def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "ql_nav_checkpoint.pth"
    eval_mode = len(sys.argv) > 2 and str(sys.argv[2]).lower() == "true"

    agent = Agent()
    if torch and agent.load(checkpoint_path):
        print(f"Loaded checkpoint from {checkpoint_path}", file=sys.stderr)
    else:
        print("No compatible checkpoint found, starting fresh.", file=sys.stderr)

    agent.eval_mode = bool(eval_mode)

    episode_rewards, rolling_rewards = [], []
    episode_reward, episode_count = 0.0, 0

    if plt:
        plt.ion()
        fig, ax = plt.subplots()
        plt.show(block=False)

    q = Queue()
    def _reader():
        for line in sys.stdin:
            q.put(line)
    threading.Thread(target=_reader, daemon=True).start()

    while True:
        if plt:
            plt.pause(0.001)

        try:
            line = q.get(timeout=0.0)
        except Empty:
            continue

        if not line:
            break
        data = json.loads(line)

        state, dist_raw, yaw_err, v_f = agent._build_state(data)
        done = data.get("done", False)
        reached = data.get("reached", False)

        # Dense reward on delta distance (blocks) + small alignment bonus
        progress = (agent.last_dist_raw - dist_raw) if agent.last_dist_raw is not None else 0.0
        align_improve = (math.cos(yaw_err) - math.cos(agent.prev_yaw_err)) if agent.prev_yaw_err is not None else 0.0

        reward = 2.5 * progress + 0.2 * align_improve
        if done:
            reward += 50.0 if reached else -5.0

        # Learn
        if agent.last_state is not None and not agent.eval_mode:
            agent.remember(agent.last_state, agent.last_action, reward, state, float(done))
            agent.replay()

        episode_reward += reward

        # Action selection with repeat (composite)
        if agent.repeat_left > 0 and agent.cached_action is not None:
            action = agent.cached_action
            agent.repeat_left -= 1
        else:
            if torch:
                with torch.no_grad():
                    q_heads = agent.model(state.unsqueeze(0))
                action = agent.select_actions(q_heads)
            else:
                action = {
                    "move": random.randrange(MOVE_DIM),
                    "strafe": random.randrange(STRAFE_DIM),
                    "yaw": random.randrange(YAW_DIM),
                    "pitch": random.randrange(PITCH_DIM),
                    "jump": random.randrange(JUMP_DIM),
                }
            agent.cached_action = action
            agent.repeat_left = agent.action_repeat - 1

        # Emit JSON composite to Java
        sys.stdout.write(json.dumps(action) + "\n")
        sys.stdout.flush()

        agent.global_step += 1

        if done:
            episode_rewards.append(episode_reward)
            window = episode_rewards[-20:]
            rolling_rewards.append(sum(window) / max(1, len(window)))

            if plt and (PLOT_EVERY is not None) and (episode_count % PLOT_EVERY == 0):
                ax.clear()
                ax.plot(episode_rewards, label="Episode Reward")
                ax.plot(rolling_rewards, label="20-episode rolling avg")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Total Reward")
                ax.set_title("QLearningNavigator Reward Progress")
                ax.legend()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)

            # reset episode vars
            episode_reward = 0.0
            episode_count += 1
            agent.last_state = None
            agent.last_action = None
            agent.last_dist_raw = None
            agent.prev_yaw_err = None
            agent.repeat_left = 0
            agent.cached_action = None

            if torch and (not agent.eval_mode) and (episode_count % CHECKPOINT_INTERVAL == 0):
                agent.save(checkpoint_path)
        else:
            agent.last_state = state
            agent.last_action = action
            agent.last_dist_raw = dist_raw
            agent.prev_yaw_err = yaw_err

if __name__ == "__main__":
    main()
