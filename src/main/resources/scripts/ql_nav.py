#!/usr/bin/env python3
import json
import math
import os
import random
import sys
import threading
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
STATE_DIM = 5   # dx, dz, vx, vz, sin(yaw_err)
ACTION_DIM = 8  # forward, back, left, right, idle, jump, look_left, look_right
CHECKPOINT_INTERVAL = 10
PLOT_EVERY = 5  # plot every N episodes (set None to disable)

if torch:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Networks
# -----------------------------
if torch:
    class DQN(nn.Module):
        """Dueling DQN head: V(s) + A(s,a) - mean(A)"""
        def __init__(self):
            super().__init__()
            hidden = 128
            self.features = nn.Sequential(
                nn.Linear(STATE_DIM, hidden),
                nn.ReLU(),
            )
            self.adv = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, ACTION_DIM)
            )
            self.val = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1)
            )

        def forward(self, x):
            f = self.features(x)
            a = self.adv(f)
            v = self.val(f)
            return v + a - a.mean(dim=1, keepdim=True)


# -----------------------------
# Agent
# -----------------------------
class Agent:
    def __init__(self):
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay_steps = 50_000
        self.global_step = 0
        self.eval_mode = False

        self.gamma = 0.99
        self.batch_size = 128
        self.memory = deque(maxlen=50_000)
        self.update_every = 4
        self.replay_min = 1_000
        self.tau = 0.005

        if torch:
            self.model = DQN().to(DEVICE)
            self.target_model = DQN().to(DEVICE)
            self.target_model.load_state_dict(self.model.state_dict())
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
            self.loss_fn = nn.SmoothL1Loss()
            self.learn_step = 0

        self.last_state = None
        self.last_action = None
        self.last_distance = None
        self.prev_yaw_err = None
        self.start_distance = None
        self._save_thread = None

    def current_epsilon(self):
        if self.eval_mode:
            return 0.0
        if self.global_step >= self.epsilon_decay_steps:
            return self.epsilon_end
        frac = 1.0 - (self.global_step / float(self.epsilon_decay_steps))
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * frac

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
            }
            torch.save(data, path)

        self._save_thread = threading.Thread(target=_save, daemon=True)
        self._save_thread.start()

    def load(self, path):
        if not torch or not os.path.exists(path):
            return False
        data = torch.load(path, map_location=DEVICE)
        if "model" in data:
            self.model.load_state_dict(data["model"])
        if "target_model" in data:
            self.target_model.load_state_dict(data["target_model"])
        else:
            self.target_model.load_state_dict(self.model.state_dict())
        if "optimizer" in data:
            try:
                self.optimizer.load_state_dict(data["optimizer"])
            except Exception:
                pass
        if "memory" in data:
            self.memory = deque(data["memory"], maxlen=50_000)
        self.global_step = data.get("global_step", 0)
        eps = data.get("eps_params", None)
        if eps:
            self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps = eps
        return True

    def normalize(self, data):
        dx_raw = (data["gx"] - data["px"])
        dz_raw = (data["gz"] - data["pz"])

        yaw_rad = math.radians(data["yaw"])
        goal_bearing = math.atan2(dz_raw, dx_raw)
        yaw_err = (yaw_rad - goal_bearing + math.pi) % (2 * math.pi) - math.pi
        yaw_err_mag = abs(yaw_err)

        dx = dx_raw / 50.0
        dz = dz_raw / 50.0
        vx = data["vx"] / 5.0
        vz = data["vz"] / 5.0
        sy = math.sin(yaw_err)

        state = torch.tensor([dx, dz, vx, vz, sy], dtype=torch.float32, device=DEVICE) if torch else [dx, dz, vx, vz, sy]
        dist = math.hypot(dx, dz)
        return state, dist, yaw_err_mag

    def act(self, state):
        if torch and random.random() > self.current_epsilon():
            with torch.no_grad():
                q = self.model(state.unsqueeze(0))
                return int(torch.argmax(q, dim=1).item())
        return random.randrange(ACTION_DIM)

    def remember(self, s, a, r, ns, done):
        if torch:
            self.memory.append((s, a, r, ns, done))

    def replay(self):
        if not torch or len(self.memory) < self.replay_min:
            return
        if self.global_step % self.update_every != 0:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([b[0] for b in batch]).to(DEVICE)
        actions = torch.tensor([b[1] for b in batch], device=DEVICE, dtype=torch.long)
        rewards = torch.tensor([b[2] for b in batch], device=DEVICE, dtype=torch.float32)
        next_states = torch.stack([b[3] for b in batch]).to(DEVICE)
        dones = torch.tensor([b[4] for b in batch], device=DEVICE, dtype=torch.float32)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(dim=1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, target)
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
        print("No checkpoint found, starting fresh.", file=sys.stderr)

    agent.eval_mode = bool(eval_mode)

    episode_rewards = []
    rolling_rewards = []
    episode_reward = 0.0
    episode_count = 0

    if plt:
        plt.ion()
        fig, ax = plt.subplots()
        plt.show(block=False)

    # --------------------
    # stdin reader thread
    # --------------------
    q = Queue()
    def _reader():
        for line in sys.stdin:
            q.put(line)
    threading.Thread(target=_reader, daemon=True).start()

    while True:
        # service GUI every tick
        if plt:
            plt.pause(0.001)

        try:
            line = q.get(timeout=0.0)
        except Empty:
            continue

        if not line:
            break
        data = json.loads(line)

        state, dist, yaw_err_mag = agent.normalize(data)
        done = data.get("done", False)
        reached = data.get("reached", False)
        stuck = data.get("stuck", False)

        if agent.last_distance is None:
            agent.last_distance = dist
            agent.start_distance = dist
        if agent.prev_yaw_err is None:
            agent.prev_yaw_err = yaw_err_mag

        progress = (agent.last_distance - dist) if agent.last_distance is not None else 0.0
        yaw_improve = (agent.prev_yaw_err - yaw_err_mag) if agent.prev_yaw_err is not None else 0.0

        reward = 3.0 * progress + 0.5 * yaw_improve - 0.01

        if dist < 0.06:
            reward += 0.05

        if done:
            if reached:
                reward += 100.0
            else:
                reward -= 10.0
                if stuck:
                    reward -= 10.0

        if agent.last_state is not None and not agent.eval_mode:
            agent.remember(agent.last_state, agent.last_action, reward, state, done)
            agent.replay()

        episode_reward += reward

        action = agent.act(state)
        sys.stdout.write(str(action) + "\n")
        sys.stdout.flush()

        agent.global_step += 1

        if done:
            episode_rewards.append(episode_reward)
            window = episode_rewards[-20:]
            rolling_rewards.append(sum(window) / len(window))

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

            episode_reward = 0.0
            episode_count += 1

            agent.last_state = None
            agent.last_action = None
            agent.last_distance = None
            agent.prev_yaw_err = None
            agent.start_distance = None

            if torch and (not agent.eval_mode) and (episode_count % CHECKPOINT_INTERVAL == 0):
                agent.save(checkpoint_path)
        else:
            agent.last_state = state
            agent.last_action = action
            agent.last_distance = dist
            agent.prev_yaw_err = yaw_err_mag


if __name__ == "__main__":
    main()
