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

STATE_DIM = 18
MOVE_DIM   = 3
STRAFE_DIM = 3
YAW_DIM    = 3
PITCH_DIM  = 3
JUMP_DIM   = 2

ACTION_REPEAT = 2
CHECKPOINT_INTERVAL = 10
PLOT_EVERY = 5

POSITION_SCALE = 30.0
VELOCITY_SCALE = 5.0
TIME_SCALE = 8.0
DISTANCE_SCALE = 30.0

if torch:
    class MultiHeadDuelingDQN(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = 160
            self.features = nn.Sequential(
                nn.Linear(STATE_DIM, hidden),
                nn.ReLU(),
            )
            self.val = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1)
            )
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

class Agent:
    def __init__(self):
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = 15_000
        self.global_step = 0
        self.eval_mode = False

        self.gamma = 0.995
        self.batch_size = 128
        self.memory = deque(maxlen=200_000)
        self.update_every = 4
        self.replay_min = 1024
        self.tau = 0.01

        self.action_repeat = ACTION_REPEAT
        self.repeat_left = 0
        self.cached_action = None

        if torch:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = MultiHeadDuelingDQN().to(self.device)
            self.target_model = MultiHeadDuelingDQN().to(self.device)
            self.target_model.load_state_dict(self.model.state_dict())
            self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
            self.loss_fn = nn.SmoothL1Loss()
        else:
            self.device = None

        self.last_state = None
        self.last_action = None
        self.last_dist = None
        self.repeat_left = 0
        self.cached_action = None

    def current_epsilon(self):
        if self.eval_mode or not torch:
            return 0.0 if self.eval_mode else 1.0
        if self.global_step >= self.epsilon_decay_steps:
            return self.epsilon_end
        frac = self.global_step / float(self.epsilon_decay_steps)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac

    def save(self, path):
        if not torch:
            return
        data = {
            "model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "target_model": {k: v.detach().cpu() for k, v in self.target_model.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "eps_params": (self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps),
            "meta": {
                "state_dim": STATE_DIM,
                "heads": [MOVE_DIM, STRAFE_DIM, YAW_DIM, PITCH_DIM, JUMP_DIM],
                "format": "activity_trainer_v1",
            },
        }
        tmp = f"{path}.tmp"
        torch.save(data, tmp)
        os.replace(tmp, path)

    def load(self, path):
        if not torch or not os.path.exists(path):
            return False
        try:
            data = torch.load(path, map_location=self.device)
        except Exception:
            return False
        meta = data.get("meta", {})
        if meta.get("state_dim") != STATE_DIM:
            return False
        if meta.get("heads") != [MOVE_DIM, STRAFE_DIM, YAW_DIM, PITCH_DIM, JUMP_DIM]:
            return False
        self.model.load_state_dict(data.get("model", self.model.state_dict()))
        self.target_model.load_state_dict(data.get("target_model", self.model.state_dict()))
        opt_state = data.get("optimizer")
        if opt_state:
            try:
                self.optimizer.load_state_dict(opt_state)
            except Exception:
                pass
        self.global_step = data.get("global_step", 0)
        eps = data.get("eps_params")
        if eps:
            self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps = eps
        return True

    def reset_episode(self):
        self.last_state = None
        self.last_action = None
        self.last_dist = None
        self.repeat_left = 0
        self.cached_action = None

    def build_state(self, data):
        px, py, pz = data["px"], data["py"], data["pz"]
        vx, vy, vz = data["vx"], data["vy"], data["vz"]
        pitch = math.radians(data.get("pitch", 0.0))
        yaw = math.radians(data.get("yaw", 0.0))
        sprint = 1.0 if data.get("sprinting", False) else 0.0
        gx, gy, gz = data["gx"], data["gy"], data["gz"]

        dx = gx - px
        dy = gy - py
        dz = gz - pz

        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        progress = 0.0 if self.last_dist is None else self.last_dist - dist

        state_values = [
            max(-1.0, min(1.0, dx / POSITION_SCALE)),
            max(-1.0, min(1.0, dy / POSITION_SCALE)),
            max(-1.0, min(1.0, dz / POSITION_SCALE)),
            max(-1.0, min(1.0, px / POSITION_SCALE)),
            max(-1.0, min(1.0, py / POSITION_SCALE)),
            max(-1.0, min(1.0, pz / POSITION_SCALE)),
            max(-1.0, min(1.0, vx / VELOCITY_SCALE)),
            max(-1.0, min(1.0, vy / VELOCITY_SCALE)),
            max(-1.0, min(1.0, vz / VELOCITY_SCALE)),
            math.sin(pitch),
            math.cos(pitch),
            math.sin(yaw),
            math.cos(yaw),
            sprint,
            min(1.0, dist / DISTANCE_SCALE),
            max(0.0, min(1.0, data.get("time_remaining", 0.0) / TIME_SCALE)),
            max(0.0, min(1.0, data.get("time_elapsed", 0.0) / TIME_SCALE)),
            max(-1.0, min(1.0, progress)),
        ]

        if torch:
            state = torch.tensor(state_values, dtype=torch.float32, device=self.device)
        else:
            state = state_values
        return state, dist, progress

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

    def remember(self, s, a_dict, r, ns, done):
        if not torch:
            return
        a = (a_dict["move"], a_dict["strafe"], a_dict["yaw"], a_dict["pitch"], a_dict["jump"])
        self.memory.append((s, a, r, ns, done))

    def replay(self):
        if not torch or len(self.memory) < self.replay_min:
            return
        if self.global_step % self.update_every != 0:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([b[0] for b in batch]).to(self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([b[3] for b in batch]).to(self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)

        q_heads = self.model(states)
        q_move   = q_heads["move"].gather(1, actions[:, 0:1]).squeeze(1)
        q_strafe = q_heads["strafe"].gather(1, actions[:, 1:2]).squeeze(1)
        q_yaw    = q_heads["yaw"].gather(1, actions[:, 2:3]).squeeze(1)
        q_pitch  = q_heads["pitch"].gather(1, actions[:, 3:4]).squeeze(1)
        q_jump   = q_heads["jump"].gather(1, actions[:, 4:5]).squeeze(1)
        q_sum = q_move + q_strafe + q_yaw + q_pitch + q_jump

        with torch.no_grad():
            next_q_online = self.model(next_states)
            greedy = {
                "move":   torch.argmax(next_q_online["move"], dim=1),
                "strafe": torch.argmax(next_q_online["strafe"], dim=1),
                "yaw":    torch.argmax(next_q_online["yaw"], dim=1),
                "pitch":  torch.argmax(next_q_online["pitch"], dim=1),
                "jump":   torch.argmax(next_q_online["jump"], dim=1),
            }
            next_q_target = self.target_model(next_states)
            nq_move   = next_q_target["move"].gather(1, greedy["move"].unsqueeze(1)).squeeze(1)
            nq_strafe = next_q_target["strafe"].gather(1, greedy["strafe"].unsqueeze(1)).squeeze(1)
            nq_yaw    = next_q_target["yaw"].gather(1, greedy["yaw"].unsqueeze(1)).squeeze(1)
            nq_pitch  = next_q_target["pitch"].gather(1, greedy["pitch"].unsqueeze(1)).squeeze(1)
            nq_jump   = next_q_target["jump"].gather(1, greedy["jump"].unsqueeze(1)).squeeze(1)
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


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "activity_trainer_checkpoint.pth"
    eval_mode = len(sys.argv) > 2 and str(sys.argv[2]).lower() == "true"

    agent = Agent()
    if torch and agent.load(checkpoint_path):
        print(f"Loaded checkpoint from {checkpoint_path}", file=sys.stderr)
    else:
        print("No compatible checkpoint found, starting fresh.", file=sys.stderr)

    agent.eval_mode = bool(eval_mode)

    episode_rewards = []
    rolling_rewards = []
    episode_reward = 0.0
    episode_count = 0

    if plt:
        plt.ion()
        fig, ax = plt.subplots()
        plt.show(block=False)
    else:
        fig = ax = None

    q = Queue()

    def _reader():
        for line in sys.stdin:
            q.put(line)

    threading.Thread(target=_reader, daemon=True).start()

    while True:
        if plt:
            plt.pause(0.001)
        try:
            line = q.get(timeout=0.05)
        except Empty:
            continue
        if not line:
            break

        data = json.loads(line)
        if data.get("reset"):
            agent.reset_episode()
            episode_reward = 0.0

        state, dist, progress = agent.build_state(data)
        done = data.get("done", False)
        reached = data.get("reached", False)
        time_elapsed = data.get("time_elapsed", 0.0)

        reward = 3.0 * progress - 0.02
        if done:
            if reached:
                reward += 60.0
                reward += max(0.0, TIME_SCALE - time_elapsed) * 3.0
            else:
                reward -= 25.0

        if agent.last_state is not None and not agent.eval_mode and torch:
            agent.remember(agent.last_state, agent.last_action, reward, state, float(done))
            agent.replay()

        episode_reward += reward

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

        sys.stdout.write(json.dumps(action) + "\n")
        sys.stdout.flush()

        agent.global_step += 1

        if done:
            episode_rewards.append(episode_reward)
            window = episode_rewards[-25:]
            rolling_rewards.append(sum(window) / max(1, len(window)))

            if plt and fig is not None and ax is not None and (episode_count % max(1, PLOT_EVERY) == 0):
                ax.clear()
                ax.plot(episode_rewards, label="Episode Reward")
                ax.plot(rolling_rewards, label="Rolling Avg (25)")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Reward")
                ax.set_title("Activity Trainer Progress")
                ax.legend()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            episode_count += 1
            agent.reset_episode()
            if torch and (not agent.eval_mode) and episode_count % CHECKPOINT_INTERVAL == 0:
                agent.save(checkpoint_path)
            episode_reward = 0.0
        else:
            agent.last_state = state
            agent.last_action = action
            agent.last_dist = dist


def run():
    try:
        main()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
