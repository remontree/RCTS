import rocket
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import os

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RocketEnv:
    def __init__(self):
        self.done_cnt = 0
        print("[   INFO] Rocket object generating...")
        self.rocket = rocket.Rocket(
            0.661, 0.954, 3.92, 0.38, 0, 0, 1.146174346, [-3, 2, -0.3], 0.001,
            0.26, 0.0018, "", "", 0.2617993333, 1.2, 0.018, "", "", 0.99,
            0.4, 0.28, 0.2, 0.02, 0.01,
            "/home/remon/문서/rocket simulator/real recent simul/rocket flight simulator/data/thrust.csv",
            "/home/remon/문서/rocket simulator/real recent simul/rocket flight simulator/data/pressure.csv",
            0, 0, 0, 0.052, 2
        )
        print("[   INFO] Rocket object generated successfully.")

    def get_reward(self):
        total_error = self.rocket.abs_integral_yaw + self.rocket.abs_integral_pitch
        if total_error == 0:
            return 0
        total_error = 1 / total_error
        total_error = (total_error * 1000) ** 2

        if self.rocket.kp < 0 or self.rocket.ki <0 or self.rocket.kd < 0:
            total_error = -5

        print("Total Reward: ", total_error)
        return total_error

    def reset(self):
        self.rocket.parameterUpdate(
            0.661, 0.954, 3.92, 0.38, 0, 0, 1.146174346, [-3, 2, -0.3], 0.001,
            0.26, 0.0018, "", "", 0.2617993333, 1.2, 0.018, "", "", 0.99,
            0.4, 0.28, 0.2, 0.02, 0.01,
            "/home/remon/문서/rocket simulator/real recent simul/rocket flight simulator/data/thrust.csv",
            "/home/remon/문서/rocket simulator/real recent simul/rocket flight simulator/data/pressure.csv",
            0, 0, 0, 0.052, 2
        )
        self.done_cnt = 0
        state = [
            self.rocket.mass, self.rocket.I_rocket,
            self.rocket.environment.wind[0], self.rocket.environment.wind[1],
            self.rocket.environment.wind[2], self.rocket.pitch_canard.Area,
            self.rocket.pitch_tail.Area, self.rocket.kp, self.rocket.ki,
            self.rocket.kd, self.rocket.abs_integral_yaw, self.rocket.abs_integral_pitch
        ]
        reward = 0
        done = False
        info = False
        
        return [state, reward, done, info]

    def step(self, num):
        self.done_cnt += 1
        coeff = [1, 0.5, 0.1, 0.05, 0.01, 1.5, 0.03, -0.1]
        kp = self.rocket.kp
        ki = self.rocket.ki
        kd = self.rocket.kd
        if num < 24:
            if num // 8 == 0:
                kp += coeff[num % 8]
            elif num // 8 == 1:
                ki += coeff[num % 8]
            elif num // 8 == 2:
                kd += coeff[num % 8]

            self.rocket.parameterUpdate(
                0.661, 0.954, 3.92, 0.38, 0, 0, 1.146174346, [-3, 2, -0.3], 0.001,
                0.26, 0.0018, "", "", 0.2617993333, 1.2, 0.018, "", "", 0.99,
                0.4, 0.28, 0.2, 0.02, 0.01,
                "/home/remon/문서/rocket simulator/real recent simul/rocket flight simulator/data/thrust.csv",
                "/home/remon/문서/rocket simulator/real recent simul/rocket flight simulator/data/pressure.csv",
                kp, ki, kd, 0.052, 2
            )

            print("     {0}th simulation is running... PID coefficients: {1}, {2}, {3}".format(
                self.done_cnt, self.rocket.kp, self.rocket.ki, self.rocket.kd))
            self.rocket.simulate()

            state = [
                self.rocket.mass, self.rocket.I_rocket,
                self.rocket.environment.wind[0], self.rocket.environment.wind[1],
                self.rocket.environment.wind[2], self.rocket.pitch_canard.Area,
                self.rocket.pitch_tail.Area, self.rocket.kp, self.rocket.ki,
                self.rocket.kd, self.rocket.abs_integral_yaw, self.rocket.abs_integral_pitch
            ]

            reward = self.get_reward()
            done = self.done_cnt >= 40
            return [state, reward, done, False]
        else:
            self.rocket.parameterUpdate(
                0.661, 0.954, 3.92, 0.38, 0, 0, 1.146174346, [-3, 2, -0.3], 0.001,
                0.26, 0.0018, "", "", 0.2617993333, 1.2, 0.018, "", "", 0.99,
                0.4, 0.28, 0.2, 0.02, 0.01,
                "/home/remon/문서/rocket simulator/real recent simul/rocket flight simulator/data/thrust.csv",
                "/home/remon/문서/rocket simulator/real recent simul/rocket flight simulator/data/pressure.csv",
                self.rocket.kp, self.rocket.ki, self.rocket.kd, 0.052, 2
            )

            print("[    INFO] {0}th simulation is running...".format(self.done_cnt))
            self.rocket.simulate()

            state = [
                self.rocket.mass, self.rocket.I_rocket,
                self.rocket.environment.wind[0], self.rocket.environment.wind[1],
                self.rocket.environment.wind[2], self.rocket.pitch_canard.Area,
                self.rocket.pitch_tail.Area, self.rocket.kp, self.rocket.ki,
                self.rocket.kd, self.rocket.abs_integral_yaw, self.rocket.abs_integral_pitch
            ]
            reward = self.get_reward()
            done = self.done_cnt >= 40
            return [state, reward, done, False]

EPISODES = 1000
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
GAMMA = 0.8
LR = 0.001
BATCH_SIZE = 10

class DQNAgent:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(12, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 25)
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), LR)
        self.steps_done = 0
        self.memory = deque(maxlen=10000)

    def memorize(self, state, action, reward, next_state):
        self.memory.append((state.to(device), action.to(device), torch.FloatTensor([reward]).to(device), torch.FloatTensor([next_state]).to(device)))

    def act(self, state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps_threshold:
            return self.model(state).data.max(1)[1].view(1, 1)
        else:
            return torch.LongTensor([[random.randrange(25)]]).to(device)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)

        current_q = self.model(states).gather(1, actions)
        max_next_q = self.model(next_states).detach().max(1)[0]
        expected_q = rewards + (GAMMA * max_next_q)

        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

env = RocketEnv()
agent = DQNAgent()
score_history = []

# Create a directory to save the model if it doesn't exist
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

for e in range(1, EPISODES + 1):
    print("{0}th Episode".format(e))
    state = env.reset()[0]
    steps = 0

    while True:
        state_tensor = torch.FloatTensor([state]).to(device)
        action = agent.act(state_tensor)
        next_state, reward, done, info = env.step(action.item())
        
        agent.memorize(state_tensor, action, reward, next_state)
        agent.learn()

        state = next_state

        if done:
            print("Episode:{0} Score:{1}".format(e, reward))
            score_history.append(reward)
            break

    # Save the model after each episode
    torch.save(agent.model.state_dict(), os.path.join(model_dir, f"model_episode_{e}.pth"))
