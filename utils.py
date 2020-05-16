from unityagents import UnityEnvironment
from collections import deque
import torch as T
import numpy as np



class EnvWrapper:
    def __init__(self, path, no_graphics=True):
        self.env = env = UnityEnvironment(file_name=path, no_graphics=no_graphics)
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]
        self.env_info = None
        self.reset()

        self.action_space = self.brain.vector_action_space_size
        self.observation_space = self.env_info.vector_observations.shape[1]

    def step(self, actions):
        self.env_info = self.env.step(actions)[self.brain_name]
        next_state = self.env_info.vector_observations
        reward = self.env_info.rewards
        done = self.env_info.local_done
        return next_state, reward, done, None

    def reset(self):
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        return self.env_info.vector_observations


class ExperienceBuffer:
    def __init__(self, size, device):
        self.device = device
        self.states = deque(maxlen=size)
        self.actions = deque(maxlen=size)
        self.rewards = deque(maxlen=size)

    def __len__(self):
        return len(self.states)

    def add(self, states, actions, rewards):
        self.states.append(states)
        self.actions.append(actions)
        self.rewards.append(rewards)

    def draw(self):
        return [T.from_numpy(np.stack(x)).float().squeeze().unsqueeze(0).to(self.device) for x in
                [self.states, self.actions, self.rewards]]

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()



def calculate_G(rewards, values, done, gamma):
    rewards = rewards.cpu().numpy().reshape(-1)
    rewards = np.flip(rewards)
    G = np.zeros(len(rewards))
    if done:
        vn = 0
    else:
        vn = values[-1].detach().cpu().item()
    for i in range(len(rewards)):
        vn = rewards[i] + gamma * vn
        G[i] = vn
    return T.Tensor(G).float()
