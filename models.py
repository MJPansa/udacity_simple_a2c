import torch as T
import torch.optim as optim
import torch.nn.modules as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden, lr, device):
        super(Actor, self).__init__()
        self.device = device
        self.input = nn.Linear(n_states, n_hidden)
        self.hidden_1 = nn.Linear(n_hidden, n_hidden)
        self.out_actor_sigma = nn.Linear(n_hidden, n_actions)
        self.out_actor_mu = nn.Linear(n_hidden, n_actions)

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        if not isinstance(x, T.Tensor):
            x = T.Tensor(x).to(self.device)
        x = x.float()
        x = F.relu(self.input(x))
        x = F.relu(self.hidden_1(x))

        mus = F.tanh(self.out_actor_mu(x))
        sigmas = F.softplus(self.out_actor_sigma(x))
        sigmas = T.clamp(sigmas, 5e-4, 2)

        return mus, sigmas


class Critic(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden, lr, device):
        super(Critic, self).__init__()
        self.device = device
        self.input = nn.Linear(n_states, n_hidden)
        self.hidden_1 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, n_actions)

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        if not isinstance(x, T.Tensor):
            x = T.Tensor(x).to(self.device)
        x = x.float()
        x = F.relu(self.input(x))
        x = F.relu(self.hidden_1(x))

        value = self.out(x)
        return value
