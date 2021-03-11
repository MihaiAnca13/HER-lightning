import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Normal
from copy import deepcopy


HID_SIZE = 256
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class Actor(nn.Module):
    def __init__(self, obs_size, goal_size, act_size, action_bounds, offset):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(obs_size + goal_size, HID_SIZE)
        self.l2 = nn.Linear(HID_SIZE, HID_SIZE)

        self.mean_l = nn.Linear(HID_SIZE, act_size)
        self.log_std_l = nn.Linear(HID_SIZE, act_size)

        self.action_bounds = nn.Parameter(action_bounds, requires_grad=False)
        self.offset = nn.Parameter(offset, requires_grad=False)

    def forward(self, state, goal):
        assert not torch.any(torch.isnan(state)) and not torch.any(torch.isnan(goal))

        obs = torch.cat([state, goal], dim=1)

        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        mean = self.mean_l(x)
        log_std = self.log_std_l(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # re-parametrization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_bounds + self.offset
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_bounds * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_bounds + self.offset
        return action, log_prob, mean


class Critic(nn.Module):
    def __init__(self, obs_size, goal_size, act_size):
        super(Critic, self).__init__()

        self.q1 = nn.Sequential(
            nn.Linear(obs_size + goal_size + act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(obs_size + goal_size + act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1)
        )

    def forward(self, state, goal, action):
        obs = torch.cat([state, goal, action], dim=1)
        return self.q1(obs), self.q2(obs)


class SAC(nn.Module):
    def __init__(self, params, obs_size, goal_size, act_size, action_bounds, action_offset):
        super().__init__()
        self.actor = Actor(obs_size, goal_size, act_size, action_bounds, action_offset)
        self.critic = Critic(obs_size, goal_size, act_size)
        self.tgt_crt_net = deepcopy(self.critic)
        self.alpha = 0.2
        self.target_entropy = -act_size
        self.log_alpha = nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad=True)
        self.act_opt = optim.Adam(self.actor.parameters(), lr=params.lr_actor)
        self.crt_opt = optim.Adam(self.critic.parameters(), lr=params.lr_critic)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=params.lr_alpha)

    def alpha_sync(self, alpha):
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0

        state = self.critic.state_dict()
        tgt_state = self.tgt_crt_net.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.tgt_crt_net.load_state_dict(tgt_state)