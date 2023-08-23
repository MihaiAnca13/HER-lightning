from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch import optim

HID_SIZE = 256


class Actor(nn.Module):
    def __init__(self, obs_size, goal_size, act_size, action_bounds, offset):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            # state + goal
            nn.Linear(obs_size + goal_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, act_size)
        )

        self.action_bounds = nn.Parameter(action_bounds, requires_grad=False)
        self.offset = nn.Parameter(offset, requires_grad=False)

    def forward(self, state, goal, return_logits=False):
        o = self.net(torch.cat([state, goal], dim=1))
        mo = torch.tanh(o) * self.action_bounds + self.offset
        if not return_logits:
            return mo
        else:
            return mo, o


class Critic(nn.Module):
    """ double Q network """
    def __init__(self, obs_size, goal_size, act_size, H):
        super(Critic, self).__init__()

        self.q1 = nn.Sequential(
            nn.Linear(obs_size + goal_size + act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
            # nn.Sigmoid()
        )

        self.q2 = nn.Sequential(
            nn.Linear(obs_size + goal_size + act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
            # nn.Sigmoid()
        )

        self.H = nn.Parameter(torch.tensor(H, dtype=torch.float32, requires_grad=False), requires_grad=False)

    def forward(self, state, goal, action):
        sga = torch.cat([state, goal, action], dim=1)
        q1 = self.q1(sga)
        q2 = self.q2(sga)

        return q1, q2

    def Q1(self, state, goal, action):
        return self.q1(torch.cat([state, goal, action], dim=1))


class Agent:
    """ Handles action calls and noise """
    def __init__(self, net, action_clips, expl_noise, policy_noise, noise_clip, random_eps):
        self.net = net
        self.expl_noise = expl_noise
        self.max_action = abs(action_clips[1] - action_clips[0])
        self.policy_noise = policy_noise * self.max_action
        self.noise_clip = noise_clip * self.max_action
        self.action_clips = action_clips
        self.random_eps = random_eps

    def __call__(self, states, goals):
        mu_v = self.net(states, goals)
        actions = mu_v.data.detach().cpu().numpy()

        # completely randomise the values based on probability
        for i in range(len(actions)):
            if np.random.random() < self.random_eps:
                actions[i] = np.random.uniform(self.action_clips[0], self.action_clips[1])

        # add extra noise
        actions += np.random.normal(0, self.max_action * self.expl_noise, size=actions.shape)
        # keep action within boundary
        actions = np.clip(actions, self.action_clips[0], self.action_clips[1])
        return actions

    def add_train_noise(self, actions):
        """ adds noise to a whole batch of actions. specifically used in TD3 training """
        l = torch.FloatTensor(-self.noise_clip).to(actions.device)
        u = torch.FloatTensor(self.noise_clip).to(actions.device)
        policy_noise = torch.FloatTensor(self.policy_noise).to(actions.device)
        noise = torch.max(torch.min(torch.randn_like(actions) * policy_noise, u), l)
        l = torch.FloatTensor(self.action_clips[0]).to(actions.device)
        u = torch.FloatTensor(self.action_clips[1]).to(actions.device)
        return torch.max(torch.min(actions + noise, u), l)

    def test(self, states, goals):
        """ return action without noise """
        mu_v = self.net(states, goals)
        actions = mu_v.data.detach().cpu().numpy()
        actions = np.clip(actions, self.action_clips[0], self.action_clips[1])
        return actions


class TD3(nn.Module):
    def __init__(self, params, obs_size, goal_size, act_size, action_clips, action_bounds, action_offset, lr):
        super().__init__()
        self.actor = Actor(obs_size, goal_size, act_size, action_bounds, action_offset)
        self.critic = Critic(obs_size, goal_size, act_size, params.H)
        self.agent = Agent(self.actor, action_clips, params.expl_noise, params.noise_eps, params.noise_clip, params.random_eps)
        self.tgt_act_net = deepcopy(self.actor)
        self.tgt_crt_net = deepcopy(self.critic)
        self.act_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.crt_opt = optim.Adam(self.critic.parameters(), lr=lr)

    def alpha_sync(self, alpha):
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0

        for param, target_param in zip(self.critic.parameters(), self.tgt_crt_net.parameters()):
            target_param.data.copy_((1 - alpha) * param.data + alpha * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.tgt_act_net.parameters()):
            target_param.data.copy_((1 - alpha) * param.data + alpha * target_param.data)