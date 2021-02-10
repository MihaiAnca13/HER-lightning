import os

import numpy as np
import torch
import torch.multiprocessing as mp

from utils import make_env, Experience

LOW_STATE_IDX = [0, 1, 2, 9]


def spawn_processes(params, high_replay_buffer, low_replay_buffer, high_model, low_model, high_state_normalizer,
                    low_state_normalizer, env_goal_normalizer, log_result, active):
    # limit the number of threads started by OpenMP
    os.environ['OMP_NUM_THREADS'] = "1"

    data_proc_list = []
    for proc_idx in range(params.np):
        p_args = (
            proc_idx, params, high_replay_buffer, low_replay_buffer, high_model, low_model, high_state_normalizer,
            low_state_normalizer, env_goal_normalizer, log_result, active)
        data_proc = mp.Process(target=process_func, args=p_args)
        data_proc.start()
        data_proc_list.append(data_proc)


def process_func(proc_idx, params, high_replay_buffer, low_replay_buffer, high_model, low_model, high_state_normalizer,
                 low_state_normalizer, env_goal_normalizer, log_result, active):
    env = make_env(params, proc_idx)
    w = Worker(proc_idx, params, env, high_replay_buffer, low_replay_buffer, high_model, low_model,
               high_state_normalizer, low_state_normalizer, env_goal_normalizer, log_result, active)
    print(f"Spawning worker with id: {proc_idx}")
    w.loop()
    print(f"De-spawning worker with id: {proc_idx}")


class Worker:
    def __init__(self, wid, params, env, high_replay_buffer, low_replay_buffer, high_model, low_model,
                 high_state_normalizer, low_state_normalizer, env_goal_normalizer, log_result, active):
        self.wid = wid
        self.params = params
        self.env = env
        self.replay_buffers = [low_replay_buffer, high_replay_buffer]
        self.high_model = high_model
        self.low_model = low_model
        self.high_state_normalizer = high_state_normalizer
        self.low_state_normalizer = low_state_normalizer
        self.env_goal_normalizer = env_goal_normalizer
        self.log_result = log_result
        self.active = active

    def loop(self):
        assert self.params.H > 0

        device = next(self.high_model.actor.parameters()).device
        obs = self.env.reset()
        env_goal = torch.from_numpy(obs['desired_goal']).float().unsqueeze(0).to(device)
        norm_env_goal = self.env_goal_normalizer.normalize(env_goal)
        episode_high_transitions = []
        episode_low_transitions = []
        episode_reward = 0

        state_shape = self.env.observation_space['observation'].sample().shape[0]
        goal_shape = self.env.observation_space['achieved_goal'].sample().shape[0]
        new_states = np.zeros((self.params.max_timesteps, state_shape), dtype=np.float32)
        new_states[0] = obs['observation']
        new_env_goals = np.zeros((self.params.max_timesteps // self.params.H, goal_shape), dtype=np.float32)
        idx = 0

        accuracy = [[], []]
        goal_reached = False

        while self.active:
            new_env_goals[idx // self.params.H] = obs['achieved_goal']

            high_obs = obs.copy()
            high_state = torch.from_numpy(obs['observation']).float().unsqueeze(0).to(device)
            norm_high_state = self.high_state_normalizer.normalize(high_state)

            with torch.no_grad():
                norm_target_np = self.high_model.agent(norm_high_state, norm_env_goal)[0]
                norm_target = torch.from_numpy(norm_target_np).float().unsqueeze(0).to(device)
                target_np = self.low_state_normalizer.denormalize(norm_target).detach().cpu().numpy()[0]

            is_subgoal_test = False
            if np.random.uniform() < self.params.subgoal_testing:
                is_subgoal_test = True

            target_reached = False
            low_steps = 0
            for i in range(self.params.H):
                low_state = torch.from_numpy(obs['observation'][LOW_STATE_IDX]).float().unsqueeze(0).to(device)
                norm_low_state = self.low_state_normalizer.normalize(low_state)

                with torch.no_grad():
                    if not is_subgoal_test:
                        action = self.low_model.agent(norm_low_state, norm_target)[0]
                    else:
                        action = self.low_model.agent.test(norm_low_state, norm_target)[0]

                new_obs, reward, done, info = self.env.step(action)
                low_steps += 1
                episode_reward += reward
                new_states[idx] = new_obs['observation']
                idx += 1

                low_obs = {
                    'observation': obs['observation'][LOW_STATE_IDX],
                    'achieved_goal': obs['observation'][LOW_STATE_IDX],
                    'desired_goal': target_np,
                    'is_grasping': self.env.is_gripper_grasping()
                }
                new_low_obs = {
                    'observation': new_obs['observation'][LOW_STATE_IDX],
                    'achieved_goal': new_obs['observation'][LOW_STATE_IDX],
                    'desired_goal': target_np
                }

                low_level_thresholds = np.array([0.01, 0.01, 0.01, 0.01])
                # if target requires grasping
                if target_np[-1] < 0.04:
                    if self.env.is_gripper_grasping():
                        low_level_thresholds[-1] = 1

                r = self.env.compute_reward(achieved_goal=low_obs['achieved_goal'],
                                            desired_goal=low_obs['desired_goal'],
                                            info={'thresholds': low_level_thresholds})
                target_reached = (True if r == 0 else False) or target_reached

                obs = new_obs

                episode_low_transitions.append((low_obs, action, r, new_low_obs, self.params.gamma))
                # if not info['is_success']:
                #     episode_low_transitions.append((low_obs, action, r, new_low_obs, False))
                # else:
                #     episode_low_transitions.append((low_obs, action, 0, new_low_obs, False))
                #     break

                if done or info['is_success'] or target_reached:
                    break

            accuracy[0].append(1 if target_reached else 0)
            goal_reached = (True if info['is_success'] else False) or goal_reached

            if not target_reached:
                if is_subgoal_test:
                    exp = Experience(state=high_obs['observation'], action=norm_target_np.copy(), next_state=new_obs['observation'],
                                     reward=-1, gamma=np.power(self.params.gamma, low_steps), goal=high_obs['desired_goal'])
                    self.replay_buffers[1].append(exp)

                low_achieved_goal = torch.from_numpy(new_low_obs['achieved_goal']).float().unsqueeze(0).to(device)
                high_action = self.low_state_normalizer.normalize(low_achieved_goal)[0].detach().cpu().numpy()
            else:
                high_action = norm_target_np.copy()

            episode_high_transitions.append((high_obs, high_action, reward, new_obs, low_steps))

            if done:
                accuracy[1].append(1 if goal_reached else 0)

                self.log_result(episode_reward, accuracy)
                episode_reward = 0
                goal_reached = False
                accuracy = [[], []]

                self.high_state_normalizer.update(new_states)
                self.high_state_normalizer.recompute_stats()
                self.low_state_normalizer.update(new_states[:, LOW_STATE_IDX])
                self.low_state_normalizer.recompute_stats()
                self.env_goal_normalizer.update(new_env_goals)

                idx = 0

                self.create_her_transition(episode_high_transitions, 1)
                self.create_her_transition(episode_low_transitions, 0)
                episode_high_transitions = []
                episode_low_transitions = []

                obs = self.env.reset()
                goal = torch.from_numpy(obs['desired_goal']).float().unsqueeze(0).to(device)
                env_goals = torch.tensor([obs['desired_goal']] * (self.params.max_timesteps // self.params.H)).to(
                    device)

                self.env_goal_normalizer.update(env_goals)
                self.env_goal_normalizer.recompute_stats()

                norm_env_goal = self.env_goal_normalizer.normalize(goal)

    def create_her_transition(self, episode_transitions, level):
        episode_obs = np.array(episode_transitions)
        transitions = []
        for idx, (obs, action, reward, new_obs, steps) in enumerate(episode_obs):
            exp = Experience(state=obs['observation'], action=action, next_state=new_obs['observation'], reward=reward,
                             gamma=np.power(self.params.gamma, steps), goal=obs['desired_goal'])
            self.replay_buffers[level].append(exp)

            if self.params.replay_strategy == 'final':
                final_o = episode_obs[-1][0]

                if level == 0:
                    if final_o['achieved_goal'][-1] < 0.04 and obs['is_grasping']:
                        info = {'thresholds': np.array([0.01, 0.01, 0.01, 1])}
                    else:
                        info = {'thresholds': np.array([0.01, 0.01, 0.01, 0.01])}

                    new_gamma = self.params.gamma
                else:
                    info = None
                    new_gamma = np.power(self.params.gamma, min(self.params.H, np.sum(episode_obs[idx:-1, 4])))

                # check for rewarding random movements when the object hasn't moved
                # if high level is on goal, the reward would come from the env
                # if level == 1 and self.env.env._is_success(obs['achieved_goal'], new_obs['achieved_goal']) \
                #         and self.env.env._is_success(obs['achieved_goal'], final_o['achieved_goal']):
                #     continue

                new_reward = self.env.compute_reward(achieved_goal=new_obs['achieved_goal'],
                                                     desired_goal=final_o['achieved_goal'], info=info)

                new_exp = Experience(state=obs['observation'], action=action, next_state=new_obs['observation'],
                                     reward=new_reward, gamma=new_gamma, goal=final_o['achieved_goal'])

                transitions.append(new_exp)

            elif self.params.replay_strategy == 'future':
                if (episode_obs.shape[0] - idx - 1) > 0:
                    future_offset = np.unique(np.random.choice(range(episode_obs.shape[0] - idx - 1), self.params.replay_k))
                    future_idx = future_offset + idx + 1
                    future_idx = future_idx.astype(int)

                    for i, future_o in enumerate(episode_obs[future_idx][:, 0]):
                        if level == 0:
                            if future_o['achieved_goal'][-1] < 0.04 and obs['is_grasping']:
                                info = {'thresholds': np.array([0.01, 0.01, 0.01, 1])}
                            else:
                                info = {'thresholds': np.array([0.01, 0.01, 0.01, 0.01])}

                            new_gamma = self.params.gamma
                        else:
                            info = None
                            new_gamma = np.power(self.params.gamma, min(self.params.H, np.sum(episode_obs[idx:future_idx[i], 4])))

                        # check for rewarding random movements when the object hasn't moved
                        # if high level is on goal, the reward would come from the env
                        # if level == 1 and self.env.env._is_success(obs['achieved_goal'], new_obs['achieved_goal']) \
                        #         and self.env.env._is_success(obs['achieved_goal'], future_o['achieved_goal']):
                        #     continue

                        new_reward = self.env.compute_reward(achieved_goal=new_obs['achieved_goal'],
                                                             desired_goal=future_o['achieved_goal'], info=info)

                        # new_done = done
                        # if new_reward == 0 and level == 1:
                        #     new_done = True

                        new_exp = Experience(state=obs['observation'], action=action, next_state=new_obs['observation'],
                                             reward=new_reward, gamma=new_gamma, goal=future_o['achieved_goal'])
                        transitions.append(new_exp)

        if len(transitions) > 1:
            for t in transitions:
                self.replay_buffers[level].append(t)