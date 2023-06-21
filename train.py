from multiprocessing import Manager, Lock

import numpy as np
import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from torch.utils.data._utils import collate

from arguments import get_args
from buffer import RLDataset, SharedReplayBuffer, TestDataset
from model import DDPG
from normalizer import Normalizer
from utils import make_env, get_env_boundaries
from worker import spawn_processes


class SpawnCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        spawn_processes(pl_module.hparams, pl_module.high_replay_buffer, pl_module.low_replay_buffer,
                        pl_module.high_model, pl_module.low_model, pl_module.high_state_normalizer,
                        pl_module.low_state_normalizer, pl_module.env_goal_normalizer, pl_module.log_result)
        print("Finished spawning workers")


class HER(pl.LightningModule):
    def __init__(self, hparams):
        super(HER, self).__init__()

        self.hparams = hparams

        self.test_env = make_env(hparams, render=self.hparams.render_test)
        sample_obs = self.test_env.observation_space['observation'].sample()
        sample_goal = self.test_env.observation_space['achieved_goal'].sample()

        action_limits, state_limits = get_env_boundaries()
        action_offset, action_bounds, action_clip_low, action_clip_high = action_limits
        state_offset, state_bounds, state_clip_low, state_clip_high = state_limits

        state_shape = sample_obs.shape[0]
        action_shape = self.test_env.action_space.shape[0]
        goal_shape = sample_goal.shape[0]

        self.hl_state_shape = state_shape
        self.ll_state_shape = action_shape

        self.high_model = DDPG(params=self.hparams, obs_size=state_shape, goal_size=goal_shape, act_size=action_shape,
                               action_clips=(state_clip_low, state_clip_high), action_bounds=state_bounds,
                               action_offset=state_offset)
        self.low_model = DDPG(params=self.hparams, obs_size=action_shape, goal_size=action_shape, act_size=action_shape,
                              action_clips=(action_clip_low, action_clip_high), action_bounds=action_bounds,
                              action_offset=action_offset)

        self.high_model.actor.share_memory()
        self.high_model.critic.share_memory()
        self.low_model.actor.share_memory()
        self.low_model.critic.share_memory()

        self.high_state_normalizer = Normalizer(state_shape, default_clip_range=self.hparams.clip_range)
        self.low_state_normalizer = Normalizer(action_shape, default_clip_range=self.hparams.clip_range)
        self.env_goal_normalizer = Normalizer(goal_shape, default_clip_range=self.hparams.clip_range)

        self.low_replay_buffer = SharedReplayBuffer(self.hparams.buffer_size, action_shape, action_shape, action_shape)
        self.high_replay_buffer = SharedReplayBuffer(self.hparams.buffer_size, state_shape, action_shape, goal_shape)

        m = Manager()
        self.lock = Lock()
        self.shared_log_list = m.list()

    def collate_fn(self, batch):
        return collate.default_convert(batch)

    def log_result(self, episode_result, accuracy):
        to_log = np.array([episode_result, np.array(accuracy[0]), np.array(accuracy[1])])
        with self.lock:
            self.shared_log_list.append(to_log)

    def __dataloader(self) -> DataLoader:
        dataset = RLDataset([self.low_replay_buffer, self.high_replay_buffer], self.hparams.batch_size,
                            self.hparams.n_batches, self.hparams.replay_initial)
        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=self.collate_fn,
            batch_size=1,
            num_workers=1,
            pin_memory=True
        )
        return dataloader

    def train_dataloader(self):
        return self.__dataloader()

    def __testloader(self):
        testset = TestDataset(hparams=self.hparams, test_env=self.test_env, high_model=self.high_model,
                              low_model=self.low_model, high_state_normalizer=self.high_state_normalizer,
                              low_state_normalizer=self.low_state_normalizer,
                              env_goal_normalizer=self.env_goal_normalizer)
        testloader = DataLoader(
            dataset=testset,
            batch_size=1
        )

        return testloader

    def val_dataloader(self):
        return self.__testloader()

    def configure_optimizers(self):
        return [self.high_model.crt_opt, self.high_model.act_opt, self.low_model.crt_opt, self.low_model.act_opt], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        if batch_idx % self.hparams.sync_batches == 0 and optimizer_idx == 0:
            self.high_model.alpha_sync(self.hparams.polyak)
            self.low_model.alpha_sync(self.hparams.polyak)

        # log this once per train step
        if optimizer_idx == 3:
            with self.lock:
                if len(self.shared_log_list) > 50:
                    log_list = np.array(self.shared_log_list)
                    mean_ep_reward = torch.tensor(np.array(log_list[:, 0], dtype=np.float32)).mean()
                    low_accuracy = torch.tensor(np.concatenate(log_list[:, 1]).flatten()).float().mean()
                    high_accuracy = torch.tensor(np.concatenate(log_list[:, 2]).flatten()).float().mean()
                    self.log_dict({'mean_ep_reward': mean_ep_reward}, prog_bar=True, on_step=True)
                    self.log_dict({'low_accuracy': low_accuracy}, prog_bar=True, on_step=True)
                    self.log_dict({'high_accuracy': high_accuracy}, prog_bar=True, on_step=True)
                    self.shared_log_list[:] = []

        states_v, actions_v, next_states_v, rewards_v, dones_mask, goals_v = batch[0]

        if states_v.shape[1] == self.hl_state_shape and optimizer_idx in [0, 1]:
            net = self.high_model
            state_normalizer = self.high_state_normalizer
            goal_normalizer = self.env_goal_normalizer
            level = 'high'
        elif states_v.shape[1] == self.ll_state_shape and optimizer_idx in [2, 3]:
            net = self.low_model
            state_normalizer = self.low_state_normalizer
            goal_normalizer = self.low_state_normalizer
            level = 'low'
        else:
            return

        norm_states_v = state_normalizer.normalize(states_v)
        norm_goals_v = goal_normalizer.normalize(goals_v)
        if optimizer_idx % 2 == 0:
            net.critic.H.requires_grad = False
            norm_next_states_v = state_normalizer.normalize(next_states_v)
            # train critic
            q_v = net.critic(norm_states_v, norm_goals_v, actions_v)
            with torch.no_grad():
                next_act_v = net.tgt_act_net(
                    norm_next_states_v, norm_goals_v)
                q_next_v = net.tgt_crt_net(
                    norm_next_states_v, norm_goals_v, next_act_v)
                q_next_v[dones_mask] = 0.0
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_next_v * self.hparams.gamma
            critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())

            tqdm_dict = {
                f'{level}_critic_loss': critic_loss_v
            }
            self.log_dict(tqdm_dict, prog_bar=True, on_step=True)

            return critic_loss_v

        else:
            # train actor
            net.actor.offset.requires_grad = False
            net.actor.action_bounds.requires_grad = False

            cur_actions_v = net.actor(norm_states_v, norm_goals_v)
            actor_loss_v = -net.critic(norm_states_v, norm_goals_v, cur_actions_v).mean()

            tqdm_dict = {
                f'{level}_actor_loss': actor_loss_v
            }
            self.log_dict(tqdm_dict, prog_bar=True, on_step=True)

            return actor_loss_v

    def validation_step(self, batch, batch_idx):
        to_log = dict()
        for k, v in batch.items():
            to_log[k] = v.detach().cpu().numpy()
        to_log['epoch_nr'] = int(self.current_epoch)
        if self.logger is not None:
            self.logger.experiment.log(to_log)


if __name__ == '__main__':
    mp.set_start_method('spawn')

    hparams = get_args()

    if hparams.debug:
        hparams.logger = None
        hparams.profiler = SimpleProfiler()
    else:
        hparams.logger = WandbLogger(project=hparams.project)

    seed_everything(hparams.seed)
    her = HER(hparams)
    trainer = pl.Trainer.from_argparse_args(hparams)
    trainer.callbacks.append(SpawnCallback())
    trainer.fit(her)
