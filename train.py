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
        spawn_processes(pl_module.hparams, pl_module.replay_buffer, pl_module.model, pl_module.state_normalizer,
                        pl_module.goal_normalizer, pl_module.log_func)
        print("Finished spawning workers")


class HER(pl.LightningModule):
    def __init__(self, hparams):
        super(HER, self).__init__()

        self.hparams = hparams

        self.test_env = make_env(hparams, render=self.hparams.render_test)
        sample_obs = self.test_env.observation_space['observation'].sample()
        sample_goal = self.test_env.observation_space['achieved_goal'].sample()

        # HARD CODED VALUES FOR Bullet-HRL
        action_limits, state_limits = get_env_boundaries()
        action_offset, action_bounds, action_clip_low, action_clip_high = action_limits

        state_shape = sample_obs.shape[0]
        action_shape = self.test_env.action_space.shape[0]
        goal_shape = sample_goal.shape[0]
        self.action_clips = (action_clip_low, action_clip_high)

        self.model = DDPG(params=self.hparams, obs_size=state_shape, goal_size=goal_shape, act_size=action_shape,
                          action_clips=(action_clip_low, action_clip_high), action_bounds=action_bounds,
                          action_offset=action_offset)

        self.model.actor.share_memory()
        self.model.critic.share_memory()

        self.state_normalizer = Normalizer(state_shape, default_clip_range=self.hparams.clip_range)
        self.goal_normalizer = Normalizer(goal_shape, default_clip_range=self.hparams.clip_range)

        self.replay_buffer = SharedReplayBuffer(self.hparams.buffer_size, state_shape, action_shape, goal_shape)

    def log_func(self, d):
        self.log_dict(d, on_step=True, prog_bar=True)

    def collate_fn(self, batch):
        return collate.default_convert(batch)

    def __dataloader(self) -> DataLoader:
        dataset = RLDataset(self.replay_buffer, self.hparams.batch_size, self.hparams.n_batches,
                            self.hparams.replay_initial)
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
        testset = TestDataset(hparams=self.hparams, test_env=self.test_env, model=self.model,
                              state_normalizer=self.state_normalizer, goal_normalizer=self.goal_normalizer)
        testloader = DataLoader(
            dataset=testset,
            batch_size=1
        )

        return testloader

    def val_dataloader(self):
        return self.__testloader()

    def configure_optimizers(self):
        return [self.model.crt_opt, self.model.act_opt], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        states_v, actions_v, next_states_v, rewards_v, dones_mask, goals_v = batch[0]
        norm_states_v = self.state_normalizer.normalize(states_v)
        norm_goals_v = self.goal_normalizer.normalize(goals_v)
        if optimizer_idx == 0:
            norm_next_states_v = self.state_normalizer.normalize(next_states_v)
            # train critic
            q_v = self.model.critic(norm_states_v, norm_goals_v, actions_v)
            with torch.no_grad():
                next_act_v = self.model.tgt_act_net(
                    norm_next_states_v, norm_goals_v)
                q_next_v = self.model.tgt_crt_net(
                    norm_next_states_v, norm_goals_v, next_act_v)
                q_next_v[dones_mask] = 0.0
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_next_v * self.hparams.gamma
                # clip the q value
                clip_return = 1 / (1 - self.hparams.gamma)
                q_ref_v = torch.clamp(q_ref_v, -clip_return, 0)
            critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
            tqdm_dict = {
                'critic_loss': critic_loss_v
            }
            self.log_dict(tqdm_dict, prog_bar=True)

            return critic_loss_v

        elif optimizer_idx == 1:
            # train actor
            self.model.actor.offset.requires_grad = False
            self.model.actor.action_bounds.requires_grad = False

            cur_actions_v = self.model.actor(norm_states_v, norm_goals_v)
            actor_loss_v = -self.model.critic(norm_states_v, norm_goals_v, cur_actions_v).mean()
            actor_loss_v += ((cur_actions_v - self.model.actor.offset) / self.model.actor.action_bounds).pow(2).mean()
            tqdm_dict = {
                'actor_loss': actor_loss_v
            }
            self.log_dict(tqdm_dict, prog_bar=True)

            if batch_idx % self.hparams.sync_batches == 0:
                self.model.alpha_sync(self.hparams.polyak)

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
