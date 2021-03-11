import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default="FetchBulletEnv", help='the environment name')
    parser.add_argument('--debug', type=bool, default=False, help='debugging mode')
    parser.add_argument('--project', type=str, default="her-pl", help='the project name')
    parser.add_argument('--gpus', type=int, default=1, help='the number of gpus to train on')
    parser.add_argument('--max-epochs', type=int, default=50, help='the number of epochs to train the agent')
    parser.add_argument('--n-batches', type=int, default=2000, help='the times to update the network per epoch')
    parser.add_argument('--sync-batches', type=int, default=40, help='freq (batches) at which to update the target networks')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--max-timesteps', type=int, default=50, help='max number of steps per episode')
    parser.add_argument('--np', type=int, default=4, help='the number of processes to collect samples')
    parser.add_argument('--distributed_backend', type=str, default='dp', help='the parallelisation method')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-initial', type=int, default=int(1e3), help='the required initial size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--lr-alpha', type=float, default=0.0001, help='the learning rate of the temperature')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=2, help='the number of tests')
    parser.add_argument('--val-check-interval', type=int, default=1000, help='frequency of tests (batches)')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--render-test', type=bool, default=True, help='whether the test should be rendered or not')

    parser.add_argument('--num-sanity-val-steps', type=int, default=0, help='initial tests')

    args, _ = parser.parse_known_args()

    return args
