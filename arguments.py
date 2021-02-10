import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default="FetchBulletEnv", help='the environment name')
    parser.add_argument('--debug', type=bool, default=False, help='debugging mode')
    parser.add_argument('--project', type=str, default="her-pl", help='the project name')
    parser.add_argument('--gpus', type=int, default=1, help='the number of gpus to train on')
    parser.add_argument('--H', type=int, default=10, help='the number of low level steps per high level target')
    parser.add_argument('--max-epochs', type=int, default=35, help='the number of epochs to train the agent')
    parser.add_argument('--n-batches', type=int, default=2000, help='the times to update the network per epoch')
    parser.add_argument('--seed', type=int, default=100, help='random seed')
    parser.add_argument('--max-timesteps', type=int, default=50, help='max number of steps per episode')
    parser.add_argument('--np', type=int, default=4, help='the number of processes to collect samples')
    parser.add_argument('--distributed_backend', type=str, default='dp', help='the parallelisation method')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--noise-eps', type=float, default=0.01, help='Noise added to target policy during critic update')
    parser.add_argument('--random-eps', type=float, default=0.04, help='chance of completely random action')
    parser.add_argument('--noise-clip', type=float, default=0.05, help='Range to clip target policy noise')
    parser.add_argument('--expl-noise', type=float, default=0.01, help='Std of Gaussian exploration noise')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-initial', type=int, default=int(1e3), help='the required initial size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='number of HER transitions for the future method')
    parser.add_argument('--subgoal-testing', type=float, default=0.3, help='subgoal testing chance')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--lr-high', type=float, default=1e-3, help='the learning rate of the actor')
    parser.add_argument('--lr-low', type=float, default=1e-3, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.995, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=3, help='the number of tests')
    parser.add_argument('--val-check-interval', type=int, default=1000, help='frequency of tests (batches)')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range for the normalizer')
    parser.add_argument('--policy-freq', type=float, default=4, help='the freq at which to update the actor compared to critic')
    parser.add_argument('--action-l2-low', type=float, default=0.4, help='tanh saturation loss for low level')
    parser.add_argument('--action-l2-high', type=float, default=0.04, help='tanh saturation loss for high level')
    parser.add_argument('--render-test', type=bool, default=True, help='whether the test should be rendered or not')

    parser.add_argument('--num-sanity-val-steps', type=int, default=0, help='initial tests')

    args, _ = parser.parse_known_args()

    return args
