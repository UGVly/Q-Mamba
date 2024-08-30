import os
import time
import argparse
import torch

# return cfg


def get_options(args=None):
    parser = argparse.ArgumentParser(description="Start q-mamba learning")

    # mode
    parser.add_argument('--mode', type=str,
                        default='train', help='train or test')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=["cuda", "gpu", "cpu"], help='cuda or cpu')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--resume', default=None,
                        help='resume training from previous checkpoint file')

    # model
    parser.add_argument('--model', type=str, default='q-mamba',
                        choices=["q-mamba", "q-transformer", "q-lstm"], help='model name')
    parser.add_argument('--num_actions', type=int,
                        default=2, help='number of actions')
    parser.add_argument('--action_bins', type=int,
                        default=8, help='number of action bins')

    # environment settings
    parser.add_argument('--population_size', type=int, default=100,
                        help='population size use in backbone algorithm')  # recommend 100
    parser.add_argument('--dim', type=int, default=9,
                        help='dimension of the sovling problems')
    parser.add_argument('--max_x', type=float, default=100,
                        help='the upper bound of the searching range')
    parser.add_argument('--min_x', type=float, default=-100,
                        help='the lower bound of the searching range')
    parser.add_argument('--boarder_method', default='clipping', choices=[
                        'clipping', 'random', 'periodic', 'reflect'], help='boarding methods')
    parser.add_argument('--skip_step', default=5, type=int,
                        help='apply the update function every skip_step step of updating')
    parser.add_argument('--max_fes', type=int, default=50000,
                        help='max function evaluation times')
    parser.add_argument('--reward_func', default='gap_near',
                        choices=['w', 'gap_near'], help='several dist functions for comparison')
    parser.add_argument('--b_reward_func', default='5', choices=[
                        '1', '2', '3', '4', '2div2', '5', '6', '7', '8', '9', '10'], help='different baseline reward selections')
    parser.add_argument('--fea_mode', default='full', choices=[
                        'full', 'no_fit', 'no_dis', 'no_opt', 'only_dis', 'only_opt', 'only_fit', 'xy'], help='feature selection')
    parser.add_argument('--fea_dim', type=int, default=9,
                        help='dim of feature encoding( excluding those historical information)')
    parser.add_argument("--problem", type=str, default="Schwefel", choices=['Bent_cigar', 'Schwefel', 'bi_Rastrigin', 'Grie_rosen',
                        'Hybrid1', 'Hybrid2', 'Hybrid3', 'Composition1', 'Composition2', 'Composition3'], help="Question to ask on")

    # save
    parser.add_argument('--no_tb', action='store_true',
                        help='disable Tensorboard logging')

    # agent
    parser.add_argument('--num_episodes', type=int,
                        default=10000, help='number of episodes')
    parser.add_argument('--max_num_steps_per_episode', type=int,
                        default=100, help='max number of steps per episode')
    parser.add_argument('--epsilon_start', type=float,
                        default=0.25, help='start epsilon')
    parser.add_argument('--epsilon_end', type=float,
                        default=0.001, help='end epsilon')
    parser.add_argument('--num_steps_to_target_epsilon', type=int,
                        default=1000, help='number of steps to target epsilon')
    parser.add_argument('--replay_memories_folder', type=str,
                        default='./replay_memories_data_debug', help='folder to save replay memories')
    parser.add_argument('--create_replay_memories',default=None , action='store_true', help='if create replay memories')
    
    # q-learning training parameters
    parser.add_argument('--num_train_steps', type=int,
                        default=1000, help='number of training steps')
    parser.add_argument('--learning_rate', type=float,
                        default=3e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--grad_accum_every', type=int,
                        default=4, help='gradient accumulation every n steps')
    parser.add_argument('--checkpoint_every', type=int,
                        default=100, help='save checkpoint every n steps')
    parser.add_argument('--checkpoint_folder', type=str,
                        default='./checkpoints', help='folder to save checkpoints')

    parser.add_argument('--log_dir', type=str,
                        default='./logs', help='folder to save logs')

    opts = parser.parse_args(args)

    opts.distributed = False

    return opts
