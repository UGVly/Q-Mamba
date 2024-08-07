import os
import time
import argparse
import torch

# return cfg
def get_options(args=None):
    parser = argparse.ArgumentParser(description = "Start q-mamba learning")

    # mode
    parser.add_argument('--mode', type = str, default = 'train', help = 'train or test')
    parser.add_argument('--device', type = str, default = 'cpu', choices = ["cuda", "gpu", "cpu"], help = 'cuda or cpu')
    parser.add_argument('--seed', type = int, default = 0, help = 'random seed')
    
    # model
    parser.add_argument('--model', type = str, default = 'q-transformer', choices = ["q-mamba", "q-transformer", "q-lstm"], help = 'model name')
    
    # save
    parser.add_argument('--checkpoint_folder', type = str, default = './checkpoints', help = 'folder to save checkpoints')
    parser.add_argument('--checkpoint_every', type = int, default = 1000, help = 'save checkpoint every n steps')
    
    # q-learning training parameters
    parser.add_argument('--num_train_steps', type = int, default = 10000, help = 'number of training steps')
    parser.add_argument('--learning_rate', type = float, default = 3e-4, help = 'learning rate')
    parser.add_argument('--batch_size', type = int, default = 4, help = 'batch size')
    parser.add_argument('--grad_accum_every', type = int, default = 4, help = 'gradient accumulation every n steps')
    
    parser.add_argument('--replay_memories_folder',type = str, default = './replay_memories_data_default', help = 'folder to save replay memories')
    parser.add_argument('--log_dir', type = str, default = './logs', help = 'folder to save logs')
    
    opts = parser.parse_args(args)
    
    opts.distributed = False
    
    return opts



