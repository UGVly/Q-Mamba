import os
import time
import argparse
import torch

# return cfg
def get_options(args=None):
    parser = argparse.ArgumentParser(description = "Start q-mamba learning")

    # mode
    parser.add_argument('--mode', type = 'str', default = 'train', help = 'train or test')
    parser.add_argument('--device', type = 'str', default = 'cpu', choices = ["cuda", "gpu", "cpu"], help = 'cuda or cpu')
    parser.add_argument('--seed', type = 'int', default = 0, help = 'random seed')
    
    # data
    
    
    
    opts = parser.parse_args(args)
    
    opts.distributed = False
    
    return opts



