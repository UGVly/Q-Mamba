import os
import time
import argparse
import torch

def options():
    parser = argparse.ArgumentParser(description = "Start q-mamba learning")

    # mode
    parser.add_argument('--mode', type = 'str', default = 'train', help = 'train or test')
    parser.add_argument('--device', type = 'str', default = 'cuda',help = 'cuda or cpu')


