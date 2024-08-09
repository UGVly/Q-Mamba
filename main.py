# -*- coding: utf-8 -*-?

import torch
import pprint
from options import get_options

from model.q_mamba import QMamba
from model.q_transformer import QTransformer
from model.q_lstm import QLSTM

from pbo_env.agent import Agent
from execute.q_learner import QLearner

from pbo_env.env import Environment
def run(cfg):
    
    if cfg.device == 'cuda' or cfg.device == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    
    if cfg.model == 'q-mamba':
        model = QMamba(cfg)
    elif cfg.model == 'q-transformer':
        model = QTransformer(cfg)
    elif cfg.model == 'q-lstm':
        model = QLSTM(cfg)
    else:
        raise ValueError('Model not found') 
        
    
    # Pretty print the run args
    pprint.pprint(vars(cfg))
    
    
    env = Environment()
    if cfg.mode == 'train':
        
        # agent is a class that allows the q-model to interact with the environment to generate a replay memory dataset for learning
        
        agent = Agent(
        model,
        environment = env,
        num_episodes = 10,
        max_num_steps_per_episode = 5,
       )

        agent()
        
        
        
        q_learner = QLearner(
        model,
        dataset = ReplayMemoryDataset(),
        num_train_steps = cfg.num_train_steps,
        learning_rate = cfg.learning_rate,
        batch_size = cfg.batch_size,
        grad_accum_every = cfg.grad_accum_every,
        checkpoint_folder = cfg.checkpoint_folder,
        checkpoint_every = cfg.checkpoint_every,
        )

        q_learner()
        pass
    
    if cfg.mode == 'test':
        pass
    
    
    pass


    

if __name__ == '__main__':
    cfg = get_options()
    run(cfg)
