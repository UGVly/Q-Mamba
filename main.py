# -*- coding: utf-8 -*-?

import torch
import pprint
from options import get_options

from model.q_mamba import QMamba
from model.q_transformer import QTransformer
from model.q_lstm import QLSTM


from execute.q_learner import QLearner
from execute.rollout import rollout

from pbo_env.agent import Agent, ReplayMemoryDataset
from pbo_env.reals import RealEnvironment

from datetime import datetime

from tensorboardX import SummaryWriter
import os

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
        
    if cfg.mode == 'train' and cfg.resume == None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.checkpoint_folder = cfg.checkpoint_folder + '/' + cfg.model + '/' + current_time
        
    if not cfg.no_tb:
        tb_logger = SummaryWriter(os.path.join(cfg.log_dir, "{}D".format(cfg.dim), cfg.model))
    # cfg.replay_memories_folder = cfg.replay_memories_folder + '/' + cfg.model
         
    
    # Pretty print the run args
    pprint.pprint(vars(cfg))
    
    
    env = RealEnvironment(cfg = cfg, state_shape = (cfg.fea_dim,))
    if cfg.mode == 'train':
        
        # agent is a class that allows the q-model to interact with the environment to generate a replay memory dataset for learning
        
        # agent = Agent(
        #     q_model = model,
        #     memories_dataset_folder = cfg.replay_memories_folder,
        #     environment = env,
        #     num_episodes = cfg.num_episodes,
        #     max_num_steps_per_episode = cfg.max_num_steps_per_episode,
        #     epsilon_start= cfg.epsilon_start,
        #     epsilon_end = cfg.epsilon_end,
        #     num_steps_to_target_epsilon = cfg.num_steps_to_target_epsilon,
        #    )

        # agent()
        
        cfg.replay_memories_folder=r"/home/data3/ZhouJiang2/AAAAAAAAA/Gong/Q-Mamba/replay_memories_data_debug/q-transformer-"
        
        q_learner = QLearner(
        model,
        dataset = ReplayMemoryDataset(replay_memories_folder=cfg.replay_memories_folder),
        num_train_steps = cfg.num_train_steps,
        learning_rate = cfg.learning_rate,
        batch_size = cfg.batch_size,
        grad_accum_every = cfg.grad_accum_every,
        checkpoint_every = cfg.checkpoint_every,
        checkpoint_folder = cfg.checkpoint_folder,
        tb_logger = tb_logger
        )
        
        q_learner()
        
        # path = r"/home/data3/ZhouJiang2/AAAAAAAAA/Gong/Q-Mamba/checkpoints/q-transformer/20240816_101332/checkpoint-10.pt"
        # q_learner.load(path)

    #     q_learner()

    
    if cfg.mode == 'test' and cfg.resume == None:
        raise ValueError('Test mode requires a checkpoint to load')
    elif cfg.mode == 'test' and cfg.resume != None:
        
        cfg.replay_memories_folder=r"/home/data3/ZhouJiang2/AAAAAAAAA/Gong/Q-Mamba/replay_memories_data_debug/q-transformer-"
        
        q_learner = QLearner(
        model,
        dataset = ReplayMemoryDataset(replay_memories_folder=cfg.replay_memories_folder),
        num_train_steps = cfg.num_train_steps,
        learning_rate = cfg.learning_rate,
        batch_size = cfg.batch_size,
        grad_accum_every = cfg.grad_accum_every,
        checkpoint_every = cfg.checkpoint_every,
        checkpoint_folder = cfg.checkpoint_folder,
        tb_logger = tb_logger
        )
        
        # q_learner()
        
        path = r"/home/data3/ZhouJiang2/AAAAAAAAA/Gong/Q-Mamba/checkpoints/q-transformer/20240816_101332/checkpoint-10.pt"
        q_learner.load(path)
        
        rollout(cfg, model, tb_logger, testing = True)

    
    if not cfg.no_tb:
        tb_logger.close()
    
    pass


    

if __name__ == '__main__':
    cfg = get_options()
    run(cfg)
