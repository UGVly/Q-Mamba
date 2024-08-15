# -*- coding: utf-8 -*-?

import torch
import pprint
from options import get_options

from model.q_mamba import QMamba
from model.q_transformer import QTransformer
from model.q_lstm import QLSTM


from execute.q_learner import QLearner

from pbo_env.agent import Agent, ReplayMemoryDataset
from pbo_env.reals import RealEnvironment

from datetime import datetime

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
        
    cfg.replay_memories_folder = cfg.replay_memories_folder + '/' + cfg.model
         
        
    
    
    # Pretty print the run args
    pprint.pprint(vars(cfg))
    
    
    env = RealEnvironment(cfg = cfg, state_shape = (cfg.fea_dim,))
    if cfg.mode == 'train':
        
        # agent is a class that allows the q-model to interact with the environment to generate a replay memory dataset for learning
        
        agent = Agent(
            q_model = model,
            memories_dataset_folder = cfg.replay_memories_folder,
            environment = env,
            num_episodes = cfg.num_episodes,
            max_num_steps_per_episode = cfg.max_num_steps_per_episode,
            epsilon_start= cfg.epsilon_start,
            epsilon_end = cfg.epsilon_end,
            num_steps_to_target_epsilon = cfg.num_steps_to_target_epsilon,
           )

        agent()
        
        
        q_learner = QLearner(
        model,
        dataset = ReplayMemoryDataset(replay_memories_folder=cfg.replay_memories_folder),
        num_train_steps = cfg.num_train_steps,
        learning_rate = cfg.learning_rate,
        batch_size = cfg.batch_size,
        grad_accum_every = cfg.grad_accum_every,
        checkpoint_every = cfg.checkpoint_every,
        checkpoint_folder = cfg.checkpoint_folder
        )
        
        q_learner()
        
    #     path = r"/home/data3/ZhouJiang2/AAAAAAAAA/Gong/Q-Mamba/checkpoints/checkpoint-1.pt"
    #     q_learner.load(path)

    #     q_learner()
        
    #     state = torch.randn((1,9),device='cuda')
    #     action = model.get_optimal_actions(state)
    #     print(action)
    
    if cfg.mode == 'test':
        pass
    
    
    pass


    

if __name__ == '__main__':
    cfg = get_options()
    run(cfg)
