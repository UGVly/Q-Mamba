import numpy as np
import pprint
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from random import randrange
from pathlib import Path
from numpy.lib.format import open_memmap
from torchtyping import TensorType
from beartype import beartype
from beartype.typing import Iterator, Tuple, Union, Optional
from tqdm import tqdm



import sys
sys.path.append('../')
from population.population import Population

# from pbo_env.agent import BaseEnvironment
import gym
from gym import spaces

rand_seed=42
reward_threshold=1e-7



pprint.pprint(vars(Population))
# pprint.pprint(Population)

# class DE_Environment(BaseEnvironment):
#      def init(self) -> Tuple[
#         Optional[str],
#         TensorType[float]
#     ]:
#         return 'please clean the kitchen', torch.randn(self.state_shape, device = self.device)

#     def forward(self, actions) -> Tuple[
#         TensorType[(), float],
#         TensorType[float],
#         TensorType[(), bool]
#     ]:
#         rewards = torch.randn((), device = self.device)
#         next_states = torch.randn(self.state_shape, device = self.device)
#         done = torch.zeros((), device = self.device, dtype = torch.bool)

#         return rewards, next_states, done
    
    
# inherit gym.Env for the convenience of parallelism
class DE(gym.Env):
    def __init__(self,dim,ps,max_x,min_x,max_fes):
        super(DE,self).__init__()
        
        self.dim = dim
        
        self.max_x=max_x
        self.min_x=min_x
        self.ps=ps

        self.max_fes=max_fes
        
        self.max_dist=np.sqrt((2*max_x)**2*dim)

        self.action_space=spaces.Box(low=0,high=1,shape=(self.ps,))
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(self.ps,9))
        self.name='DE'

    # def rand1(self, position, Fs):
    #     NP = position.shape[0]
    #     ps = self.ps
    #     r1 = np.random.randint(ps, size=NP)
    #     r2 = np.random.randint(ps,size=NP)
    #     r3 = np.random.randint(ps,size=NP)

    #     x1 = self.population.current_position[r1]
    #     x2 = self.population.current_position[r2]
    #     x3 = self.population.current_position[r3]
    #     trail = x3 + Fs * (x1 - x2) 

    #     return trail

    # def best2(self, position, Fs):
    #     NP = position.shape[0]
    #     ps = self.ps
    #     r1 = np.random.randint(ps, size=NP)
    #     r2 = np.random.randint(ps, size=NP)
    #     r3 = np.random.randint(ps, size=NP)
    #     r4 = np.random.randint(ps, size=NP)

    #     x1 = self.population.current_position[r1]
    #     x2 = self.population.current_position[r2]
    #     x3 = self.population.current_position[r3]
    #     x4 = self.population.current_position[r4]
    #     trail = self.population.gbest_position + Fs * (x1 - x2) + Fs * (x3 - x4)

    #     return trail

    
    # def current2rand(self, position, Fs):
    #     NP = position.shape[0]
    #     ps = self.ps
    #     r1 = np.random.randint(ps, size=NP)
    #     r2 = np.random.randint(ps,size=NP)
    #     r3 = np.random.randint(ps,size=NP)

    #     x1 = self.population.current_position[r1]
    #     x2 = self.population.current_position[r2]
    #     x3 = self.population.current_position[r3]
    #     trail = position + Fs * (x1 - position) + Fs * (x2 - x3)

    #     return trail
    
    # def current2best(self, position, F1,F2):
    #     NP = position.shape[0]
    #     ps = self.ps
    #     r1 = np.random.randint(ps, size=NP)
    #     r2 = np.random.randint(ps,size=NP)

    #     x1 = self.population.current_position[r1]
    #     x2 = self.population.current_position[r2]
    #     trail = position + F1 * (self.population.gbest_position - position) + F2 * (x1 - x2)

    #     return trail

    # def current2pbest(self, position,pbest, F1,F2):
    #     NP = position.shape[0]
    #     ps = self.ps
    #     r1 = np.random.randint(ps, size=NP)
    #     r2 = np.random.randint(ps,size=NP)

    #     x1 = self.population.current_position[r1]
    #     x2 = self.population.current_position[r2]
    #     trail = position + F1 * (pbest- position) + F2 * (x1 - x2)

    #     return trail
    
    # def get_random_pbest(self):
    #     p_rate= ( 0.4  -   1  ) * self.fes/self.max_fes + 1
    #     p_random_index = np.random.randint(0, int(np.ceil(self.ps*p_rate)), size=(self.ps))

    #     sorted_index=np.argsort(self.population.pbest_cost)
    #     sorted_pbest_pos=self.population.pbest_position[sorted_index]
    #     return sorted_pbest_pos[p_random_index]

    def mutate(self, position, Fs):
        NP = position.shape[0]
        ps = self.ps
        r1 = np.random.randint(ps, size=NP)
        r2 = np.random.randint(ps,size=NP)
        r3 = np.random.randint(ps,size=NP)

        x1 = self.population.current_position[r1]
        x2 = self.population.current_position[r2]
        x3 = self.population.current_position[r3]
        trail = x3 + Fs * (x1 - x2) 

        return trail

    def crossover(self, position, Cr):
        NP = position.shape[0]
        ps = self.ps
        r = np.random.rand(NP, self.dim)
        
        if position==self.population.gbest_position:
            return position
        
        trails = np.where(r<=Cr,position,self.population.gbest_position)
        
        while np.all(trails==self.population.gbest_position):
            # 选择一个原位置的随机个体
            r1 = np.random.randint(ps)
            trails[r1]=position[r1]
        
        return trails
    # the interface for environment reseting
    def reset(self):

        self.population=Population(self.dim,self.ps,self.min_x,self.max_x,self.max_fes,self.problem)
        self.population.reset()
    
    
        return self.population

    
    def step(self, action=None):
        if action.get('problem') is not None:
            self.problem=action['problem']
            return None,None,None,{}
        
        '''
            action is a dict like  
            {"F":0.5,"Cr":0.9}
            or {"F":0.5,"Cr":0.9,"fes":1000}
            or {"F":0.5,"Cr":0.9,"skip_step":10}
        '''
        
        F = action['F']
        Cr = action['Cr']

        # skip_step=action['skip_step']

        if action.get('skip_step') is not None:
            skip_step=action['skip_step']
        elif action.get('fes') is not None:
            step_fes=action['fes']
            next_fes=self.population.cur_fes+step_fes
        else:
            skip_step = 100


        step_end=False
        step=0

        
        # while self.population.cur_fes<next_fes and self.population.cur_fes<self.max_fes:
        while not step_end:
            old_pos=self.population.current_position
            
            trails = self.mutate(old_pos, F)
            
            trails = self.crossover(old_pos, Cr)
            
            # trails=self.current2best(old_pos,0.9,0.9)

            # trails=old_pos+F1*(rand_par_position-old_pos)+F2*(gbest_pos-old_pos)
            trails=np.clip(trails,self.min_x,self.max_x)
            
            # trails=np.where(r<=Cr[:,None],trails,old_pos)
            self.population.update(trails)

            step+=1
            if action.get('fes') is not None:
                if self.population.cur_fes>=next_fes or self.population.cur_fes>=self.max_fes:
                    step_end=True
                    break
            elif action.get('skip_step') is not None:
                if step>=skip_step:
                    step_end=True
                    break
                
        return self.population,0,self.population.cur_fes>=self.max_fes,{}
    