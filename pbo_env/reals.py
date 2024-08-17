from random import randrange

import torch
from torch.utils.data import Dataset

from beartype.typing import Tuple, Optional

from torchtyping import TensorType

from pbo_env.agent import BaseEnvironment
from pbo_env.traditional_DE import DE

from dataset.generate_dataset import sample_batch_task_cec21, get_train_set_cec21
from dataset.cec_test_func import *


# def test_de(opts):
#     dim = opts.dim
#     ps = opts.population_size
#     min_x = opts.min_x
#     max_x = opts.max_x
#     max_fes = opts.max_fes
    
    

#     env = DE(dim, ps, max_x, min_x, max_fes)
#     # env.reset()
#     train_set,train_pro_id=get_train_set_cec21(opts)
#     # instances,ids=sample_batch_task_cec21(opts)
#     # vector_env=SubprocVectorEnv if opts.is_linux else DummyVectorEnv
    
#     # surrogate_gbest=get_surrogate_gbest_for_one(env,train_set,train_pro_id,seed=999,fes=opts.max_fes,F=0.5,Cr=0.9)
    
#     i = 1
#     # action={'problem':copy.deepcopy(instances[i])}
#     action={'problem':copy.deepcopy(train_set[i])}
#     env.step(action)
#     # for p in instances:
#     #     print(f'offset:{p.shift}, rotate:{p.rotate}')
#     ori_population=env.reset()
#     action = {'F': 0.5, 'Cr': 0.9, 'skip_step': opts.skip_step}
#     # action = {'F': 0.5, 'Cr': 0.9, 'skip_step': 50}
#     action = {'F': 0.5, 'Cr': 0.9, 'fes': 1000}
#     done = False
#     prepopulation=ori_population
#     # s_gbest = surrogate_gbest[train_pro_id[i]]
#     while not done:
#         population, _, done, _ = env.step(action)
#         # print(f"Current FEs: {population.cur_fes}, Best Cost: {population.gbest_cost}")
#         reward = get_reward(prepopulation, population,s_gbest=0)
#         state = population.feature_encoding(opts.fea_mode)
#         # print(state)
#         prepopulation=population

    

class RealEnvironment(BaseEnvironment):
    def init(self) -> Tuple[
        TensorType[float]
    ]:
        opts = self.cfg
        dim = opts.dim
        ps = opts.population_size
        min_x = opts.min_x
        max_x = opts.max_x
        max_fes = opts.max_fes
        self.env = DE(dim, ps, max_x, min_x, max_fes)
        action={'problem':copy.deepcopy(self.get_random_problem(opts))}
        self.env.step(action)
        ori_population=self.env.reset()
        self.prepopulation = ori_population
        state = ori_population.feature_encoding(opts.fea_mode)
        state = torch.tensor(state, device = self.device, dtype = torch.float32)
        # print("state.shape: ",state.shape)
        return state

    def get_random_problem(self, opts):
        train_set,train_pro_id=get_train_set_cec21(opts)
        problem = train_set[randrange(len(train_set))]
        return problem
    
    def pure_opt_reward(self, pre_stu_pop,next_stu_pop):
        assert pre_stu_pop.init_cost>pre_stu_pop.problem.optimum, 'error: init cost == problem.optimum!!'
        r=(pre_stu_pop.gbest_cost-next_stu_pop.gbest_cost)/(pre_stu_pop.init_cost-pre_stu_pop.problem.optimum)
        return r

    def neg_opt_reward(self, stu_pop):
        assert stu_pop.init_cost>stu_pop.problem.optimum, 'error: init cost == problem.optimum!!'
        r=-(stu_pop.gbest_cost-stu_pop.problem.optimum)/(stu_pop.init_cost-stu_pop.problem.optimum)
        # print((stu_pop.gbest_cost-stu_pop.problem.optimum)," ", (stu_pop.init_cost-stu_pop.problem.optimum))
        return r
    def get_reward(self, prepopulation, population):
        r = self.pure_opt_reward(prepopulation, population)  + 0.5 * self.neg_opt_reward(population)
        return r
    def forward(self, actions) -> Tuple[
        TensorType[(), float],
        TensorType[float],
        TensorType[(), bool]
    ]:
        action = actions.cpu().numpy()[0]
        F = action[0]/255.0 * 2 # (0, 2)
        Cr = action[1]/255.0 # (0, 1)
        action = {'F': F, 'Cr': Cr, 'skip_step': 50}
        # print("##action: ", action)
        
        population, _, done, _ = self.env.step(action)
        rewards = self.get_reward(self.prepopulation, population)
        # print("rewards:",rewards)
        rewards = torch.tensor(rewards, device = self.device)
        next_states = population.feature_encoding(self.cfg.fea_mode)
        next_states = torch.tensor(next_states, device = self.device, dtype = torch.float32)
        # rewards = torch.randn((), device = self.device)
        # next_states = torch.randn(self.state_shape, device = self.device)
        done = torch.tensor(done, device = self.device, dtype = torch.bool)
        self.prepopulation = population
        # print("next_states", next_states.shape)
        return rewards, next_states, done