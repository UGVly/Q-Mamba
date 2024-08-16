import torch
from torch import Tensor

import sys
sys.path.append('..')
from pbo_env.traditional_DE import DE
# from dataset.cec_test_func import SphereFunction
from dataset.generate_dataset import sample_batch_task_cec21, get_train_set_cec21
from dataset.generate_dataset import sample_batch_task_id_cec21
from dataset.cec_test_func import *

from tqdm import tqdm

from einops import pack, unpack, repeat, reduce, rearrange

class Memory():
    def __init__(self) -> None:
        self.basline_cost=[]
        self.rendom_cost=[]
        self.model_cost=[]
    
    def clear(self):
        del self.basline_cost[:]
        del self.rendom_cost[:]
        del self.model_cost[:]
        
def neg_opt_reward(stu_pop):
    assert stu_pop.init_cost>stu_pop.problem.optimum, 'error: init cost == problem.optimum!!'
    r=-(stu_pop.gbest_cost-stu_pop.problem.optimum)/(stu_pop.init_cost-stu_pop.problem.optimum)
    # print((stu_pop.gbest_cost-stu_pop.problem.optimum)," ", (stu_pop.init_cost-stu_pop.problem.optimum))
    return r

def pure_opt_reward(pre_stu_pop,next_stu_pop):
    assert pre_stu_pop.init_cost>pre_stu_pop.problem.optimum, 'error: init cost == problem.optimum!!'
    r=(pre_stu_pop.gbest_cost-next_stu_pop.gbest_cost)/(pre_stu_pop.init_cost-pre_stu_pop.problem.optimum)
    return r

def get_reward(prepopulation, population,s_gbest=0):
    r = pure_opt_reward(prepopulation, population)  + 0.5 * neg_opt_reward(population)
    return r


def base_de(cfg,problems):
    dim = cfg.dim
    ps = cfg.population_size
    min_x = cfg.min_x
    max_x = cfg.max_x
    max_fes = cfg.max_fes
    
    best_costs=None
    for i in range(len(problems)):
        env = DE(dim, ps, max_x, min_x, max_fes)
        action={'problem':copy.deepcopy(problems[i])}
        env.step(action)
        # for p in instances:
        #     print(f'offset:{p.shift}, rotate:{p.rotate}')
        ori_population=env.reset()

        action = {'F': 0.5, 'Cr': 0.9, 'skip_step': 100}

        done = False
        prepopulation=ori_population
        best_cost=[ori_population.gbest_cost]
        while not done:
            population, _, done, _ = env.step(action)
            print(f"Current FEs: {population.cur_fes}, Best Cost: {population.gbest_cost}")
            best_cost.append(population.gbest_cost)
            reward = get_reward(prepopulation, population)
            state = population.feature_encoding(cfg.fea_mode)
            # print(state)
            prepopulation=population
        if best_costs is None:
            best_costs=np.array(best_cost)
        else:
            best_costs=np.add(best_costs,np.array(best_cost))
        return best_costs/len(problems)
def random_de(cfg,problems):
    dim = cfg.dim
    ps = cfg.population_size
    min_x = cfg.min_x
    max_x = cfg.max_x
    max_fes = cfg.max_fes
    
    best_costs=None
    for i in range(len(problems)):
        env = DE(dim, ps, max_x, min_x, max_fes)
        action={'problem':copy.deepcopy(problems[i])}
        env.step(action)
        # for p in instances:
        #     print(f'offset:{p.shift}, rotate:{p.rotate}')
        ori_population=env.reset()

        F = np.random.uniform(0, 2)
        Cr = np.random.uniform(0, 1)
        action = {'F': F, 'Cr': Cr, 'skip_step': 100}

        done = False
        prepopulation=ori_population
        best_cost=[ori_population.gbest_cost]
        while not done:
            population, _, done, _ = env.step(action)
            print(f"Current FEs: {population.cur_fes}, Best Cost: {population.gbest_cost}")
            best_cost.append(population.gbest_cost)
            reward = get_reward(prepopulation, population)
            state = population.feature_encoding(cfg.fea_mode)
            # print(state)
            prepopulation=population
        if best_costs is None:
            best_costs=np.array(best_cost)
        else:
            best_costs=np.add(best_costs,np.array(best_cost))
        return best_costs/len(problems)
def adaptive_de(cfg,model,problems):
    dim = cfg.dim
    ps = cfg.population_size
    min_x = cfg.min_x
    max_x = cfg.max_x
    max_fes = cfg.max_fes
    model.eval()
    best_costs=None
    for i in range(len(problems)):
        env = DE(dim, ps, max_x, min_x, max_fes)
        action={'problem':copy.deepcopy(problems[i])}
        env.step(action)
        # for p in instances:
        #     print(f'offset:{p.shift}, rotate:{p.rotate}')
        ori_population=env.reset()

        state = ori_population.feature_encoding(cfg.fea_mode)
        state = rearrange(state, 'd -> 1 d')
        state = torch.tensor(state, device = model.device, dtype = torch.float32)
        actions = model.get_optimal_actions(state, return_q_values=False)
        action = actions.cpu().numpy()[0]
        F = action[0]/255.0 * 2 # (0, 2)
        Cr = action[1]/255.0 # (0, 1)
        
        
        action = {'F': F, 'Cr': Cr, 'skip_step': 100}
        
        done = False
        prepopulation=ori_population
        best_cost=[ori_population.gbest_cost]
        while not done:
            population, _, done, _ = env.step(action)
            print(f"Current FEs: {population.cur_fes}, Best Cost: {population.gbest_cost}")
            best_cost.append(population.gbest_cost)
            reward = get_reward(prepopulation, population)
            state = population.feature_encoding(cfg.fea_mode)
            # print(state)
            prepopulation=population
        if best_costs is None:
            best_costs=np.array(best_cost)
        else:
            best_costs=np.add(best_costs,np.array(best_cost))
        return best_costs/len(problems)
    

def rollout(cfg, model, tb_logger, testing = True):
    """
    Rollout the model in the environment
    """
    
    
    if testing:
        # model.eval()
            
        test_id=range(1,11)
        base_de_costs_mean = None
        random_de_costs_mean = None
        adaptive_de_costs_mean = None
        with tqdm(range(len(test_id)),desc='rollout') as pbar:
            for bat_id,id in enumerate(test_id):
                # generate batch instances for testing
                instances,p_name=sample_batch_task_id_cec21(dim=cfg.dim,batch_size=cfg.batch_size,problem_id=id,seed=999)
                print("len(instances):",len(instances))
                print("p_name:",p_name)
                adaptive_de_costs = adaptive_de(cfg,model,instances)
                base_de_costs = base_de(cfg,instances)
                random_de_costs = random_de(cfg,instances)
                assert len(base_de_costs) == len(random_de_costs) == len(adaptive_de_costs) 
                print("len(base_de_costs):",len(base_de_costs))
                
                for i in range(len(base_de_costs)):
                    tb_logger.add_scalars(f'{p_name}', {'base_de':base_de_costs[i],
                                    'random_de':random_de_costs[i],
                                    'adaptive_de':adaptive_de_costs[i]}, i)
                
                if base_de_costs_mean is None:
                    base_de_costs_mean =np.log( base_de_costs)
                    random_de_costs_mean =np.log( random_de_costs)
                    adaptive_de_costs_mean = np.log(adaptive_de_costs)
                else:
                    base_de_costs_mean = np.add(base_de_costs_mean,np.log(base_de_costs))
                    random_de_costs_mean = np.add(random_de_costs_mean,np.log(random_de_costs))
                    adaptive_de_costs_mean = np.add(adaptive_de_costs_mean,np.log(adaptive_de_costs))

        for i in range(len(base_de_costs_mean)):
            tb_logger.add_scalars(f'overall', {'base_de':base_de_costs_mean[i],
                            'random_de':random_de_costs_mean[i],
                            'adaptive_de':adaptive_de_costs_mean[i]}, i)   
        
        
        tb_logger.add_text('Rollout', 'Testing')
        
        
if __name__ == "__main__":
    from tensorboardX import SummaryWriter
    cfg = None
    model = None
    tb_logger = SummaryWriter()
    from options import get_options
    cfg = get_options()
    rollout(cfg, model, tb_logger)
