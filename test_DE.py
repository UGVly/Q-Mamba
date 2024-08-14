from pbo_env.traditional_DE import DE
# from dataset.cec_test_func import SphereFunction
from dataset.generate_dataset import sample_batch_task_cec21, get_train_set_cec21
from dataset.cec_test_func import *
# train_set,train_pro_id=get_train_set_cec21(self.opts)


from options import get_options

def pure_opt_reward(pre_stu_pop,next_stu_pop):
    assert pre_stu_pop.init_cost>pre_stu_pop.problem.optimum, 'error: init cost == problem.optimum!!'
    r=(pre_stu_pop.gbest_cost-next_stu_pop.gbest_cost)/(pre_stu_pop.init_cost-pre_stu_pop.problem.optimum)
    return r

# neg optimization reward, used in train mode 4
def neg_opt_reward(stu_pop):
    assert stu_pop.init_cost>stu_pop.problem.optimum, 'error: init cost == problem.optimum!!'
    r=-(stu_pop.gbest_cost-stu_pop.problem.optimum)/(stu_pop.init_cost-stu_pop.problem.optimum)
    # print((stu_pop.gbest_cost-stu_pop.problem.optimum)," ", (stu_pop.init_cost-stu_pop.problem.optimum))
    return r

def base_reward6(stu_pop,tea_pop,s_gbest):
    r=-np.tanh(max(stu_pop.gbest_cost-s_gbest,1e-8)/max(tea_pop.gbest_cost-s_gbest,1e-8))
    return r

def base_reward7(stu_pop,tea_pop,s_gbest):
    r=(-np.tanh((stu_pop.gbest_cost-tea_pop.gbest_cost)/max(tea_pop.gbest_cost-s_gbest,1e-8))-1)/2
    return r

def base_reward8(stu_pop,tea_pop,s_gbest):
    r=-np.tanh((stu_pop.gbest_cost-s_gbest)/max(tea_pop.gbest_cost-s_gbest,1e-8))
    return r

# -tanh((t.init_cost-tg)/(t.init_cost-g)/(s.init_cost-sg)/(s.init_cost-g))
def base_reward9(stu_pop,tea_pop,s_gbest):
    r=-np.tanh(((tea_pop.init_cost-tea_pop.gbest_cost+1e-8)/(tea_pop.init_cost-s_gbest+1e-8))/((stu_pop.init_cost-stu_pop.gbest_cost+1e-8)/(stu_pop.init_cost-s_gbest+1e-8)))
    return r

# -tanh((t.init_cost-tg)/(t.init_cost-g)/(s.init_cost-sg)/(s.init_cost-g)-1)
def base_reward10(stu_pop,tea_pop,s_gbest):
    r=-np.tanh(((tea_pop.init_cost-tea_pop.gbest_cost+1e-8)/(tea_pop.init_cost-s_gbest+1e-8))/((stu_pop.init_cost-stu_pop.gbest_cost+1e-8)/(stu_pop.init_cost-s_gbest+1e-8))-1)
    return r
def get_reward(prepopulation, population,s_gbest):
    r = pure_opt_reward(prepopulation, population)  + 0.5 * neg_opt_reward(population)
    # print(f"reward: {base_reward6(prepopulation, population,s_gbest)}",end=' ')
    # print(f"reward: {base_reward7(prepopulation, population,s_gbest)}",end=' ')
    # print(f"reward: {base_reward8(prepopulation, population,s_gbest)}",end=' ')
    # print(f"reward: {base_reward9(prepopulation, population,s_gbest)}",end=' ')
    # print(f"reward: {base_reward10(prepopulation, population,s_gbest)}")
    print(f"pure_opt_reward: {pure_opt_reward(prepopulation, population)} reward: {r}")
    return r


# def get_surrogate_gbest_for_one(env,dataset,ids,seed,fes,F,Cr):
#     print('getting surrogate gbest...')
#     gbests={}
#     # set_seed(seed)
#     for id,pro in zip(ids,dataset):
#         action={'problem':copy.deepcopy(pro),'sgbest':0} 
#         env.step(action)
#         env.reset()
#         is_done=False
#         while not is_done:
#             action={'F': F, 'Cr':Cr, 'fes':fes} 
#             pop,_,is_done,_=env.step(action)
#             # is_done=is_done.all()
#         # gbest_list=[p.gbest_cost for p in pop]
#         gbests[id]= pop.gbest_cost # np.min(gbest_list)
#     print('done...')
#     return gbests


def test_de(opts):
    dim = opts.dim
    ps = opts.population_size
    min_x = opts.min_x
    max_x = opts.max_x
    max_fes = opts.max_fes
    
    

    env = DE(dim, ps, max_x, min_x, max_fes)
    # env.reset()
    train_set,train_pro_id=get_train_set_cec21(opts)
    # instances,ids=sample_batch_task_cec21(opts)
    # vector_env=SubprocVectorEnv if opts.is_linux else DummyVectorEnv
    
    # surrogate_gbest=get_surrogate_gbest_for_one(env,train_set,train_pro_id,seed=999,fes=opts.max_fes,F=0.5,Cr=0.9)
    
    i = 1
    # action={'problem':copy.deepcopy(instances[i])}
    action={'problem':copy.deepcopy(train_set[i])}
    env.step(action)
    # for p in instances:
    #     print(f'offset:{p.shift}, rotate:{p.rotate}')
    ori_population=env.reset()
    action = {'F': 0.5, 'Cr': 0.9, 'skip_step': opts.skip_step}
    # action = {'F': 0.5, 'Cr': 0.9, 'skip_step': 50}
    action = {'F': 0.5, 'Cr': 0.9, 'fes': 1000}
    done = False
    prepopulation=ori_population
    # s_gbest = surrogate_gbest[train_pro_id[i]]
    while not done:
        population, _, done, _ = env.step(action)
        # print(f"Current FEs: {population.cur_fes}, Best Cost: {population.gbest_cost}")
        reward = get_reward(prepopulation, population,s_gbest=0)
        state = population.feature_encoding(opts.fea_mode)
        # print(state)
        prepopulation=population





if __name__ == "__main__":
    opts = get_options()
    test_de(opts)