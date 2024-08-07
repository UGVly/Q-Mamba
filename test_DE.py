from pbo_env.tranditional_DE import DE
# from dataset.cec_test_func import SphereFunction
from dataset.generate_dataset import sample_batch_task_cec21,get_train_set_cec21
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
    print((stu_pop.gbest_cost-stu_pop.problem.optimum)," ", (stu_pop.init_cost-stu_pop.problem.optimum))
    return r

def get_reward(prepopulation, population):
    r = pure_opt_reward(prepopulation, population)  + 0.5 * neg_opt_reward(population)
    print(f"reward: {r}")
    return r



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
    i = 2
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
    while not done:
        population, _, done, _ = env.step(action)
        print(f"Current FEs: {population.cur_fes}, Best Cost: {population.gbest_cost}")
        reward = get_reward(prepopulation, population)
        state = population.feature_encoding(opts.fea_mode)
        # print(state)
        prepopulation=population

if __name__ == "__main__":
    opts = get_options()
    test_de(opts)