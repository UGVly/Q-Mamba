import torch
from torch.utils.data import Dataset

from options import get_options
from pbo_env.agent import Agent, ReplayMemoryDataset
from pbo_env.traditional_DE import DE
from pbo_env.mocks import MockEnvironment
from pbo_env.reals import RealEnvironment
from model.q_transformer import  QTransformer

from execute.q_learner import QLearner

def test_agent(opts):
    
    model = QTransformer(opts)
    # env = MockEnvironment(state_shape = (9,))
    env = RealEnvironment(cfg = opts, state_shape = (9,))
    
    agent = Agent(
        q_model = model,
        environment = env,
        num_episodes = 10,
        max_num_steps_per_episode = 5,
       )

    agent()
    
    
    q_learner = QLearner(
    model,
    dataset = ReplayMemoryDataset(opts),
    num_train_steps = 20,
    learning_rate = 3e-4,
    batch_size = 4,
    grad_accum_every = 4,
    checkpoint_every = 10,
    )

    q_learner()
    
    state = torch.randn(9)
    action = model.get_optimal_actions(state)
    print(action)


if __name__ == '__main__':
    cfg = get_options()
    # print(cfg)
    test_agent(cfg)
    
    
    pass