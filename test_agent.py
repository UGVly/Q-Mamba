import torch
from torch.utils.data import Dataset

from options import get_options
from pbo_env.agent import Agent, ReplayMemoryDataset
from pbo_env.traditional_DE import DE
from pbo_env.mocks import MockEnvironment, MockReplayDataset, MockReplayNStepDataset
from model.q_transformer import  QTransformer

from execute.q_learner import QLearner

def test_agent(opts):
    
    model = QTransformer(opts)
    env = MockEnvironment(state_shape = (10,))
    
    # agent = Agent(
    #     q_model = model,
    #     environment = env,
    #     num_episodes = 10,
    #     max_num_steps_per_episode = 5,
    #    )

    # agent()
    
    
    q_learner = QLearner(
    model,
    dataset = ReplayMemoryDataset(opts),
    num_train_steps = 200,
    learning_rate = 3e-4,
    batch_size = 4,
    grad_accum_every = 4,
    )

    q_learner()
    
    pass


if __name__ == '__main__':
    cfg = get_options()
    # print(cfg)
    test_agent(cfg)
    
    
    pass