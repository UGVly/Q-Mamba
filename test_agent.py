import torch
from torch.utils.data import Dataset

from options import get_options
from pbo_env.agent import Agent
from pbo_env.tranditional_DE import DE
from pbo_env.mocks import MockEnvironment, MockReplayDataset, MockReplayNStepDataset
from model.q_transformer import  QTransformer

def test_agent(opts):
    
    model = QTransformer(opts)
    env = MockEnvironment(state_shape = (10,))
    
    agent = Agent(
        q_model = model,
        environment = env,
        num_episodes = 10,
        max_num_steps_per_episode = 5,
       )

    agent()
    
    
    
    pass


if __name__ == '__main__':
    cfg = get_options()
    # print(cfg)
    test_agent(cfg)
    
    
    pass