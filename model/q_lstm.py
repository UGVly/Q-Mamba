import torch
import torch.nn as nn

class QLSTM(nn.Module):
    def __init__(self, cfg):
        super(QLSTM, self).__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
    def forward(self, feats, actions):
        
        
        q_values = self.q_head(feats, actions = actions)

        return q_values