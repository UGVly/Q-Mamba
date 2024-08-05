import torch
import torch.nn as nn

from model.mamba_minimal import Mamba

class q_head(nn.Module):
    def __init__(self, cfg):
        super(q_head, self).__init__()
        self.cfg = cfg
        
    def forward(self, feats, actions):
        
        """
        einops
        b - batch
        n - number of actions
        a - action bins
        d - dimension
        """
        
        encoded_state = feats

        # this is the scheme many hierarchical transformer papers do

        sos_token = reduce(encoded_state, 'b ... d -> b 1 d', 'mean')

        tokens = self.maybe_append_actions(sos_token, actions = actions)

        embed = self.transformer(tokens, context = encoded_state)

        return self.get_q_values(embed)

class Q_Mamba(nn.module):
    def __init__(self, cfg):
        super(Q_Mamba, self).__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
    def forward(self, feats, actions):
        
        
        q_values = self.q_head(feats, actions = actions)

        return q_values
        
   
    
