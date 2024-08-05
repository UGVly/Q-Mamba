# -*- coding: utf-8 -*-?

import torch
import pprint
from options import get_options

from model.q_mamba import QMamba
from model.q_transformer import QTransformer
from model.q_lstm import QLSTM


def run(cfg):
    
    if cfg.device == 'cuda' or cfg.device == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    
    if cfg.model == 'q-mamba':
        model = QMamba(cfg)
    elif cfg.model == 'q-transformer':
        model = QTransformer(cfg)
    elif cfg.model == 'q-lstm':
        model = QLSTM(cfg)
    else:
        raise ValueError('Model not found') 
        
    
    # Pretty print the run args
    pprint.pprint(vars(cfg))
    
    
    
    
    if cfg.mode == 'train':
        pass
    
    if cfg.mode == 'test':
        pass
    
    
    pass


    

if __name__ == '__main__':
    cfg = get_options()
    run(cfg)
