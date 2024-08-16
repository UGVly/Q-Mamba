from random import random
from functools import partial, cache

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList

from beartype import beartype
from beartype.typing import Union, List, Optional, Callable, Tuple, Dict, Any

from einops import pack, unpack, repeat, reduce, rearrange
from einops.layers.torch import Rearrange, Reduce


import sys
sys.path.append('../')
from model.Attend import Attend

# from model.mamba_minimal import Mamba
from mamba_ssm import Mamba

# from classifier_free_guidance_pytorch import (
#     TextConditioner,
#     AttentionTextConditioner,
#     NullConditioner,
#     classifier_free_guidance
# )

# helpers

def exists(val):
    return val is not None

def xnor(x, y):
    """ (True, True) or (False, False) -> True """
    return not (x ^ y)

def divisible_by(num, den):
    return (num % den) == 0

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# tensor helpers

def l2norm(t, dim = -1):
    return F.normalize(t, dim = dim)

def pack_one(x, pattern):
    return pack([x], pattern)

def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]



# channel rmsnorm

class RMSNorm(Module):
    def __init__(self, dim, affine = True):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if affine else 1.

    def forward(self, x):
        return l2norm(x) * self.gamma * self.scale



# Dueling heads for Q value

class DuelingHead(Module):
    def __init__(
        self,
        dim,
        expansion_factor = 2,
        action_bins = 256
    ):
        super().__init__()
        dim_hidden = dim * expansion_factor

        self.stem = nn.Sequential(
            nn.Linear(dim, dim_hidden),
            nn.SiLU()
        )

        self.to_values = nn.Sequential(
            nn.Linear(dim_hidden, 1)
        )

        self.to_advantages = nn.Sequential(
            nn.Linear(dim_hidden, action_bins)
        )

    def forward(self, x):
        x = self.stem(x)

        advantages = self.to_advantages(x)
        advantages = advantages - reduce(advantages, '... a -> ... 1', 'mean')

        values = self.to_values(x)

        q_values = values + advantages
        return q_values.sigmoid()

# Q head modules, for either single or multiple actions

class QHeadSingleAction(Module):
    def __init__(
        self,
        dim,
        *,
        num_learned_tokens = 8,
        action_bins = 256,
        dueling = False
    ):
        super().__init__()
        self.action_bins = action_bins

        if dueling:
            self.to_q_values = nn.Sequential(
                Reduce('b (f n) d -> b d', 'mean', n = num_learned_tokens),
                DuelingHead(
                    dim,
                    action_bins = action_bins
                )
            )
        else:
            self.to_q_values = nn.Sequential(
                Reduce('b (f n) d -> b d', 'mean', n = num_learned_tokens),
                RMSNorm(dim),
                nn.Linear(dim, action_bins),
                nn.Sigmoid()
            )

    def get_random_actions(self, batch_size):
        return torch.randint(0, self.action_bins, (batch_size,), device = self.device)

    def get_optimal_actions(
        self,
        encoded_state,
        return_q_values = False,
        actions = None,
        **kwargs
    ):
        assert not exists(actions), 'single actions will never receive previous actions'

        q_values = self.forward(encoded_state)

        max_q, action_indices = q_values.max(dim = -1)

        if not return_q_values:
            return action_indices

        return action_indices, max_q

    def forward(self, encoded_state):
        return self.to_q_values(encoded_state)

class QHeadMultipleActions(Module):
    def __init__(
        self,
        dim,
        *,
        num_actions = 8,
        action_bins = 256,
        attn_depth = 2,
        attn_dim_head = 32,
        attn_heads = 8,
        dueling = False,
        weight_tie_action_bin_embed = False
    ):
        super().__init__()
        self.num_actions = num_actions
        self.action_bins = action_bins

        self.action_bin_embeddings = nn.Parameter(torch.zeros(num_actions, action_bins, dim))
        nn.init.normal_(self.action_bin_embeddings, std = 0.02)

        self.to_q_values = None
        if not weight_tie_action_bin_embed:
            self.to_q_values = nn.Linear(dim, action_bins)

        # TODO: add mamba
        self.mamba = model = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        ).to("cuda")
        
        # self.transformer = Transformer(
        #     dim = dim,
        #     depth = attn_depth,
        #     dim_head = attn_dim_head,
        #     heads = attn_heads,
        #     cross_attend = True,
        #     adaptive_ln = False,
        #     causal = True,
        #     final_norm = True
        # )

        self.final_norm = RMSNorm(dim)

        self.dueling = dueling
        if dueling:
            self.to_values = nn.Parameter(torch.zeros(num_actions, dim))

    @property
    def device(self):
        return self.action_bin_embeddings.device

    def maybe_append_actions(self, sos_tokens, actions: Optional[Tensor] = None):
        # print('actions:', actions.shape)
        # print('sos_tokens:', sos_tokens.shape)
        
        if not exists(actions):
            # print('not exist actions')
            return sos_tokens
        
        batch, num_actions = actions.shape
        action_embeddings = self.action_bin_embeddings[:num_actions]
        # print('action_embeddings:', action_embeddings.shape)
        
        action_embeddings = repeat(action_embeddings, 'n a d -> b n a d', b = batch)
        past_action_bins = repeat(actions, 'b n -> b n 1 d', d = action_embeddings.shape[-1])

        bin_embeddings = action_embeddings.gather(-2, past_action_bins)
        bin_embeddings = rearrange(bin_embeddings, 'b n 1 d -> b n d')

        tokens, _ = pack((sos_tokens, bin_embeddings), 'b * d')
        tokens = tokens[:, :self.num_actions] # last action bin not needed for the proposed q-learning
        return tokens

    def get_q_values(self, embed):
        num_actions = embed.shape[-2]

        if exists(self.to_q_values):
            logits = self.to_q_values(embed)
        else:
            # each token predicts next action bin
            action_bin_embeddings = self.action_bin_embeddings[:num_actions]
            action_bin_embeddings = torch.roll(action_bin_embeddings, shifts = -1, dims = 1)
            logits = einsum('b n d, n a d -> b n a', embed, action_bin_embeddings)

        if self.dueling:
            advantages = logits
            values = einsum('b n d, n d -> b n', embed, self.to_values[:num_actions])
            values = rearrange(values, 'b n -> b n 1')

            q_values = values + (advantages - reduce(advantages, '... a -> ... 1', 'mean'))
        else:
            q_values = logits

        return q_values.sigmoid()

    def get_random_actions(self, batch_size, num_actions = None):
        num_actions = default(num_actions, self.num_actions)
        return torch.randint(0, self.action_bins, (batch_size, num_actions), device = self.device)

    @torch.no_grad()
    def get_optimal_actions(
        self,
        encoded_state,
        return_q_values = False,
        actions: Optional[Tensor] = None,
        prob_random_action: float = 0.5,
        **kwargs
    ):
        assert 0. <= prob_random_action <= 1.
        batch = encoded_state.shape[0]

        if prob_random_action == 1:
            return self.get_random_actions(batch)

        sos_token = reduce(encoded_state, 'b ... d -> b 1 d', 'mean')
        
        # print('sos_token:', sos_token.shape)
        tokens = self.maybe_append_actions(sos_token, actions = actions)
        # print('tokens:', tokens.shape)
        action_bins = []
        cache = None

        for action_idx in range(self.num_actions):
            
            # TODO: add mamba

            # embed, cache = self.transformer(
            #     tokens,
            #     context = encoded_state,
            #     cache = cache,
            #     return_cache = True
            # )
            
            # if tokens.is_cuda:
            #     print(f"tokens is on CUDA device: {tokens.device}")
            # else:
            #     print("tokens is on CPU")
            
            # if next(self.mamba.parameters()).is_cuda:
            #     print(f"Model 'mamba' is on CUDA device: {next(self.mamba.parameters()).device}")
            # else:
            #     print("Model 'mamba' is on CPU")
            
            device = tokens.device
            tokens = tokens.to("cuda")
            embed = self.mamba(tokens)
            embed = embed.to(device)
            tokens = tokens.to(device)
            
            # print('embed:', embed.shape)
            # print("cache:", cache.shape)
            

            last_embed = embed[:, action_idx]
            bin_embeddings = self.action_bin_embeddings[action_idx]

            q_values = einsum('b d, a d -> b a', last_embed, bin_embeddings)

            selected_action_bins = q_values.argmax(dim = -1)

            if prob_random_action > 0.:
                random_mask = torch.zeros_like(selected_action_bins).float().uniform_(0., 1.) < prob_random_action
                random_actions = self.get_random_actions(batch, 1)
                random_actions = rearrange(random_actions, '... 1 -> ...')

                selected_action_bins = torch.where(
                    random_mask,
                    random_actions,
                    selected_action_bins
                )


            next_action_embed = bin_embeddings[selected_action_bins]

            tokens, _ = pack((tokens, next_action_embed), 'b * d')

            action_bins.append(selected_action_bins)

        action_bins = torch.stack(action_bins, dim = -1)

        if not return_q_values:
            return action_bins

        all_q_values = self.get_q_values(embed)
        return action_bins, all_q_values

    def forward(
        self,
        encoded_state: Tensor,
        actions: Optional[Tensor] = None
    ):
        """
        einops
        b - batch
        n - number of actions
        a - action bins
        d - dimension
        """

        # this is the scheme many hierarchical transformer papers do

        sos_token = reduce(encoded_state, 'b ... d -> b 1 d', 'mean')

        tokens = self.maybe_append_actions(sos_token, actions = actions)

        # TODO: add mamba
        # embed = self.transformer(tokens, context = encoded_state)
        device = tokens.device
        tokens = tokens.to("cuda")
        embed = self.mamba(tokens)
        embed = embed.to(device)

        return self.get_q_values(embed)

# Robotic Transformer

class QMamba(Module):

    @beartype
    def __init__(
        self,
        cfg,
        *,
        num_actions = 8,
        action_bins = 256,
        dim_head = 9, 
        cond_drop_prob = 0.2,
        dueling = False,                       # https://arxiv.org/abs/1511.06581
        q_head_attn_kwargs: dict = dict(
            attn_heads = 8,
            attn_dim_head = 64, # 64,
            attn_depth = 2
        ),
        weight_tie_action_bin_embed = True      # when projecting to action bin Q values, whether to weight tie to original embeddings
    ):
        super().__init__()


        # q-transformer related action embeddings

        assert num_actions >= 1

        if cfg.num_actions != None:
            num_actions = cfg.num_actions
        self.num_actions = num_actions
        self.is_single_action = num_actions == 1
        self.action_bins = action_bins


        self.cond_drop_prob = cond_drop_prob

        # Q head
        

        if self.is_single_action:
            self.q_head = QHeadSingleAction(
                dim = dim_head,
                num_learned_tokens = self.num_learned_tokens,
                action_bins = action_bins,
                dueling = dueling
            )
        else:
            self.q_head = QHeadMultipleActions(
                dim = dim_head,
                num_actions = num_actions,
                action_bins = action_bins,
                dueling = dueling,
                weight_tie_action_bin_embed = weight_tie_action_bin_embed,
                **q_head_attn_kwargs
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def get_random_actions(self, batch_size = 1):
        return self.q_head.get_random_actions(batch_size)

    # @beartype
    # def embed_texts(self, texts: List[str]):
    #     return self.conditioner.embed_texts(texts)

    @torch.no_grad()
    def get_optimal_actions(
        self,
        *args,
        return_q_values = False,
        actions: Optional[Tensor] = None,
        **kwargs
    ):
        
        encoded_state = args[0]
        
        encoded_state = rearrange(encoded_state, 'b d -> b 1 d')
        # print("--encoded_state.shape",encoded_state.shape)
        # print("--encoded_state.dtype",encoded_state.dtype)
        return self.q_head.get_optimal_actions(encoded_state, return_q_values = return_q_values, actions = actions)

    def get_actions(
        self,
        state,
        *args,
        prob_random_action = 0.,  # otherwise known as epsilon in RL
        **kwargs,
    ):
        batch_size = state.shape[0]
        assert 0. <= prob_random_action <= 1.

        if random() < prob_random_action:
            return self.get_random_actions(batch_size = batch_size)

        return self.get_optimal_actions(state, *args, **kwargs)

    # @classifier_free_guidance
    def forward(
        self,
        feats: Tensor,
        actions: Optional[Tensor] = None,
    ):

        # just auto-move inputs to the same device as optimize transformer

        feats = feats.to(self.device)

        if exists(actions):
            actions = actions.to(self.device)

        # head that returns the q values
        # supporting both single and multiple actions

        feats = rearrange(feats, 'b d -> b 1 d')
        if self.is_single_action:
            assert not exists(actions), 'actions should not be passed in for single action optimize transformer'
            q_values = self.q_head(feats)
        else:
            q_values = self.q_head(feats, actions = actions)

        return q_values
