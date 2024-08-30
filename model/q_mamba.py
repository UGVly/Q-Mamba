from mamba_ssm import Mamba
# from model.Attend import Attend
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

# from model.mamba_minimal import Mamba

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


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)

# tensor helpers


def l2norm(t, dim=-1):
    return F.normalize(t, dim=dim)


def pack_one(x, pattern):
    return pack([x], pattern)


def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]


def int_to_binary_float_tensor(int_tensor, action_dim):

    device = int_tensor.device
    
    binary_strings = [bin(i)[2:].zfill(action_dim) for i in int_tensor.view(-1).tolist()]
    
    float_tensor = torch.tensor([[float(bit) for bit in binary_str] for binary_str in binary_strings])
    
    return float_tensor.view(*int_tensor.shape, -1).to(device)

def binary_float_tensor_to_int(binary_float_tensor):

    device = binary_float_tensor.device

    int_tensor = torch.tensor([
        int(''.join(str(int(bit)) for bit in binary_row), 2)
        for binary_row in binary_float_tensor.view(-1, binary_float_tensor.size(-1))
    ])

    return int_tensor.view(*binary_float_tensor.shape[:-1]).to(device)
# channel rmsnorm

class RMSNorm(Module):
    def __init__(self, dim, affine=True):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if affine else 1.

    def forward(self, x):
        return l2norm(x) * self.gamma * self.scale


# # Dueling heads for Q value

# class DuelingHead(Module):
#     def __init__(
#         self,
#         dim,
#         expansion_factor = 2,
#         action_bins = 16,
#     ):
#         super().__init__()
#         dim_hidden = dim * expansion_factor

#         self.stem = nn.Sequential(
#             nn.Linear(dim, dim_hidden),
#             nn.SiLU()
#         )

#         self.to_values = nn.Sequential(
#             nn.Linear(dim_hidden, 1)
#         )

#         self.to_advantages = nn.Sequential(
#             nn.Linear(dim_hidden, action_bins)
#         )

#     def forward(self, x):
#         x = self.stem(x)

#         advantages = self.to_advantages(x)
#         advantages = advantages - reduce(advantages, '... a -> ... 1', 'mean')

#         values = self.to_values(x)

#         q_values = values + advantages
#         return q_values.sigmoid()

# Q head modules, for either single or multiple actions

# class QHeadSingleAction(Module):
#     def __init__(
#         self,
#         dim,
#         *,
#         num_learned_tokens = 8,
#         action_bins = 256,
#         dueling = False
#     ):
#         super().__init__()
#         self.action_bins = action_bins

#         if dueling:
#             self.to_q_values = nn.Sequential(
#                 Reduce('b (f n) d -> b d', 'mean', n = num_learned_tokens),
#                 DuelingHead(
#                     dim,
#                     action_bins = action_bins
#                 )
#             )
#         else:
#             self.to_q_values = nn.Sequential(
#                 Reduce('b (f n) d -> b d', 'mean', n = num_learned_tokens),
#                 RMSNorm(dim),
#                 nn.Linear(dim, action_bins),
#                 nn.Sigmoid()
#             )

#     def get_random_actions(self, batch_size):
#         return torch.randint(0, self.action_bins, (batch_size,), device = self.device)

#     def get_optimal_actions(
#         self,
#         encoded_state,
#         return_q_values = False,
#         actions = None,
#         **kwargs
#     ):
#         assert not exists(actions), 'single actions will never receive previous actions'

#         q_values = self.forward(encoded_state)

#         max_q, action_indices = q_values.max(dim = -1)

#         if not return_q_values:
#             return action_indices

#         return action_indices, max_q

#     def forward(self, encoded_state):
#         return self.to_q_values(encoded_state)



class DAC_block(Module):
    def __init__(self,state_dim,actions_dim,action_bins,d_state=32,d_conv=4,expand=2,num_hidden_mlp=32,device='cuda',mamba_num = 1):
        super().__init__()
        self.device = device
        self.mamba_blocks = nn.ModuleList([Mamba(
            d_model=state_dim + actions_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        ) for _ in range(mamba_num)])
        
        self.ff1 = nn.Linear(state_dim + actions_dim, num_hidden_mlp)
        self.ff2 = nn.Linear(num_hidden_mlp, action_bins)
        

    def forward(self, x):
        for mamba_block in self.mamba_blocks:
            x = mamba_block(x)
        x = F.leaky_relu(self.ff1(x))
        x = self.ff2(x)
        x = x.sigmoid()
        return x
    
    
class QHeadMultipleActions(Module):
    def __init__(
        self,
        state_dim=9,
        action_dim=5,
        num_actions=2,
        action_bins=16,
        use_binary_encode = True,
        # dueling=False,
        # weight_tie_action_bin_embed=False
    ):
        super().__init__()
        self.num_actions = num_actions
        self.action_bins = action_bins
        self.action_dim = action_dim
        self.use_binary_encode = use_binary_encode
        
        if use_binary_encode:
            assert action_bins + 1 <= 2**action_dim
        else:
            assert action_bins  == action_dim
        # self.action_bin_embeddings = nn.Parameter(torch.zeros(num_actions, action_bins, dim))
        # nn.init.normal_(self.action_bin_embeddings, std = 0.02)

        self.to_q_values = None
        # if not weight_tie_action_bin_embed:
        #     self.to_q_values = nn.Linear(state_dim + action_dim, action_bins)

        self.DAC_block = DAC_block(state_dim,action_dim,action_bins)

        # self.final_norm = RMSNorm(dim)

        # self.dueling = dueling
        # if dueling:
        #     self.to_values = nn.Parameter(torch.zeros(num_actions, dim))

    # @property
    # def device(self):
    #     return self.action_bin_embeddings.device

    def maybe_append_actions(self, sos_tokens, actions: Optional[Tensor] = None):
        # print('actions:', actions.shape)
        # print('sos_tokens:', sos_tokens.shape)
        batch = sos_tokens.shape[0]
        if not exists(actions):
            start_binary = torch.zeros(batch, 1, self.action_dim, device = sos_tokens.device)
            token = torch.cat((sos_tokens, start_binary), dim=-1)
            # print('not exist actions')
            return token
        
        batch, num_actions = actions.shape
        actions_binary_float = int_to_binary_float_tensor(actions, self.action_dim)
        start_binary = torch.zeros(batch, 1, self.action_dim, device = actions.device)
        binary_float = torch.cat((start_binary, actions_binary_float[:,:-1,]), dim=1)
        sos_tokens = repeat(sos_tokens, 'b 1 d -> b n d', n = num_actions)
        # print('sos_tokens:', sos_tokens.shape)
        # print('binary_float:', binary_float.shape)
        token = torch.cat((sos_tokens, binary_float), dim=-1)
        return token

        # batch, num_actions = actions.shape
        # action_embeddings = self.action_bin_embeddings[:num_actions]
        # # print('action_embeddings:', action_embeddings.shape)

        # action_embeddings = repeat(action_embeddings, 'n a d -> b n a d', b = batch)
        # past_action_bins = repeat(actions, 'b n -> b n 1 d', d = action_embeddings.shape[-1])

        # bin_embeddings = action_embeddings.gather(-2, past_action_bins)
        # bin_embeddings = rearrange(bin_embeddings, 'b n 1 d -> b n d')

        # tokens, _ = pack((sos_tokens, bin_embeddings), 'b * d')
        # tokens = tokens[:, :self.num_actions] # last action bin not needed for the proposed q-learning
        
        # return tokens

    # def get_q_values(self, embed):
    #     num_actions = embed.shape[-2]

    #     if exists(self.to_q_values):
    #         logits = self.to_q_values(embed)
    #     else:
    #         # each token predicts next action bin
    #         action_bin_embeddings = self.action_bin_embeddings[:num_actions]
    #         action_bin_embeddings = torch.roll(
    #             action_bin_embeddings, shifts=-1, dims=1)
    #         logits = einsum('b n d, n a d -> b n a',
    #                         embed, action_bin_embeddings)

    #     if self.dueling:
    #         advantages = logits
    #         values = einsum('b n d, n d -> b n', embed,
    #                         self.to_values[:num_actions])
    #         values = rearrange(values, 'b n -> b n 1')

    #         q_values = values + \
    #             (advantages - reduce(advantages, '... a -> ... 1', 'mean'))
    #     else:
    #         q_values = logits

    #     return q_values.sigmoid()

    def get_random_actions(self, batch_size, num_actions=None):
        num_actions = default(num_actions, self.num_actions)
        return torch.randint(0, self.action_bins, (batch_size, num_actions), device=self.device)

    @torch.no_grad()
    def get_optimal_actions(
        self,
        encoded_state,
        return_q_values=False,
        actions: Optional[Tensor] = None,
        prob_random_action: float = 0.5,
        **kwargs
    ):
        assert 0. <= prob_random_action <= 1.
        batch = encoded_state.shape[0]

        # if prob_random_action == 1:
        #     return self.get_random_actions(batch)

        sos_token = reduce(encoded_state, 'b ... d -> b 1 d', 'mean')

        # print('sos_token:', sos_token.shape)
        tokens = self.maybe_append_actions(sos_token, actions=actions)
        # print('tokens:', tokens.shape)
        action_bins = []

        print("numactions:", self.num_actions)
        for action_idx in range(self.num_actions):

            # if tokens.is_cuda:
            #     print(f"tokens is on CUDA device: {tokens.device}")
            # else:
            #     print("tokens is on CPU")

            # if next(self.mamba.parameters()).is_cuda:
            #     print(f"Model 'mamba' is on CUDA device: {next(self.mamba.parameters()).device}")
            # else:
            #     print("Model 'mamba' is on CPU")

            # device = tokens.device
            # tokens = tokens.to("cuda")
            # embed = self.mamba(tokens)
            # embed = embed.to(device)
            # tokens = tokens.to(device)

            q_values = self.DAC_block(tokens)
            # print('embed:', embed.shape)
            # print("cache:", cache.shape)

            # last_embed = q_values[:, action_idx]
            # bin_embeddings = self.action_bin_embeddings[action_idx]

            # q_values = einsum('b d, a d -> b a', last_embed, bin_embeddings)

            selected_action_bins = q_values.argmax(dim=-1)

            # if prob_random_action > 0.:
            #     random_mask = torch.zeros_like(selected_action_bins).float().uniform_(
            #         0., 1.) < prob_random_action
            #     random_actions = self.get_random_actions(batch, 1)
            #     random_actions = rearrange(random_actions, '... 1 -> ...')

            #     selected_action_bins = torch.where(
            #         random_mask,
            #         random_actions,
            #         selected_action_bins
            #     )

            # next_action_embed = bin_embeddings[selected_action_bins]
            
            next_action_embed = int_to_binary_float_tensor(selected_action_bins, self.action_dim)[:,-1,]
            print('selected_action_bins:', selected_action_bins.shape)
            print('next_action_embed:', next_action_embed.shape)
            next_token, _ = pack((sos_token, next_action_embed), 'b *')
            next_token = rearrange(next_token, 'b d -> b 1 d')
            print('next_token:', next_token.shape)
            print('tokens:', tokens.shape)
            tokens, _ = pack((tokens, next_token), 'b * d')

            # action_bins.append(selected_action_bins)

        # action_bins = torch.stack(action_bins, dim=-1)
        action_bins = selected_action_bins

        if not return_q_values:
            return action_bins

        # all_q_values = self.get_q_values(embed)
        return action_bins, q_values

    def forward(
        self,
        encoded_state: Tensor,  # TensorType['b t', 'f', float],
        num_timesteps=1,
        actions: Optional[Tensor] = None  # TensorType['b t', 'n', float],
    ):
        """
        einops
        b - batch
        n - number of actions
        a - action bins
        d - dimension
        """

        # if num_timesteps > 1:
        #     encoded_state = rearrange(encoded_state, 'b (t d) -> b t d', t = num_timesteps)
        #     encoded_state = rearrange(encoded_state, 'b t d -> (b t) d', t = num_timesteps)
        # sos_token = rearrange(encoded_state, 'b d -> b 1 d') # bz*num_timesteps 1 9
        sos_token = reduce(encoded_state, 'b ... d -> b 1 d', 'mean')
        # actions_token = reduce(actions, 'b ... d -> b 1 d', 'mean')

        # torch.cat((encoded_state1, encoded_state2), dim=1)

        tokens = self.maybe_append_actions(sos_token, actions=actions) # bz*num_timesteps 1 16

        if num_timesteps > 1:
            tokens = rearrange(tokens, '(b t) n d -> b t n d', t=num_timesteps)
            tokens = rearrange(tokens, 'b t n d -> b (t n) d', t=num_timesteps)

        q_values = self.DAC_block(tokens)
        
        if num_timesteps > 1:
            # embed = rearrange(embed, 'b (t n) d -> b t n d', t=num_timesteps)
            # embed = rearrange(embed, 'b t n d -> (b t) n d', t=num_timesteps)
            q_values = rearrange(q_values, 'b (t n) a -> b t n a', t=num_timesteps)
            q_values = rearrange(q_values, 'b t n a -> (b t) n a', t=num_timesteps)
        return q_values

# Robotic Transformer


class QMamba(Module):

    @beartype
    def __init__(
        self,
        cfg,
        *,
        state_dim=9,
        action_dim=5,
        num_actions=8,
        action_bins=16,
        use_binary_encode = True,
        # dim_head=9,
        # cond_drop_prob=0.2,
        # dueling=False,                       # https://arxiv.org/abs/1511.06581
        
        device = 'cuda'
        # when projecting to action bin Q values, whether to weight tie to original embeddings
        # weight_tie_action_bin_embed=True
    ):
        super().__init__()

        self.device = device
        # q-transformer related action embeddings

        assert num_actions >= 1

        if cfg.num_actions != None:
            num_actions = cfg.num_actions
        self.num_actions = num_actions
        self.is_single_action = num_actions == 1
        self.action_bins = action_bins

        # self.cond_drop_prob = cond_drop_prob

        # Q head

        # if self.is_single_action:
        #     self.q_head = QHeadSingleAction(
        #         dim=dim_head,
        #         num_learned_tokens=self.num_learned_tokens,
        #         action_bins=action_bins,
        #         dueling=dueling
        #     )
        # else:
        self.q_head = QHeadMultipleActions(
            state_dim=state_dim,
            action_dim=action_dim,
            num_actions=num_actions,
            action_bins=action_bins,
            use_binary_encode = use_binary_encode,
        )

    # @property
    # def device(self):
    #     return next(self.parameters()).device

    def get_random_actions(self, batch_size=1):
        return self.q_head.get_random_actions(batch_size)

    # @beartype
    # def embed_texts(self, texts: List[str]):
    #     return self.conditioner.embed_texts(texts)

    @torch.no_grad()
    def get_optimal_actions(
        self,
        *args,
        return_q_values=False,
        actions: Optional[Tensor] = None,
        **kwargs
    ):

        encoded_state = args[0]

        encoded_state = rearrange(encoded_state, 'b d -> b 1 d')
        
        return self.q_head.get_optimal_actions(encoded_state, return_q_values=return_q_values, actions=actions)

    def get_actions(
        self,
        state,
        *args,
        prob_random_action=0.,  # otherwise known as epsilon in RL
        **kwargs,
    ):
        batch_size = state.shape[0]
        assert 0. <= prob_random_action <= 1.

        # if random() < prob_random_action:
        #     return self.get_random_actions(batch_size=batch_size)

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
        # if self.is_single_action:
        #     assert not exists(
        #         actions), 'actions should not be passed in for single action optimize transformer'
        #     q_values = self.q_head(feats)
        # else:
        q_values = self.q_head(feats, actions=actions)

        return q_values
