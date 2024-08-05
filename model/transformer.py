import torch
import torch.nn as nn
import torch.nn.functional as F


from einops import pack, unpack, repeat, reduce, rearrange
from einops.layers.torch import Rearrange, Reduce

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


# 2d rotary positional embedding
# https://arxiv.org/abs/2104.09864

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, omega = 10000):
        super().__init__()
        inv_freq = 1.0 / (omega ** (torch.arange(0, dim, 4).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    @autocast(enabled = False)
    def forward(self, height_width):
        device, dtype = self.inv_freq.device, self.inv_freq.dtype

        axial_pos = torch.arange(height_width, device = device).type(dtype)

        freqs = torch.einsum('i, j -> i j', axial_pos, self.inv_freq)
        freqs = repeat(freqs, '... f -> ... (f c)', c = 2)

        freqs = torch.broadcast_tensors(freqs[None, :, :], freqs[:, None, :])
        freqs = torch.cat(freqs, dim = -1)
        return rearrange(freqs, '... f -> (...) f')

def rotate_half(x):
    x1, x2 = rearrange(x, '... (d c) -> ... d c', c = 2).unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d c -> ... (d c)')

@autocast(enabled = False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()

# sync batchnorm

@cache
def get_is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def MaybeSyncBatchnorm2d(is_distributed = None):
    is_distributed = default(is_distributed, get_is_distributed())
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm2d

# channel rmsnorm

class RMSNorm(nn.Module):
    def __init__(self, dim, affine = True):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if affine else 1.

    def forward(self, x):
        return l2norm(x) * self.gamma * self.scale



class TransformerAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        dim_context = None,
        heads = 8,
        num_mem_kv = 4,
        norm_context = False,
        adaptive_ln = False,
        dropout = 0.1,
        flash = True,
        causal = False
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.adaptive_ln = adaptive_ln
        self.norm = RMSNorm(dim, affine = not adaptive_ln)

        self.context_norm = RMSNorm(dim_context) if norm_context else None

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias = False)

        self.num_mem_kv = num_mem_kv
        self.mem_kv = None
        if num_mem_kv > 0:
            self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))

        self.attend = Attend(
            dropout = dropout,
            flash = flash,
            causal = causal
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_mask = None,
        cond_fn: Optional[Callable] = None,
        cache: Optional[Tensor] = None,
        return_cache = False
    ):
        b = x.shape[0]

        assert xnor(exists(context), exists(self.context_norm))

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        assert xnor(exists(cond_fn), self.adaptive_ln)

        if exists(cond_fn):
            x = cond_fn(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        if exists(cache):
            ck, cv = cache
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        new_kv_cache = torch.stack((k, v))

        if exists(self.mem_kv):
            mk, mv = map(lambda t: repeat(t, '... -> b ...', b = b), self.mem_kv)

            k = torch.cat((mk, k), dim = -2)
            v = torch.cat((mv, v), dim = -2)

            if exists(mask):
                mask = F.pad(mask, (self.num_mem_kv, 0), value = True)

            if exists(attn_mask):
                attn_mask = F.pad(attn_mask, (self.num_mem_kv, 0), value = True)

        out = self.attend(q, k, v, mask = mask, attn_mask = attn_mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if not return_cache:
            return out

        return out, new_kv_cache
    
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        depth = 6,
        attn_dropout = 0.,
        ff_dropout = 0.,
        adaptive_ln = False,
        flash_attn = True,
        cross_attend = False,
        causal = False,
        final_norm = True
    ):
        super().__init__()
        self.layers = ModuleList([])

        attn_kwargs = dict(
            dim = dim,
            heads = heads,
            dim_head = dim_head,
            dropout = attn_dropout,
            flash = flash_attn
        )

        for _ in range(depth):
            self.layers.append(ModuleList([
                TransformerAttention(**attn_kwargs, causal = causal, adaptive_ln = adaptive_ln, norm_context = False),
                TransformerAttention(**attn_kwargs, norm_context = True) if cross_attend else None,
                FeedForward(dim = dim, dropout = ff_dropout, adaptive_ln = adaptive_ln)
            ]))

        self.norm = RMSNorm(dim) if final_norm else nn.Identity()

    @beartype
    def forward(
        self,
        x,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        attn_mask = None,
        context: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
        return_cache = False
    ):
        has_cache = exists(cache)

        if has_cache:
            x_prev, x = x[..., :-1, :], x[..., -1:, :]

        cond_fns = iter(default(cond_fns, []))
        cache = iter(default(cache, []))

        new_caches = []

        for attn, maybe_cross_attn, ff in self.layers:
            attn_out, new_cache = attn(
                x,
                attn_mask = attn_mask,
                cond_fn = next(cond_fns, None),
                return_cache = True,
                cache = next(cache, None)
            )

            new_caches.append(new_cache)

            x = x + attn_out

            if exists(maybe_cross_attn):
                assert exists(context)
                x = maybe_cross_attn(x, context = context) + x

            x = ff(x, cond_fn = next(cond_fns, None)) + x

        new_caches = torch.stack(new_caches)

        if has_cache:
            x = torch.cat((x_prev, x), dim = -2)

        out = self.norm(x)

        if not return_cache:
            return out

        return out, new_caches

# token learner module
