"""
This module patches the original Qwen-Image-2.1 transformer model by injecting into Nunchaku optimized attention and feed forward submodules.

Note
----
The injected transformer model is `comfy.ldm.qwen_image21.model.QwenImage21Transformer2DModel` in ComfyUI repository
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel, QwenImage21TransformerBlock, SwiGLUFeedForward, Attention
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.modules.attention import AttentionTensorContainer

from .zimage import fused_qkv_norm_rotary

from nunchaku.models.embeddings import pack_rotemb
from nunchaku.models.linear import SVDQW4A4Linear
from nunchaku.utils import pad_tensor


def add_comfy_cast_weights_attr(svdq_linear: SVDQW4A4Linear, comfy_linear: nn.Linear):
    """
    Add dummy `comfy_cast_weights` and `weight`fields to a SVDQW4A4Linear module.

    Make it compatible with offloading mechanism in ModelPatcher class in a lower vram condition.

    Note
    ----
    See method `comfy.model_patcher.ModelPatcher#_load_list` and method `comfy.model_patcher.ModelPatcher#load`
    """
    if hasattr(comfy_linear, "comfy_cast_weights"):
        svdq_linear.comfy_cast_weights = comfy_linear.comfy_cast_weights
        svdq_linear.weight = None


def concat_to_svdq_linear(qkv_linears: list[nn.Linear], **kwargs) -> SVDQW4A4Linear:
    """
    Concat q/k/v linear modules into one SVDQW4A4Linear.

    The q/k/v linear modules MUST have equal `in_features`.

    Parameters
    ----------
    qkv_linear : list[nn.Linear]
        q/k/v linear modules as a list

    Returns
    -------
    SVDQW4A4Linear:
        The fused module.
    """
    assert len(qkv_linears) == 3 and all([l is not None for l in qkv_linears])
    assert all([l.in_features == qkv_linears[0].in_features for l in qkv_linears])
    assert all([l.bias is None for l in qkv_linears])
    if "torch_dtype" in kwargs:
        torch_dtype = kwargs.pop("torch_dtype")
    elif qkv_linears[0].weight is not None:
        torch_dtype = qkv_linears[0].weight.dtype
    else:
        raise ValueError("no valid torch_dtype for SVDQW4A4Linear")
    device = None if qkv_linears[0].weight is None else qkv_linears[0].weight.device
    svdq_linear = SVDQW4A4Linear(
        qkv_linears[0].in_features,
        sum([l.out_features for l in qkv_linears]),
        bias=False,
        torch_dtype=torch_dtype,
        device=device,
        **kwargs,
    )
    add_comfy_cast_weights_attr(svdq_linear, qkv_linears[0])
    return svdq_linear


class ComfyNunchakuQwenImage21Attention(Attention):
    """
    Nunchaku optimized attention module for QwenImage21.
    """

    def __init__(self, orig_attn: Attention, **kwargs):
        nn.Module.__init__(self)
        self.comfy_attention = orig_attn.comfy_attention
        self.heads = orig_attn.heads
        self.to_qkv = concat_to_svdq_linear([orig_attn.to_q, orig_attn.to_k, orig_attn.to_v], **kwargs)
        self.to_out = nn.ModuleList([SVDQW4A4Linear.from_linear(orig_attn.to_out[0], **kwargs)])
        add_comfy_cast_weights_attr(self.to_out[0], orig_attn.to_out[0])
        self.norm_q = orig_attn.norm_q
        self.norm_k = orig_attn.norm_k

    def forward(self, x, pe, attn_fn, prefix_len, transformer_options={}, **kwargs):
        """
        Adapted from comfy.ldm.qwen_image21.model.Attention#forward
        
        Optimized with fused and quantized qkv projection, q_norm, k_norm and RoPE.
        """
        
        # (B, N, H, D) throughout: no transposes, the rope table is laid out to match
        B, N, _ = x.shape
        patches = transformer_options.get("patches", {}).get("attn1_patch", [])
        if comfy.model_management.in_training or patches or ("packed_pe" not in kwargs):
            qkv = self.to_qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, N, self.heads, -1)
            k = k.view(B, N, self.heads, -1)
            v = v.view(B, N, self.heads, -1)
            
            q, k = self.norm_q(q), self.norm_k(k)
            if patches:
                # patches see the Qwen-Image convention: (B, H, N, D) before rope, rope table (1, 1, N, ...), target image rows in img_slice
                q, k, v, pe = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), pe.transpose(1, 2)
                extra_options = {**transformer_options, "img_slice": [prefix_len, N]}
                for p in patches:
                    out = p(q, k, v, pe=pe, attn_mask=None, extra_options=extra_options)
                    q, k, v, pe = out.get("q", q), out.get("k", k), out.get("v", v), out.get("pe", pe)
                q, k, v, pe = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(), pe.transpose(1, 2).contiguous()
            q = apply_rope1(q, pe)
            k = apply_rope1(k, pe)
        else:
            assert "packed_pe" in kwargs
            qkv = fused_qkv_norm_rotary(
                x,
                self.to_qkv,
                self.norm_q.weight,
                self.norm_k.weight,
                kwargs['packed_pe'],
            )
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, N, self.heads, -1)
            k = k.view(B, N, self.heads, -1)
            v = v.view(B, N, self.heads, -1)
        q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
        return self.to_out[0](attn_fn(q, k, v, self.heads, preferred_attention=self.comfy_attention))


class ComfyNunchakuQwenImage21FeedForward(nn.Module):
    """
    Nunchaku optimized feed forward module for QwenImage21.
    """

    def __init__(self, orig_ff: SwiGLUFeedForward, **kwargs):
        super().__init__()
        self.gate_up = SVDQW4A4Linear.from_linear(orig_ff.gate_up, **kwargs)
        add_comfy_cast_weights_attr(self.gate_up, orig_ff.gate_up)
        self.out = SVDQW4A4Linear.from_linear(orig_ff.out, **kwargs) 
        add_comfy_cast_weights_attr(self.out, orig_ff.out)

    def forward(self, x: torch.Tensor):
        x = self.gate_up(x)
        hidden_states, gate = x.chunk(2, dim=-1)        
        return self.out(hidden_states * F.silu(gate))
    

class QwenImage21RopeFuseAttentionHook:
    """
    This class acts as a pre-forward hook on `ComfyNunchakuQwenImage21Attention`,
    replace the `freqs_cis` tensor with packed one to align with the input format of `fused_qkv_norm_rotary` method.
    """

    def __init__(self):
        self.packed_freqs_cis_cache = {}
        self.hook_handles = []

    def pre_forward(self, module: ComfyNunchakuQwenImage21Attention, input_args: tuple, input_kwargs: dict):
        """
        Pre-forward hook method.
        Create a `packed_freqs_cis` tensor that is aligned with the input format of `fused_qkv_norm_rottary` method
        from the original `freqs_cis` and cache it in the  `packed_freqs_cis_cache` dict.
        """
        x: torch.Tensor = input_args[0]
        freqs_cis: torch.Tensor = input_args[1]
        if freqs_cis is None:
            return None
        cache_key = (freqs_cis.data_ptr(), freqs_cis.shape)
        packed_freqs_cis = self.packed_freqs_cis_cache.get(cache_key, None)
        if packed_freqs_cis is None:
            # freqs_cis shape example: torch.Size([1, 4160, 1, 64, 2, 2])
            # freqs_cis dtype: torch.float32
            freqs_cis = freqs_cis[..., [1], :].squeeze(2)  # See comfy.ldm.flux.math#rope, #apply_rope
            freqs_cis = freqs_cis.expand(x.shape[0], -1, -1, -1, -1)  # [b, s, 64, 1, 2]
            freqs_cis = freqs_cis.flatten(0, 1)  # [b*s, 64, 1, 2]
            freqs_cis = freqs_cis.unsqueeze(0)  # [1, b*s, 64, 1, 2]
            packed_freqs_cis = pack_rotemb(pad_tensor(freqs_cis, 256, 1))
            self.packed_freqs_cis_cache[cache_key] = packed_freqs_cis
        input_kwargs['packed_pe'] = packed_freqs_cis
        return input_args, input_kwargs

    def hook(self, module: ComfyNunchakuQwenImage21Attention):
        assert isinstance(module, ComfyNunchakuQwenImage21Attention)
        self.hook_handles.append(module.register_forward_pre_hook(self.pre_forward, with_kwargs=True))

    def unhook(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()
        self.packed_freqs_cis_cache.clear()


class QwenImage21RopeFuseTransformerHook:
    """
    This class acts as a (both pre- and post- forward) hook on `comfy.ldm.qwen_image21.model.QwenImage21Transformer2DModel`,
    add hooks to its attention submodules for qkv/norm/RoPE fusion before forward pass,
    and remove the hooks added afterwards.
    """

    def pre_forward(self, module: QwenImage21Transformer2DModel, input_args: tuple):
        """
        Pre-forward hook method.
        Add a newly created `RopeFuseAttentionHook` object to all the attention submodules in the `QwenImage21Transformer2DModel` model.
        """
        self.attn_hook = QwenImage21RopeFuseAttentionHook()
        for _, transformer_block in enumerate(module.transformer_blocks):
            self.attn_hook.hook(transformer_block.attn)
        return None

    def post_forward(self, module: QwenImage21Transformer2DModel, input_args: tuple, output: tuple):
        """
        Post-forward hook method.
        Remove the hooks added by `pre_forward` method and clear the cached items.
        """
        self.attn_hook.unhook()
        return None

    def hook(self, model: QwenImage21Transformer2DModel):
        assert isinstance(model, QwenImage21Transformer2DModel)
        self.pre_handle = model.register_forward_pre_hook(self.pre_forward)
        self.post_handle = model.register_forward_hook(self.post_forward, always_call=True)


def patch_model(diffusion_model: QwenImage21Transformer2DModel, **kwargs):
    """
    Patch the QwenImage21 diffusion model by replacing the attention and feed forward modules with Nunchaku optimized ones in the transformer blocks.

    Parameters
    ----------
    diffusion_model : QwenImage21Transformer2DModel
        The QwenImage21 diffusion model to be patched.
    """

    def _patch_transformer_block(block_list: List[QwenImage21TransformerBlock]):
        for _, block in enumerate(block_list):
            block.attn = ComfyNunchakuQwenImage21Attention(block.attn, **kwargs)
            block.img_mlp = ComfyNunchakuQwenImage21FeedForward(block.img_mlp, **kwargs)

    _patch_transformer_block(diffusion_model.transformer_blocks)
    QwenImage21RopeFuseTransformerHook().hook(diffusion_model)
