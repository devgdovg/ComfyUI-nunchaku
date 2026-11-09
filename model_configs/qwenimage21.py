"""
This module provides a wrapper for ComfyUI's Qwen-Image-2.1 model configuration.

Note
----
Codes are adapted from:
 - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_detection.py#model_config_from_unet
 - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/supported_models.py#QwenImage21
 - https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/model_base.py#QwenImage21
"""

import torch
from comfy.model_base import QwenImage21
from comfy.supported_models import QwenImage21 as QwenImage21ModelConfig

from ..models.qwenimage21 import patch_model


class NunchakuQwenImage21(QwenImage21ModelConfig):
    """
    Nunchaku Qwen-Image-2.1 model_config.
    """

    _DIT_CONFIG_ = {
        # This dict is the same as the result of execution of `comfy.model_detection#detect_unet_config` method for Qwen-Image-2.1
        'image_model': 'qwen_image21',
        'in_channels': 64,
        'out_channels': 64,
        'num_layers': 32,
        'attention_head_dim': 128,
        'num_attention_heads': 32,
        'context_in_dim': 4096,
        'mlp_ratio': 3,
        'fused_mlp': True
    }
    

    def __init__(self, rank: int = 32, precision: str = "int4"):
        super().__init__(unet_config=self._DIT_CONFIG_)
        self.rank = rank
        self.precision = precision

    def get_model(self, state_dict: dict[str, torch.Tensor], prefix: str = "", device=None, **kwargs) -> QwenImage21:
        """
        Instantiate and return a Nunchaku optimized QwenImage21 model_base object.

        Parameters
        ----------
        state_dict : dict
            Model state dictionary.
        prefix : str, optional
            Prefix for parameter names (default is "").
        device : torch.device or str, optional
            Device to load the model onto.
        **kwargs
            Additional keyword arguments for model initialization.

        Returns
        -------
        model_base.QwenImage21
            Instantiated model_base.QwenImage21 object with Nunchaku quantized transformer blocks.
        """
        out: QwenImage21 = super().get_model(state_dict, prefix, device)
        patch_model(out.diffusion_model, rank=self.rank, precision=self.precision, **kwargs)
        return out
