"""
This module provides the :class:`NunchakuQwenImage21DiTLoader` class for loading Nunchaku Qwen-Image-2.1 models.
"""

import json
import logging

import comfy.utils
import torch
from comfy import model_detection, model_management
from comfy.model_patcher import ModelPatcher

from nunchaku.models.transformers.utils import convert_fp16, patch_scale_key
from nunchaku.utils import check_hardware_compatibility, get_precision_from_quantization_config, is_turing

from ...model_configs.qwenimage21 import NunchakuQwenImage21
from ..utils import get_filename_list, get_full_path_or_raise


def _patch_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Patch the state dict.

    Convert the keys from diffusers style to Comfy-Org style keys.
    See https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/diffusion_models/qwen_image_2.1_bf16.safetensors

    Parameters
    ----------
    state_dict (dict[str, torch.Tensor]):
        the state dict loaded from diffuser style safetensors checkpoint file.

    Returns
    -------
    dict[str, torch.Tensor]:
        the patched state dict which follows Comfy-Org style keys.
    """
    patched_state_dict = {}
    quant_sub_keys = ["wscales", "wcscales", "wtscale", "smooth_factor_orig", "smooth_factor", "proj_down", "proj_up"]

    for key, value in state_dict.items():
        # region: feed forward
        if "img_mlp.net.0.proj.qweight" in key:
            patched_state_dict[key.replace("net.0.proj", "gate_up")] = value
            for subkey in quant_sub_keys:
                quant_param_key = key.replace("qweight", subkey)
                if quant_param_key in state_dict:
                    patched_state_dict[quant_param_key.replace("net.0.proj", "gate_up")] = state_dict[quant_param_key]
        elif any("img_mlp.net.0.proj." + subkey in key for subkey in quant_sub_keys):
            continue
        elif "img_mlp.net.2.qweight" in key:
            patched_state_dict[key.replace("net.2", "out")] = value
            for subkey in quant_sub_keys:
                quant_param_key = key.replace("qweight", subkey)
                if quant_param_key in state_dict:
                    patched_state_dict[quant_param_key.replace("net.2", "out")] = state_dict[quant_param_key]
        elif any("img_mlp.net.2." + subkey in key for subkey in quant_sub_keys):
            continue
        # end of region
        # region: others
        else:
            patched_state_dict[key] = value
        # end of region

    return patched_state_dict


def _load(sd: dict[str, torch.Tensor], metadata: dict[str, str] = {}):
    """
    Load a Nunchaku-quantized Qwen-Image-2.1 diffusion model.

    Parameters
    ----------
    sd : dict[str, torch.Tensor]
        The state dictionary of the model.
    metadata : dict[str, str], optional
        Metadata containing quantization configuration (default is empty dict).

    Returns
    -------
    comfy.model_patcher.ModelPatcher
        The patched and loaded Qwen-Image-2.1 model ready for inference.
    """
    quantization_config = json.loads(metadata.get("quantization_config", "{}"))
    precision = get_precision_from_quantization_config(quantization_config)
    rank = quantization_config.get("rank", 32)

    # Allow loading unets from checkpoint files
    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd

    load_device = model_management.get_torch_device()
    offload_device = model_management.unet_offload_device()
    check_hardware_compatibility(quantization_config, load_device)

    new_sd = sd

    model_config = NunchakuQwenImage21(rank=rank, precision=precision)

    if not is_turing():
        unet_dtype = torch.bfloat16
        manual_cast_dtype = None
        torch_dtype = torch.bfloat16
    else:
        unet_dtype = torch.bfloat16
        manual_cast_dtype = torch.float16
        torch_dtype = torch.float16
    logging.info(f"unet_dtype: {unet_dtype}, manual_cast_dtype: {manual_cast_dtype}, svdq_linear_dtype: {torch_dtype}")
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    patched_sd = _patch_state_dict(new_sd)
    model = model_config.get_model(patched_sd, "", torch_dtype=torch_dtype)

    patch_scale_key(model.diffusion_model, patched_sd)
    if torch_dtype == torch.float16:
        convert_fp16(model.diffusion_model, patched_sd)

    model.load_model_weights(patched_sd, "")
    return ModelPatcher(model, load_device=load_device, offload_device=offload_device)


class NunchakuQwenImage21DiTLoader:
    """
    Loader for Nunchaku Qwen-Image-2.1 models.

    Attributes
    ----------
    RETURN_TYPES : tuple
        Output types for the node ("MODEL",).
    FUNCTION : str
        Name of the function to call ("load_model").
    CATEGORY : str
        Node category ("Nunchaku").
    TITLE : str
        Node title ("Nunchaku Qwen-Image-2.1 DiT Loader").
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        Define the input types and tooltips for the node.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and their descriptions for the node interface.
        """
        return {
            "required": {
                "model_name": (
                    get_filename_list("diffusion_models"),
                    {"tooltip": "The Nunchaku Qwen-Image-2.1 model."},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku Qwen-Image-2.1 DiT Loader"

    def load_model(self, model_name: str, **kwargs):
        """
        Load the Qwen-Image-2.1 model from file and return a patched model.

        Parameters
        ----------
        model_name : str
            The filename of the Qwen-Image-2.1 model to load.

        Returns
        -------
        tuple
            A tuple containing the loaded and patched model.
        """
        model_path = get_full_path_or_raise("diffusion_models", model_name)
        sd, metadata = comfy.utils.load_torch_file(model_path, return_metadata=True)

        model = _load(sd, metadata=metadata)

        return (model,)
