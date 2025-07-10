import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
import numpy as np


def torch_dtype_to_np_dtype(torch_dtype):
    if torch_dtype == torch.float16:
        return np.float16
    elif torch_dtype == torch.float32:
        return np.float32
    elif torch_dtype == torch.float64:
        return np.float64
    elif torch_dtype == torch.bfloat16:
        # numpy 没有 bfloat16，可以选 np.uint16 或 np.float16 占位
        return np.uint16
    elif torch_dtype == torch.int8:
        return np.int8
    elif torch_dtype == torch.int16:
        return np.int16
    elif torch_dtype == torch.int32:
        return np.int32
    elif torch_dtype == torch.int64:
        return np.int64
    elif torch_dtype == torch.bool:
        return np.bool_
    else:
        raise TypeError(f"Unsupported torch dtype: {torch_dtype}")

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))