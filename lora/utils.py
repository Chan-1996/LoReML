import os
import torch
import torch.nn as nn

from collections import OrderedDict

from lora.layers import LoRaLinear, LoRaEmbedding
from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding
from transformers.models.opt.modeling_opt import OPTLearnedPositionalEmbedding
from transformers.models.mbart.modeling_mbart import MBartLearnedPositionalEmbedding

def _setattr(model, name, module):
    name_list = name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    setattr(model, name_list[-1], module)

def convert_lora_network(
        model,
        weight_rank,
        alpha,
        logger=None,
        skip_names=None
):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'classifier' not in name:
            new_module = LoRaLinear(module.in_features, module.out_features,
                                      module.bias is not None, weight_rank, alpha)

            new_module.weight.data = module.weight.data
            if module.bias is not None:
                new_module.bias.data = module.bias.data

            # replace original module by new lora module
            _setattr(model, name, new_module)

            if logger:
                logger.info(f"convert {name} to lora module.")
            else:
                print(f"convert {name} to lora module.")
        elif isinstance(module, nn.Embedding) and not isinstance(module, BartLearnedPositionalEmbedding)\
                and not isinstance(module, OPTLearnedPositionalEmbedding) and not isinstance(module, MBartLearnedPositionalEmbedding):
            if skip_names is not None:
                if name in skip_names:
                    continue
            new_module = LoRaEmbedding(module.num_embeddings, module.embedding_dim,
                                     weight_rank, alpha)

            new_module.weight.data = module.weight.data

            # replace original module by new lora module
            _setattr(model, name, new_module)

            if logger:
                logger.info(f"convert {name} to lora module.")
            else:
                print(f"convert {name} to lora module.")

def convert_lora_network_att_qv(
        model,
        weight_rank,
        alpha,
        logger=None
):
    for name, module in model.named_modules():
        query_name = 'q_proj'
        value_name = 'v_proj'
        if isinstance(module, nn.Linear) and (query_name in name or value_name in name):
            new_module = LoRaLinear(module.in_features, module.out_features,
                                      module.bias is not None, weight_rank, alpha)

            new_module.weight.data = module.weight.data
            if module.bias is not None:
                new_module.bias.data = module.bias.data

            # replace original module by new lora module
            _setattr(model, name, new_module)

            if logger:
                logger.info(f"convert {name} to lora module.")
            else:
                print(f"convert {name} to lora module.")


def lora_state_dict(model):
    my_state_dict = model.state_dict()
    to_return = {}
    for k in my_state_dict:
        if 'weight_U' in k or 'weight_V' in k:
            to_return[k] = my_state_dict[k].float()
    return to_return
