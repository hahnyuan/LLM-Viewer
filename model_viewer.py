from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    OPTForCausalLM,
    LlamaForCausalLM,
)
from transformers import AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
import numpy as np
import os
import importlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_id", type=str, help="model id")
parser.add_argument("config_path", type=str, help="config file")
args = parser.parse_args()

model_id = args.model_id
batchsize = 1
seqlen = 1024

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
# model = LlamaForCausalLM(config)
model = OPTForCausalLM(config)
# breakpoint()

layer_weight_numel = {}
decode_layer_mac = {}
prefill_layer_mac = {}
for name, module in model.named_modules():
    if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
        layer_weight_numel[name] = module.weight.numel()
    if isinstance(module, nn.Linear):
        decode_layer_mac[name] = module.weight.numel() * batchsize
        if "head" not in name:
            prefill_layer_mac[name] = module.weight.numel() * batchsize * seqlen
        else:
            prefill_layer_mac[name] = module.weight.numel() * batchsize

# from configs.opt import *
# import using import lib
config_module = importlib.import_module(
    args.config_path.replace("/", ".").replace(".py", "")
)


# num_attention_heads_config = "num_attention_heads"
# num_attention_heads = config.getattr(num_attention_heads_config)
num_attention_heads = getattr(config, config_module.num_attention_heads_config)
# hidden_size_config = "hidden_size"
# hidden_size = config.getattr(hidden_size_config)
hidden_size = getattr(config, config_module.hidden_size_config)
# num_key_value_heads_config = "num_key_value_heads"
# num_key_value_heads = config.getattr(num_key_value_heads_config)
num_key_value_heads = getattr(config, config_module.num_key_value_heads_config)
# num_hidden_layers_config = "num_hidden_layers"
# num_hidden_layers = config.getattr(num_hidden_layers_config)
num_hidden_layers = getattr(config, config_module.num_hidden_layers_config)

for layeri in range(num_hidden_layers):
    qk_name = f"layer{layeri}.qk"
    head_size = hidden_size // num_attention_heads
    decode_layer_mac[qk_name] = 1 * seqlen * head_size * num_attention_heads * batchsize
    prefill_layer_mac[qk_name] = (
        seqlen * seqlen * head_size * num_attention_heads * batchsize
    )
    sv_name = f"layer{layeri}.sv"
    head_size = hidden_size // num_attention_heads
    decode_layer_mac[sv_name] = 1 * head_size * seqlen * num_attention_heads * batchsize
    prefill_layer_mac[sv_name] = (
        seqlen * head_size * seqlen * num_attention_heads * batchsize
    )

# calculate total mac and weights
total_mac = np.sum(list(decode_layer_mac.values()))
total_weight = np.sum(list(layer_weight_numel.values()))

print(f"total mac: {total_mac}")

# export to mac csv and weight csv, not use pandas
save_path = f"output/{model_id[:model_id.rfind('/')]}"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# decode mac
mac_file = open(f"{save_path}_mac.csv", "w")
mac_file.write("layer,mac\n")
for k, v in decode_layer_mac.items():
    mac_file.write(f"{k},{v}\n")
mac_file.write(f"total,{total_mac}")
mac_file.close()

# prefill mac
mac_file = open(f"{save_path}_mac_prefill.csv", "w")
mac_file.write("layer,mac\n")
for k, v in prefill_layer_mac.items():
    mac_file.write(f"{k},{v}\n")
mac_file.write(f"total,{total_mac}")
mac_file.close()

# weight
weight_file = open(f"{save_path}_weight.csv", "w")
weight_file.write("layer,weight\n")
for k, v in layer_weight_numel.items():
    weight_file.write(f"{k},{v}\n")
weight_file.write(f"total,{total_weight}")
weight_file.close()
