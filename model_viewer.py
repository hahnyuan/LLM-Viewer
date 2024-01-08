from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    OPTForCausalLM,
    LlamaForCausalLM,
)
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
import torch.nn as nn
import numpy as np
import os
import importlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_id", type=str, help="model id")
parser.add_argument("config_path", type=str, help="config file")
parser.add_argument("--batchsize", type=int, default=1, help="batch size")
parser.add_argument("--seqlen", type=int, default=1024, help="sequence length")
args = parser.parse_args()

model_id = args.model_id
batchsize = args.batchsize
seqlen = args.seqlen

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
config_utils = importlib.import_module(
    args.config_path.replace("/", ".").replace(".py", "")
)
num_attention_heads = config_utils.get_num_attention_heads(config)
hidden_size = config_utils.get_hidden_size(config)
num_key_value_heads = config_utils.get_num_key_value_heads(config)
num_hidden_layers = config_utils.get_num_hidden_layers(config)

model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

layer_weight_numel = {}
layer_mac_decode = {}
layer_mac_prefill = {}
for name, module in model.named_modules():
    if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
        layer_weight_numel[name] = module.weight.numel()
    if isinstance(module, nn.Linear):
        layer_mac_decode[name] = module.weight.numel() * batchsize
        if "head" not in name:
            layer_mac_prefill[name] = module.weight.numel() * batchsize * seqlen
        else:
            layer_mac_prefill[name] = module.weight.numel() * batchsize


for layeri in range(num_hidden_layers):
    qk_name = f"layer{layeri}.qk"
    head_size = hidden_size // num_attention_heads
    layer_mac_decode[qk_name] = 1 * seqlen * head_size * num_attention_heads * batchsize
    layer_mac_prefill[qk_name] = (
        seqlen * seqlen * head_size * num_attention_heads * batchsize
    )
    sv_name = f"layer{layeri}.sv"
    head_size = hidden_size // num_attention_heads
    layer_mac_decode[sv_name] = 1 * head_size * seqlen * num_attention_heads * batchsize
    layer_mac_prefill[sv_name] = (
        seqlen * head_size * seqlen * num_attention_heads * batchsize
    )

# calculate total mac and weights
total_mac_decode = np.sum(list(layer_mac_decode.values()))
total_mac_prefill = np.sum(list(layer_mac_prefill.values()))
total_weight = np.sum(list(layer_weight_numel.values()))

print(f"total mac decode: {total_mac_decode}")
print(f"total mac prefill: {total_mac_prefill}")

# export to mac csv and weight csv, not use pandas
save_path = f"output/{model_id[:model_id.rfind('/')]}"
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_path += f"{model_id[model_id.rfind('/'):]}"

# decode mac
mac_file = open(f"{save_path}_mac_decode.csv", "a+")
mac_file.write(f"\n\nbatchsize,{batchsize},seqlen,{seqlen}\n")
mac_file.write("layer,mac\n")
for k, v in layer_mac_decode.items():
    mac_file.write(f"{k},{v}\n")
mac_file.write(f"total,{total_mac_decode}")
mac_file.close()

# prefill mac
mac_file = open(f"{save_path}_mac_prefill.csv", "a+")
mac_file.write(f"\n\nbatchsize,{batchsize},seqlen,{seqlen}\n")
mac_file.write("layer,mac\n")
for k, v in layer_mac_prefill.items():
    mac_file.write(f"{k},{v}\n")
mac_file.write(f"total,{total_mac_prefill}")
mac_file.close()

# weight
weight_file = open(f"{save_path}_weight.csv", "w")
weight_file.write("layer,weight\n")
for k, v in layer_weight_numel.items():
    weight_file.write(f"{k},{v}\n")
weight_file.write(f"total,{total_weight}")
weight_file.close()
