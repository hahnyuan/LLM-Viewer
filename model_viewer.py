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

model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
config = importlib.import_module(args.config_path.replace("/", ".").replace(".py", ""))
num_attention_heads = config.get_num_attention_heads(model_config)
hidden_size = config.get_hidden_size(model_config)
num_key_value_heads = config.get_num_key_value_heads(model_config)
num_hidden_layers = config.get_num_hidden_layers(model_config)

model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)

layer_weight_numel = {}
layer_mac_decode = {}
layer_input_numel_decode = {}
layer_output_numel_decode = {}
workspace_decode = {}

layer_mac_prefill = {}
layer_input_numel_prefill = {}
layer_output_numel_prefill = {}
workspace_prefill = {}
kv_cache_numel = {}
for name, module in model.named_modules():
    if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
        layer_weight_numel[name] = module.weight.numel()
    if isinstance(module, nn.Linear):
        layer_mac_decode[name] = module.weight.numel() * batchsize
        layer_input_numel_decode[name] = module.in_features * batchsize
        layer_output_numel_decode[name] = module.out_features * batchsize
        if config.lm_head_name in name:
            # the lm_head not need to
            layer_mac_prefill[name] = module.weight.numel() * batchsize
            layer_input_numel_prefill[name] = module.in_features * batchsize
            layer_output_numel_prefill[name] = module.out_features * batchsize
        else:
            layer_mac_prefill[name] = module.weight.numel() * batchsize * seqlen
            layer_input_numel_prefill[name] = module.in_features * batchsize * seqlen
            layer_output_numel_prefill[name] = module.out_features * batchsize * seqlen

# For attention
for layeri in range(num_hidden_layers):
    head_size = hidden_size // num_attention_heads
    kv_cache_numel[f"layer{layeri}.kv_cache"] = (
        seqlen * head_size * num_key_value_heads * batchsize
    )
    # QK matmul
    qk_name = f"layer{layeri}.qk"
    layer_mac_decode[qk_name] = 1 * seqlen * head_size * num_attention_heads * batchsize
    layer_input_numel_decode[qk_name] = (
        (seqlen + 1) * head_size * batchsize * num_attention_heads
    )
    layer_output_numel_decode[qk_name] = 1 * seqlen * batchsize * num_attention_heads

    layer_mac_prefill[qk_name] = (
        seqlen * seqlen * head_size * num_attention_heads * batchsize
    )
    layer_input_numel_prefill[qk_name] = (
        seqlen * head_size * batchsize * num_attention_heads * 2
    )
    layer_output_numel_prefill[qk_name] = (
        seqlen * seqlen * batchsize * num_attention_heads
    )

    # SV matmul
    sv_name = f"layer{layeri}.sv"
    layer_mac_decode[sv_name] = 1 * head_size * seqlen * num_attention_heads * batchsize
    layer_input_numel_decode[sv_name] = (
        seqlen * head_size * batchsize * num_attention_heads
        + 1 * seqlen * batchsize * num_attention_heads
    )
    layer_output_numel_decode[sv_name] = 1 * head_size * batchsize * num_attention_heads

    layer_mac_prefill[sv_name] = (
        seqlen * head_size * seqlen * num_attention_heads * batchsize
    )
    layer_input_numel_prefill[sv_name] = (
        seqlen * head_size * batchsize * num_attention_heads
        + seqlen * seqlen * batchsize * num_attention_heads
    )
    layer_output_numel_prefill[sv_name] = (
        seqlen * head_size * batchsize * num_attention_heads
    )

# for workspace

# calculate total 
total_mac_decode = np.sum(list(layer_mac_decode.values()))
total_mac_prefill = np.sum(list(layer_mac_prefill.values()))
total_weight = np.sum(list(layer_weight_numel.values()))
total_kv_cache = np.sum(list(kv_cache_numel.values()))
total_input_numel_decode = np.sum(list(layer_input_numel_decode.values()))
total_output_numel_decode = np.sum(list(layer_output_numel_decode.values()))
total_input_numel_prefill = np.sum(list(layer_input_numel_prefill.values()))
total_output_numel_prefill = np.sum(list(layer_output_numel_prefill.values()))


print(f"decode total MAC(multiply-accumulation): {total_mac_decode}")
print(f"prefill total MAC(multiply-accumulation): {total_mac_prefill}")
print(f"total weight: {total_weight}")
print(f"total kv_cache: {total_kv_cache}")
print(f"decode total input numel: {total_input_numel_decode}")
print(f"decode total output numel: {total_output_numel_decode}")
print(f"prefill total input numel: {total_input_numel_prefill}")
print(f"prefill total output numel: {total_output_numel_prefill}")

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

# KV cache
kv_cache_file = open(f"{save_path}_kv_cache.csv", "w")
kv_cache_file.write("layer,weight\n")
for k, v in kv_cache_numel.items():
    kv_cache_file.write(f"{k},{v}\n")
kv_cache_file.write(f"total,{total_kv_cache}")
kv_cache_file.close()
