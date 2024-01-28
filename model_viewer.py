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
parser.add_argument(
    "hardware",
    type=str,
    help="name of hardware, for example nvidia_V100 or nvidia_A6000",
)
parser.add_argument("--batchsize", type=int, default=1, help="batch size")
parser.add_argument("--seqlen", type=int, default=1024, help="sequence length")
args = parser.parse_args()

model_id = args.model_id
batchsize = args.batchsize
seqlen = args.seqlen
# Roofline model
w_bit = 16
a_bit = 16

model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
config = importlib.import_module(args.config_path.replace("/", ".").replace(".py", ""))
num_attention_heads = config.get_num_attention_heads(model_config)
hidden_size = config.get_hidden_size(model_config)
num_key_value_heads = config.get_num_key_value_heads(model_config)
num_hidden_layers = config.get_num_hidden_layers(model_config)
num_mlp_act_size = 0


model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)


layer_weight_numel = {}
layer_mac_decode = {}
layer_load_act_numel_decode = {}
layer_store_act_numel_decode = {}

layer_mac_prefill = {}
layer_load_act_numel_prefill = {}
layer_store_act_numel_prefill = {}
kv_cache_numel = {}
transformer_layer_names = []
for name, module in model.named_modules():
    if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
        layer_weight_numel[name] = module.weight.numel()
    if isinstance(module, nn.Linear):
        layer_mac_decode[name] = module.weight.numel() * batchsize
        layer_load_act_numel_decode[name] = module.in_features * batchsize
        layer_store_act_numel_decode[name] = module.out_features * batchsize
        if config.lm_head_name in name:
            # the lm_head not need to
            layer_mac_prefill[name] = module.weight.numel() * batchsize
            layer_load_act_numel_prefill[name] = module.in_features * batchsize
            layer_store_act_numel_prefill[name] = module.out_features * batchsize
        else:
            transformer_layer_names.append(name)
            layer_mac_prefill[name] = module.weight.numel() * batchsize * seqlen
            layer_load_act_numel_prefill[name] = module.in_features * batchsize * seqlen
            layer_store_act_numel_prefill[name] = (
                module.out_features * batchsize * seqlen
            )
        num_mlp_act_size = max(num_mlp_act_size, module.in_features)

linear_mac_prefill = sum(layer_mac_prefill.values())
linear_mac_decode = sum(layer_mac_decode.values())
linear_mem_access_prefill = (
    sum(layer_load_act_numel_prefill.values())
    + sum(layer_store_act_numel_prefill.values())
) * a_bit + sum(layer_weight_numel.values()) * w_bit
linear_mem_access_decode = (
    sum(layer_load_act_numel_decode.values())
    + sum(layer_store_act_numel_decode.values())
) * a_bit + sum(layer_weight_numel.values()) * w_bit

# For attention
for layeri in range(num_hidden_layers):
    head_size = hidden_size // num_attention_heads
    kv_cache_numel[f"layer{layeri}.kv_cache"] = (
        seqlen * head_size * num_key_value_heads * batchsize
    )
    # LayerNorm
    for name in ["atten.layernom", "mlp.layernorm"]:
        layer_name = f"layer{layeri}.{name}"
        layer_load_act_numel_decode[layer_name] = batchsize * hidden_size
        layer_store_act_numel_decode[layer_name] = batchsize * hidden_size
        layer_load_act_numel_prefill[layer_name] = batchsize * hidden_size * seqlen
        layer_store_act_numel_prefill[layer_name] = batchsize * hidden_size * seqlen

    # QK matmul
    qk_name = f"layer{layeri}.qk"
    layer_mac_decode[qk_name] = 1 * seqlen * head_size * num_attention_heads * batchsize
    layer_load_act_numel_decode[qk_name] = (
        (seqlen + 1) * head_size * batchsize * num_attention_heads
    )
    layer_store_act_numel_decode[qk_name] = 1 * seqlen * batchsize * num_attention_heads

    layer_mac_prefill[qk_name] = (
        seqlen * seqlen * head_size * num_attention_heads * batchsize
    )
    layer_load_act_numel_prefill[qk_name] = (
        seqlen * head_size * batchsize * num_attention_heads * 2
    )
    layer_store_act_numel_prefill[qk_name] = (
        seqlen * seqlen * batchsize * num_attention_heads
    )

    # SV matmul
    sv_name = f"layer{layeri}.sv"
    layer_mac_decode[sv_name] = 1 * head_size * seqlen * num_attention_heads * batchsize
    layer_load_act_numel_decode[sv_name] = (
        seqlen * head_size * batchsize * num_attention_heads
        + 1 * seqlen * batchsize * num_attention_heads
    )
    layer_store_act_numel_decode[sv_name] = (
        1 * head_size * batchsize * num_attention_heads
    )

    layer_mac_prefill[sv_name] = (
        seqlen * head_size * seqlen * num_attention_heads * batchsize
    )
    layer_load_act_numel_prefill[sv_name] = (
        seqlen * head_size * batchsize * num_attention_heads
        + seqlen * seqlen * batchsize * num_attention_heads
    )
    layer_store_act_numel_prefill[sv_name] = (
        seqlen * head_size * batchsize * num_attention_heads
    )

    # Softmax
    layer_name = f"layer{layeri}.atten.softmax"
    layer_load_act_numel_decode[layer_name] = batchsize * head_size * seqlen * 1
    layer_store_act_numel_decode[layer_name] = batchsize * head_size * seqlen * 1
    layer_load_act_numel_prefill[layer_name] = batchsize * head_size * seqlen * seqlen
    layer_store_act_numel_prefill[layer_name] = batchsize * head_size * seqlen * seqlen

    # Residual Addition
    for name in ["atten.add", "mlp.add"]:
        layer_name = f"layer{layeri}.{name}"
        layer_load_act_numel_decode[layer_name] = batchsize * hidden_size * 2
        layer_store_act_numel_decode[layer_name] = batchsize * hidden_size
        layer_load_act_numel_prefill[layer_name] = batchsize * hidden_size * seqlen * 2
        layer_store_act_numel_prefill[layer_name] = batchsize * hidden_size * seqlen

    # MLP activation
    layer_name = f"layer{layeri}.mlp.act"
    layer_load_act_numel_decode[layer_name] = batchsize * num_mlp_act_size
    layer_store_act_numel_decode[layer_name] = batchsize * num_mlp_act_size
    layer_load_act_numel_prefill[layer_name] = batchsize * num_mlp_act_size * seqlen
    layer_store_act_numel_prefill[layer_name] = batchsize * num_mlp_act_size * seqlen


# for workspace

# calculate total
total_mac_decode = np.sum(list(layer_mac_decode.values()))
total_mac_prefill = np.sum(list(layer_mac_prefill.values()))
total_weight = np.sum(list(layer_weight_numel.values()))
total_kv_cache = np.sum(list(kv_cache_numel.values()))
total_load_act_numel_decode = np.sum(list(layer_load_act_numel_decode.values()))
total_store_act_numel_decode = np.sum(list(layer_store_act_numel_decode.values()))
total_load_numel_prefill = np.sum(list(layer_load_act_numel_prefill.values()))
total_store_numel_prefill = np.sum(list(layer_store_act_numel_prefill.values()))


print(f"decode total MAC(multiply-accumulation): {total_mac_decode}")
print(f"prefill total MAC(multiply-accumulation): {total_mac_prefill}")
print(f"total weight: {total_weight}")
print(f"total kv_cache: {total_kv_cache}")
print(f"decode total load numel: {total_load_act_numel_decode}")
print(f"decode total store numel: {total_store_act_numel_decode}")
print(f"prefill total load numel: {total_load_numel_prefill}")
print(f"prefill total store numel: {total_store_numel_prefill}")

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


memory_access_decode = (
    (total_load_act_numel_decode + total_store_act_numel_decode) * a_bit
    + total_weight * w_bit
) / 8

memory_access_prefill = (
    (total_load_numel_prefill + total_store_numel_prefill) * a_bit
    + total_weight * w_bit
) / 8

from hardwares.hardware_params import hardware_params

def roofline_analyze(bandwidth, max_OPS, OPs, memory_access):
    # bandwidth is bytes/s
    # memory_access in bit
    # x axis is OPS/byte
    # y axis is OPS/s
    y_max = max_OPS
    memory_access_bytes=memory_access/8
    turning_point = y_max / bandwidth
    arithmetic_intensity=OPs/memory_access_bytes
    if arithmetic_intensity < turning_point:
        bound="memory"
        performance=arithmetic_intensity*bandwidth
    else:
        bound="compute"
        performance=y_max
    return arithmetic_intensity,performance,bound

def str_number(num):
    if num > 1e12:
        return f"{num/1e12:.0f}T"
    elif num > 1e9:
        return f"{num/1e9:.0f}G"
    elif num > 1e6:
        return f"{num/1e6:.0f}M"
    elif num > 1e3:
        return f"{num/1e3:.0f}K"
    elif num<10:
        return f"{num:.2f}"
    else:
        return f"{num:.0f}"

bandwidth = hardware_params[args.hardware]["bandwith"]
max_OPS = hardware_params[args.hardware]["FP16"]
decode_file_name=f"{save_path}_decode_roofline.csv"
prefill_file_name=f"{save_path}_prefill_roofline.csv"
for layer in config.get_transformer_layers(model)[:1]:
    for name, module in layer.named_modules():
        # for linear layers
        if isinstance(module, nn.Linear):
            # for decode
            OPs = module.weight.numel() * batchsize *2
            load_act = module.in_features * batchsize
            store_act = module.out_features * batchsize
            memory_access = (load_act + store_act) * a_bit + module.weight.numel() * w_bit
            arithmetic_intensity,performance,bound=roofline_analyze(bandwidth, max_OPS, OPs, memory_access)
            with open(decode_file_name, "a+") as f:
                f.write(f"{name},{str_number(OPs)},{str_number(memory_access/8)},{str_number(arithmetic_intensity)},{str_number(performance)},{bound}\n")
            # for prefill
            OPs=module.weight.numel() * batchsize * seqlen * 2
            load_act = module.in_features * batchsize * seqlen
            store_act = module.out_features * batchsize * seqlen
            memory_access = (load_act + store_act) * a_bit + module.weight.numel() * w_bit
            arithmetic_intensity,performance,bound=roofline_analyze(bandwidth, max_OPS, OPs, memory_access)
            with open(prefill_file_name, "a+") as f:
                f.write(f"{name},{str_number(OPs)},{str_number(memory_access/8)},{str_number(arithmetic_intensity)},{str_number(performance)},{bound}\n")
    # for attention
    head_size = hidden_size // num_attention_heads
    # for decode
    name=f"qk_matmul"
    OPs = 1 * seqlen * head_size * num_attention_heads * batchsize *2
    load_act = (seqlen + 1) * head_size * batchsize * num_attention_heads
    store_act = 1 * seqlen * batchsize * num_attention_heads
    memory_access = (load_act + store_act) * a_bit
    arithmetic_intensity,performance,bound=roofline_analyze(bandwidth, max_OPS, OPs, memory_access)
    with open(decode_file_name, "a+") as f:
        f.write(f"{name},{str_number(OPs)},{str_number(memory_access/8)},{str_number(arithmetic_intensity)},{str_number(performance)},{bound}\n")
    name=f"sv_matmul"
    OPs = 1 * head_size * seqlen * num_attention_heads * batchsize *2
    load_act = seqlen * head_size * batchsize * num_attention_heads + 1 * seqlen * batchsize * num_attention_heads
    store_act = 1 * head_size * batchsize * num_attention_heads
    memory_access = (load_act + store_act) * a_bit
    arithmetic_intensity,performance,bound=roofline_analyze(bandwidth, max_OPS, OPs, memory_access)
    with open(decode_file_name, "a+") as f:
        f.write(f"{name},{str_number(OPs)},{str_number(memory_access/8)},{str_number(arithmetic_intensity)},{str_number(performance)},{bound}\n")
    name=f"softmax"
    # max sub exp sum div
    OPs = batchsize * num_attention_heads * seqlen * 1 *5
    load_act = batchsize * num_attention_heads * seqlen * 1
    store_act = batchsize * num_attention_heads * seqlen * 1
    memory_access = (load_act + store_act) * a_bit
    arithmetic_intensity,performance,bound=roofline_analyze(bandwidth, max_OPS, OPs, memory_access)
    with open(decode_file_name, "a+") as f:
        f.write(f"{name},{str_number(OPs)},{str_number(memory_access/8)},{str_number(arithmetic_intensity)},{str_number(performance)},{bound}\n")

    name=f"norm"
    # sum sub pow sum div mul add
    OPs = batchsize * hidden_size * 1 * 7
    load_act = batchsize * hidden_size * 1
    store_act = batchsize * hidden_size * 1
    memory_access = (load_act + store_act) * a_bit
    arithmetic_intensity,performance,bound=roofline_analyze(bandwidth, max_OPS, OPs, memory_access)
    with open(decode_file_name, "a+") as f:
        f.write(f"{name},{str_number(OPs)},{str_number(memory_access/8)},{str_number(arithmetic_intensity)},{str_number(performance)},{bound}\n")
    
    name=f"add"
    # sum sub pow sum div mul add
    OPs = batchsize * hidden_size * 1 
    load_act = batchsize * hidden_size * 1
    store_act = batchsize * hidden_size * 1
    memory_access = (load_act + store_act) * a_bit
    arithmetic_intensity,performance,bound=roofline_analyze(bandwidth, max_OPS, OPs, memory_access)
    with open(decode_file_name, "a+") as f:
        f.write(f"{name},{str_number(OPs)},{str_number(memory_access/8)},{str_number(arithmetic_intensity)},{str_number(performance)},{bound}\n")
    
    # for prefill
    name=f"qk_matmul"
    OPs = seqlen * seqlen * head_size * num_attention_heads * batchsize *2
    load_act = seqlen * head_size * batchsize * num_attention_heads * 2
    store_act = seqlen * seqlen * batchsize * num_attention_heads
    memory_access = (load_act + store_act) * a_bit
    arithmetic_intensity,performance,bound=roofline_analyze(bandwidth, max_OPS, OPs, memory_access)
    with open(prefill_file_name, "a+") as f:
        f.write(f"{name},{str_number(OPs)},{str_number(memory_access/8)},{str_number(arithmetic_intensity)},{str_number(performance)},{bound}\n")
    name=f"sv_matmul"
    OPs = seqlen * head_size * seqlen * num_attention_heads * batchsize *2
    load_act = seqlen * head_size * batchsize * num_attention_heads + seqlen * seqlen * batchsize * num_attention_heads
    store_act = seqlen * head_size * batchsize * num_attention_heads
    memory_access = (load_act + store_act) * a_bit
    arithmetic_intensity,performance,bound=roofline_analyze(bandwidth, max_OPS, OPs, memory_access)
    with open(prefill_file_name, "a+") as f:
        f.write(f"{name},{str_number(OPs)},{str_number(memory_access/8)},{str_number(arithmetic_intensity)},{str_number(performance)},{bound}\n")
    name=f"softmax"
    OPs = batchsize * num_attention_heads * seqlen * seqlen *5
    load_act = batchsize * num_attention_heads * seqlen * seqlen
    store_act = batchsize * num_attention_heads * seqlen * seqlen
    memory_access = (load_act + store_act) * a_bit
    arithmetic_intensity,performance,bound=roofline_analyze(bandwidth, max_OPS, OPs, memory_access)
    with open(prefill_file_name, "a+") as f:
        f.write(f"{name},{str_number(OPs)},{str_number(memory_access/8)},{str_number(arithmetic_intensity)},{str_number(performance)},{bound}\n")
    name=f"norm"
    # sum sub pow sum div mul add
    OPs = batchsize * hidden_size * seqlen * 7
    load_act = batchsize * hidden_size * seqlen
    store_act = batchsize * hidden_size * seqlen
    memory_access = (load_act + store_act) * a_bit
    arithmetic_intensity,performance,bound=roofline_analyze(bandwidth, max_OPS, OPs, memory_access)
    with open(prefill_file_name, "a+") as f:
        f.write(f"{name},{str_number(OPs)},{str_number(memory_access/8)},{str_number(arithmetic_intensity)},{str_number(performance)},{bound}\n")
    
    name=f"add"
    OPs = batchsize * hidden_size * seqlen * 1
    load_act = batchsize * hidden_size * seqlen
    store_act = batchsize * hidden_size * seqlen
    memory_access = (load_act + store_act) * a_bit
    arithmetic_intensity,performance,bound=roofline_analyze(bandwidth, max_OPS, OPs, memory_access)
    with open(prefill_file_name, "a+") as f:
        f.write(f"{name},{str_number(OPs)},{str_number(memory_access/8)},{str_number(arithmetic_intensity)},{str_number(performance)},{bound}\n")
    
    
    

# import matplotlib.pyplot as plt
# # bandwidth, FP16 MAC

# def draw_roofline(bandwidth, max_mac_per_s):
#     # bandwidth is bytes/s
#     # x axis is mac/byte
#     # y axis is mac/s
#     y_max = max_mac_per_s
#     first_x_to_y_max = y_max / bandwidth

#     plt.plot(
#         [0, first_x_to_y_max, first_x_to_y_max * 3], [0, y_max, y_max], color="black"
#     )
#     plt.xlabel("Operation Intensity (MAC/Byte)")
#     plt.ylabel("Performance (MAC/s)")


# bandwidth = hardware_params[args.hardware]["bandwith"]
# max_mac_per_s = hardware_params[args.hardware]["FP16"]/2
# draw_roofline(bandwidth, max_mac_per_s)


# def draw_roofline_model(max_mac_per_s, mac, memory_access, name, color):
#     intensity = mac / memory_access
#     plt.vlines(intensity, 0, max_mac_per_s, colors=color, linestyles="dashed")

#     # rotate annotation
#     plt.annotate(
#         name,
#         xy=(intensity, max_mac_per_s / np.random.randint(2,5)),
#         color=color,
#         rotation=90,
#         # ha="center",
#         va="center",
#     )


# draw_roofline_model(
#     max_mac_per_s,
#     total_mac_decode,
#     memory_access_decode,
#     # f"decode \n(len={args.seqlen}, bs={args.batchsize})",
#     f"decode",
#     "red",
# )
# draw_roofline_model(
#     max_mac_per_s,
#     total_mac_prefill,
#     memory_access_prefill,
#     # f"prefill \n(len={args.seqlen}, bs={args.batchsize})",
#     f"prefill",
#     "green",
# )

# draw_roofline_model(
#     max_mac_per_s,
#     linear_mac_decode,
#     linear_mem_access_decode,
#     f"decode linear",
#     "black",
# )
# draw_roofline_model(
#     max_mac_per_s,
#     linear_mac_prefill,
#     linear_mem_access_prefill,
#     f"prefill linear",
#     "black",
# )
# qk_mac = 0
# qk_memory_access=0
# for layeri in range(num_hidden_layers):
#     qk_mac += layer_mac_decode[f"layer{layeri}.qk"]
#     qk_memory_access += (layer_load_act_numel_decode[f"layer{layeri}.qk"] + layer_store_act_numel_decode[f"layer{layeri}.qk"]) * a_bit
# draw_roofline_model(
#     max_mac_per_s,
#     qk_mac,
#     qk_memory_access,
#     f"qk",
#     "green",
# )
# kv_mac=0
# kv_memory_access=0
# for layeri in range(num_hidden_layers):
#     kv_mac += layer_mac_decode[f"layer{layeri}.sv"]
#     kv_memory_access += (layer_load_act_numel_decode[f"layer{layeri}.sv"] + layer_store_act_numel_decode[f"layer{layeri}.sv"]) * a_bit

# draw_roofline_model(
#     max_mac_per_s,
#     kv_mac,
#     kv_memory_access,
#     f"kv",
#     "green",
# )

# plt.plot()

# plt.savefig(f"{save_path}_roofline.png")
