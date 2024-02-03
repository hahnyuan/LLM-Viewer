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
w_byte = w_bit / 8
a_byte = a_bit / 8

model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
config = importlib.import_module(args.config_path.replace("/", ".").replace(".py", ""))
num_attention_heads = config.get_num_attention_heads(model_config)
hidden_size = config.get_hidden_size(model_config)
num_key_value_heads = config.get_num_key_value_heads(model_config)
num_hidden_layers = config.get_num_hidden_layers(model_config)

# # export to mac csv and weight csv, not use pandas
save_path = f"output/{model_id[:model_id.rfind('/')]}"
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_path += f"{model_id[model_id.rfind('/'):]}"


from hardwares.hardware_params import hardware_params
from roofline_model import roofline_analyze


def str_number(num):
    if num > 1e12:
        return f"{num/1e12:.0f}T"
    elif num > 1e9:
        return f"{num/1e9:.0f}G"
    elif num > 1e6:
        return f"{num/1e6:.0f}M"
    elif num > 1e3:
        return f"{num/1e3:.0f}K"
    elif num < 10:
        return f"{num:.2f}"
    else:
        return f"{num:.0f}"


bandwidth = hardware_params[args.hardware]["bandwith"]
max_OPS = hardware_params[args.hardware]["FP16"]


def write_csv(
    file_name,
    layer_name,
    OPs,
    load_weight,
    load_act,
    store_act,
    load_kv_cache,
    store_kv_cache,
):
    memory_access = load_weight + load_act + store_act + load_kv_cache + store_kv_cache
    arithmetic_intensity, performance, bound = roofline_analyze(
        bandwidth, max_OPS, OPs, memory_access
    )
    # use str_number
    with open(file_name, "a+") as f:
        f.write(
            f"{layer_name},{str_number(OPs)},{str_number(memory_access)},{str_number(arithmetic_intensity)},{str_number(performance)},{bound},{str_number(load_weight)},{str_number(load_act)},{str_number(store_act)},{str_number(load_kv_cache)},{str_number(store_kv_cache)}\n"
        )
    # not use str_number
    with open(file_name.replace(".csv", "_raw.csv"), "a+") as f:
        f.write(
            f"{layer_name},{OPs},{memory_access},{arithmetic_intensity},{performance},{bound},{load_weight},{load_act},{store_act},{load_kv_cache},{store_kv_cache}\n"
        )
    


decode_file_name = f"{save_path}_decode_roofline.csv"
prefill_file_name = f"{save_path}_prefill_roofline.csv"

for name, (ic,oc) in config.get_linear_layers(model_config).items():
    # for linear layers
    is_kv_proj = name in ["k_proj","v_proj"]
    is_normal_proj = not is_kv_proj
    write_csv(
        decode_file_name,
        name,
        OPs=ic*oc * batchsize * 2,
        load_weight=ic*oc * w_byte,
        load_act=ic * batchsize * a_byte,
        store_act=0 if is_kv_proj else oc * batchsize * a_byte,
        load_kv_cache=0,
        store_kv_cache=(
            0 if is_normal_proj else oc * batchsize * a_byte
        ),
    )
    # for prefill
    write_csv(
        prefill_file_name,
        name,
        OPs=ic*oc * batchsize * seqlen * 2,
        load_weight=ic*oc * w_byte,
        load_act=ic * batchsize * seqlen * a_byte,
        store_act=(
            0
            if is_kv_proj
            else oc * batchsize * seqlen * a_byte
        ),
        load_kv_cache=0,
        store_kv_cache=(
            0
            if is_normal_proj
            else oc * batchsize * seqlen * a_byte
        ),
    )

# for attention
head_size = hidden_size // num_attention_heads
# for decode
name = f"qk_matmul"
write_csv(
    decode_file_name,
    name,
    OPs=seqlen * head_size * num_attention_heads * batchsize * 2,
    load_weight=0,
    load_act=(1) * head_size * batchsize * num_attention_heads * a_byte,
    store_act=1 * seqlen * batchsize * num_attention_heads * a_byte,
    load_kv_cache=(seqlen) * head_size * batchsize * num_attention_heads * a_byte,
    store_kv_cache=0,
)
name = f"sv_matmul"
write_csv(
    decode_file_name,
    name,
    OPs=1 * head_size * seqlen * num_attention_heads * batchsize * 2,
    load_weight=0,
    load_act=(1 * seqlen * batchsize * num_attention_heads) * a_byte,
    store_act=1 * head_size * batchsize * num_attention_heads * a_byte,
    load_kv_cache=(seqlen * head_size * batchsize * num_attention_heads) * a_byte,
    store_kv_cache=0,
)

name = f"softmax"
# max sub exp sum div
write_csv(
    decode_file_name,
    name,
    OPs=batchsize * num_attention_heads * seqlen * 1 * 5,
    load_weight=0,
    load_act=batchsize * num_attention_heads * seqlen * 1 * a_byte,
    store_act=batchsize * num_attention_heads * seqlen * 1 * a_byte,
    load_kv_cache=0,
    store_kv_cache=0,
)

name = f"norm"
# sum sub pow sum div mul add
write_csv(
    decode_file_name,
    name,
    OPs=batchsize * hidden_size * 1 * 7,
    load_weight=0,
    load_act=batchsize * hidden_size * 1 * a_byte,
    store_act=batchsize * hidden_size * 1 * a_byte,
    load_kv_cache=0,
    store_kv_cache=0,
)

name = f"add"
write_csv(
    decode_file_name,
    name,
    OPs=batchsize * hidden_size * 1,
    load_weight=0,
    load_act=batchsize * hidden_size * 1 * a_byte,
    store_act=batchsize * hidden_size * 1 * a_byte,
    load_kv_cache=0,
    store_kv_cache=0,
)

# for prefill
name = f"qk_matmul"
write_csv(
    prefill_file_name,
    name,
    OPs=seqlen * seqlen * head_size * num_attention_heads * batchsize * 2,
    load_weight=0,
    load_act=0,
    store_act=seqlen * seqlen * batchsize * num_attention_heads * a_byte,
    load_kv_cache=seqlen * head_size * batchsize * num_attention_heads * 2 * a_byte,
    store_kv_cache=0,
)
name = f"sv_matmul"
write_csv(
    prefill_file_name,
    name,
    OPs=seqlen * head_size * seqlen * num_attention_heads * batchsize * 2,
    load_weight=0,
    load_act=seqlen * seqlen * batchsize * num_attention_heads * a_byte,
    store_act=seqlen * head_size * batchsize * num_attention_heads * a_byte,
    load_kv_cache=seqlen * head_size * batchsize * num_attention_heads * a_byte,
    store_kv_cache=0,
)
name = f"softmax"
write_csv(
    prefill_file_name,
    name,
    OPs=batchsize * num_attention_heads * seqlen * seqlen * 5,
    load_weight=0,
    load_act=batchsize * num_attention_heads * seqlen * seqlen * a_byte,
    store_act=batchsize * num_attention_heads * seqlen * seqlen * a_byte,
    load_kv_cache=0,
    store_kv_cache=0,
)
name = f"norm"
write_csv(
    prefill_file_name,
    name,
    OPs=batchsize * hidden_size * seqlen * 7,
    load_weight=0,
    load_act=batchsize * hidden_size * seqlen * a_byte,
    store_act=batchsize * hidden_size * seqlen * a_byte,
    load_kv_cache=0,
    store_kv_cache=0,
)
name = f"add"
write_csv(
    prefill_file_name,
    name,
    OPs=batchsize * hidden_size * seqlen * 1,
    load_weight=0,
    load_act=batchsize * hidden_size * seqlen * a_byte,
    store_act=batchsize * hidden_size * seqlen * a_byte,
    load_kv_cache=0,
    store_kv_cache=0,
)
