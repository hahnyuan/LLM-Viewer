from hardwares.roofline_model import get_roofline_model
from backend_settings import avaliable_model_ids_sources
import pandas as pd

import torch.nn as nn
import numpy as np
import os
import importlib
import argparse
import inspect
from prettytable import PrettyTable


parser = argparse.ArgumentParser()
parser.add_argument("model_id", type=str, help="model id")
parser.add_argument(
    "hardware",
    type=str,
    help="name of hardware, for example nvidia_V100 or nvidia_A6000",
)
parser.add_argument(
    "--source",
    type=str,
    default="huggingface",
    help="source of model, if not huggingface, will use local model in model_params.<source>",
)
parser.add_argument("--config_file", type=str, default=None, help="config file")
parser.add_argument("--batchsize", type=int, default=1, help="batch size")
parser.add_argument("--stage", default="decode", choices=["decode", "prefill", "chat"])
parser.add_argument("--seqlen", type=int, default=1024, help="sequence length")
parser.add_argument("--w_bit", type=int, default=16, help="weight bitwidth")
parser.add_argument(
    "--a_bit", type=int, default=16, help="temporary activation bitwidth"
)
parser.add_argument("--kv_bit", type=int, default=16, help="kv cache bitwidth")
parser.add_argument("--n_parallel_decode", type=int, default=1, help="number of parallel decodes")
parser.add_argument(
    "--use_flashattention", action="store_true", help="use flash attention"
)
parser.add_argument("--compute_dtype", type=str, default="FP16", help="compute dtype",choices=["FP16","INT8"])
parser.add_argument("--save_csv_path", type=str, default="output/results.csv", help="save csv path")
# for stable_diffusion
parser.add_argument(
    "--latent_size", type=int, default=64, help="latent_size for stable_diffusion"
)
parser.add_argument(
    "--time_steps", type=int, default=20, help="time_steps for stable_diffusion"
)
# for onnx
parser.add_argument("--model_path", type=str, default=None, help="onnx model path")
parser.add_argument("--input_shape_info", type=str, default=None, 
    help="input info for dynamic input onnx model. example: data:1,3,224,224"
)

args = parser.parse_args()


parser_cls, analyzer_cls = avaliable_model_ids_sources[args.model_id]
network = parser_cls(args.model_id, vars(args)).parse()
hardware_model=get_roofline_model(args.hardware)
analyzer=analyzer_cls(network,hardware_model)


def get_parameters_in_dict(func, param_args):
    signature = inspect.signature(func)
    parameters = signature.parameters

    parms = {param:getattr(param_args, param) for param in parameters if param in param_args}
    return parms
print(get_parameters_in_dict(analyzer.analyze, args))

results = analyzer.analyze(
    **get_parameters_in_dict(analyzer.analyze, args)
)

layer_result=results["layers"]
network_result=results["network"]

"""
layer_result:
{
    "layer_name1": (node, {
            "OPs": "",
            "memory_access": "",
            "arithmetic_intensity": "",
            "performance": "",
            "bound": "",
            "load_weight": "",
            "load_act": "",
            "store_act": "",
            "load_kv_cache": "",
            "store_kv_cache": "",
            "inference_time": "",
            ...
    })
    "layer_name2": ...,
    ...
}
network_result:
{
    "OPs": ...,
    "memory_access": ...,
    "inference_time": ...,
}
"""
def print_df_to_table(df):
    table = PrettyTable()
    table.field_names = ["Layer"] + list(df.columns)

    for index, row in df.iterrows():
        table.add_row([index] + list(row))
    print(table)

# transform to pandas
all_keys=set()
for layer_name,layer in layer_result.items():
    if isinstance(layer, dict):
        all_keys.update(layer.keys())
    else:
        all_keys.update(layer[1].keys())
all_keys=list(all_keys)
layer_df=pd.DataFrame(columns=all_keys)
for layer_name,layer in layer_result.items():
    # set row name as layer_name
    if isinstance(layer, dict):
        layer_df=layer_df.append(pd.Series(layer,name=layer_name))
    else:
        layer_df=layer_df.append(pd.Series(layer[1],name=layer_name))

print_df_to_table(layer_df)

# layer_df.index.name="name"

network_df=pd.DataFrame(columns=network_result.keys())

network_df=network_df.append(pd.Series(network_result,name="network"))

print_df_to_table(network_df)

save_csv_dir=os.path.dirname(args.save_csv_path)
if not os.path.exists(save_csv_dir):
    os.makedirs(save_csv_dir)

with open(args.save_csv_path,"a+") as f:
    f.write(f"\n=======\n\n{args.model_id}\n{args}\n\n")

# append to csv file
layer_df.to_csv(args.save_csv_path, mode='a+', header=True)

with open(args.save_csv_path,"a+") as f:
    f.write(f"\n")
network_df.to_csv(args.save_csv_path, mode='a+', header=True)
