from model_analyzer import ModelAnalyzer
import torch.nn as nn
import numpy as np
import os
import importlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_id", type=str, help="model id")
parser.add_argument(
    "hardware",
    type=str,
    help="name of hardware, for example nvidia_V100 or nvidia_A6000",
)
parser.add_argument("--config_file", type=str, default=None, help="config file")
parser.add_argument("--batchsize", type=int, default=1, help="batch size")
parser.add_argument("--seqlen", type=int, default=1024, help="sequence length")
parser.add_argument("--promptlen", type=int, default=128, help="prompt sequence length")
parser.add_argument("--w_bit", type=int, default=16, help="weight bitwidth")
parser.add_argument("--a_bit", type=int, default=16, help="temporary activation bitwidth")
parser.add_argument("--kv_bit", type=int, default=16, help="kv cache bitwidth")
parser.add_argument("--use_flashattention", action="store_true", help="use flash attention")
args = parser.parse_args()


analyzer=ModelAnalyzer(args.model_id,args.hardware,args.config_file)
ret = analyzer.analyze_generate_task(args.promptlen, args.seqlen, args.batchsize, args.w_bit, args.a_bit, args.kv_bit, args.use_flashattention)
elapse = ret["inference_time"]
prefill_elapse = ret["prefill_time"]
print(f"{args.hardware}: 1st token latency {prefill_elapse}, total latency {elapse}, throughput {args.seqlen * args.batchsize / elapse} Token/sec")
