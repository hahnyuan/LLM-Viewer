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
parser.add_argument("--w_bit", type=int, default=16, help="weight bitwidth")
parser.add_argument("--a_bit", type=int, default=16, help="temporary activation bitwidth")
parser.add_argument("--kv_bit", type=int, default=16, help="kv cache bitwidth")
parser.add_argument("--use_flashattention", action="store_true", help="use flash attention")
parser.add_argument("--flashattention_kv_block_size", type=int, default=128, help="flash attention kv block size")
args = parser.parse_args()

analyzer=ModelAnalyzer(args.model_id,args.hardware,args.config_file)
if not args.use_flashattention:
    args.flashattention_kv_block_size=None
results=analyzer.analyze(batchsize=args.batchsize,seqlen=args.seqlen,w_bit=args.w_bit,a_bit=args.a_bit,kv_bit=args.kv_bit,flashattention_kv_block_size=args.flashattention_kv_block_size)
analyzer.save_csv()