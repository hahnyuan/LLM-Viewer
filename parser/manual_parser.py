"""
config_file

Example
Input("input")
Linear("q_proj", ["input"], {"out_features":hidden_size})
Linear("k_proj", ["input"], {"out_features":hidden_size/num_attention_heads*num_key_value_heads})
Linear("v_proj", ["input"], {"out_features":hidden_size/num_attention_heads*num_key_value_heads})
MatMul("qk_matmul", ["q_proj", "k_proj"])
Softmax("softmax", ["qk_matmul"])
MatMul("sv_matmul", ["softmax", "v_proj"])
Linear("out_proj", ["sv_matmul"], {"out_features":hidden_size})
Add("attn_add", ["input", "out_proj"])
Norm("mlp_norm", ["attn_add"])
Linear("gate_proj", ["mlp_norm"], {"out_features":intermediate_size})
Linear("up_proj", ["mlp_norm"], {"out_features":intermediate_size})
Activation("mlp_act", ["up_proj", "gate_proj"])

"""
import re
from graph.base_nodes import *

def manual_parse(config_file,params):
    with open(config_file) as f:
        s = f.read()
        locals().update(params)
        nodes=[]
        for l in s.split("\n"):
            if l.strip()=="":
                continue
            node=eval(l.strip())
            nodes.append(node)
    return nodes
