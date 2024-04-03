"""
config_file

Example
Input("input")
Linear("q_proj", ["input"], {"out_features":hidden_size})
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
