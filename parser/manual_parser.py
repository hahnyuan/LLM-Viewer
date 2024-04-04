"""
config_file

Example
Input("input")
Linear("q_proj", ["input"], {"out_features":hidden_size})
"""
import re
from graph.base_nodes import *
import importlib

def manual_parse_network(config_file,params):
    path_to_module_name = re.sub(r"(.*)\.py", r"\1", config_file)
    path_to_module_name = re.sub(r"/", r".", path_to_module_name)
    path_to_module_name = re.sub(r"\\", r".", path_to_module_name)
    module=importlib.import_module(path_to_module_name)
    network=module.get_network_graph(params)
    return network
