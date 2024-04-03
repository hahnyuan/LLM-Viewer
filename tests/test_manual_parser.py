from graph.base_nodes import *
from parser.manual_parser import manual_parse

def test_parse():
    params = {
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "intermediate_size": 512
    }
    nodes=manual_parse("configs/Llama.cfg",params)
    print(nodes)