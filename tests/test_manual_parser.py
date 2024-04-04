from graph.base_nodes import *
from parser.manual_parser import manual_parse_network
from types import SimpleNamespace

def test_parse():
    params = {
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "intermediate_size": 512,
        "vocab_size": 1000,
    }
    params_obj = SimpleNamespace(**params)

    network=manual_parse_network("configs/manual/example.py",params_obj)
    print(network)