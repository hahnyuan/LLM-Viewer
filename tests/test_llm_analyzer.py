from graph.module import Module,Node
from analyzers.llm_analyzer import LLMAnalyzer
from hardwares.roofline_model import RooflineModel
from parser.manual_parser import manual_parse_network
from types import SimpleNamespace

def test_dummy_llm():
    params = {
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "intermediate_size": 512,
        "vocab_size": 1000,
        "num_hidden_layers": 12,
    }
    params_obj = SimpleNamespace(**params)

    network=manual_parse_network("configs/manual/Llama.py",params_obj)

    # Print the graph
    hardware_model=RooflineModel(1,1)
    analyzer=LLMAnalyzer(network,hardware_model)
    rst=analyzer.analyze(256,1)
    print(rst)

    rst=analyzer.analyze(1024,5)
    print(rst)