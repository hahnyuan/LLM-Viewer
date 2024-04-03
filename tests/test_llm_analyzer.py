from graph.model import Model,Node
from analyzers.llm_analyzer import LLMAnalyzer
from hardwares.roofline_model import RooflineModel
from parser.manual_parser import manual_parse

def test_dummy_llm():
    params = {
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "intermediate_size": 512,
        "vocab_size": 1000,
    }
    nodes=manual_parse("configs/Llama.cfg",params)
    model=Model(nodes)

    # Print the graph
    hardware_model=RooflineModel(1,1)
    analyzer=LLMAnalyzer(model,hardware_model)
    rst=analyzer.analyze(256,1)
    print(rst)