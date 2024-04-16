from net_graph.module import Module,Node
from analyzers.llm_analyzer import LLMAnalyzer
from hardwares.roofline_model import get_roofline_model
from types import SimpleNamespace
from net_parsers.manual.llm_parser import LLMParser

def test_dummy_llm():
    network=LLMParser("meta-llama/Llama-2-7b-hf", {}).parse()
    hardware_model=get_roofline_model("nvidia_A6000")
    analyzer=LLMAnalyzer(network,hardware_model)
    rst=analyzer.analyze(256,1)
    network_graph, module_graphs=analyzer.get_ui_graph(rst)
    print(rst)

def test_dummy_manual_llm():
    params = {
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "intermediate_size": 512,
        "vocab_size": 1000,
        "num_hidden_layers": 2,
    }
    network=LLMParser(None, params).parse()
    hardware_model=get_roofline_model("nvidia_A6000")
    analyzer=LLMAnalyzer(network,hardware_model)
    rst=analyzer.analyze(256,1)
    print(rst)

    rst=analyzer.analyze(1024,5)
    print(rst)

    rst=analyzer.analyze(1024,5,stage="prefill")
    print(rst)

    rst=analyzer.analyze(1024,5,w_bit=4,a_bit=2,kv_bit=6)
    print(rst)

    rst=analyzer.analyze(1024,5,use_flashattention=True)
    print(rst)

    rst=analyzer.analyze(1024,5,n_parallel_decode=16)
    print(rst)

    rst=analyzer.analyze(1024,5,compute_dtype="INT8")
    print(rst)