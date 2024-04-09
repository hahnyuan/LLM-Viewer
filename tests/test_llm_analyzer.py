from net_graph.module import Module,Node
from analyzers.llm_analyzer import LLMAnalyzer
from hardwares.roofline_model import get_roofline_model
from types import SimpleNamespace
from net_parsers.manual.Llama import get_llama_network_graph

def test_dummy_llm():
    params = {
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "intermediate_size": 512,
        "vocab_size": 1000,
        "num_hidden_layers": 1,
    }
    params_obj = SimpleNamespace(**params)

    network=get_llama_network_graph(None,params_obj)
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