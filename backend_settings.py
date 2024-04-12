from net_parsers.manual.Llama import get_llama_network_graph
from net_parsers.manual.chatglm3 import get_chatglm_network_graph
from analyzers.llm_analyzer import LLMAnalyzer
from hardwares.hardware_params import hardware_params

avaliable_model_ids_sources = {
    "meta-llama/Llama-2-7b-hf": (get_llama_network_graph, LLMAnalyzer),
    "meta-llama/Llama-2-13b-hf": (get_llama_network_graph, LLMAnalyzer),
    "meta-llama/Llama-2-70b-hf": (get_llama_network_graph, LLMAnalyzer),
    # "meta-llama/Llama-2-13b-hf": {get_llama_network_graph, LLMAnalyzer},
    # "meta-llama/Llama-2-70b-hf": {get_llama_network_graph, LLMAnalyzer},
    "THUDM/chatglm3-6b": {get_chatglm_network_graph, LLMAnalyzer},
    # "facebook/opt-125m": {"source": "huggingface"},
    # "facebook/opt-1.3b": {"source": "huggingface"},
    # "facebook/opt-2.7b": {"source": "huggingface"},
    # "facebook/opt-6.7b": {"source": "huggingface"},
    # "facebook/opt-30b": {"source": "huggingface"},
    # "facebook/opt-66b": {"source": "huggingface"},
    # "DiT-XL/2": {"source": "DiT"},
    # "DiT-XL/4": {"source": "DiT"},
}
avaliable_model_ids = [_ for _ in avaliable_model_ids_sources.keys()]
avaliable_hardwares = [_ for _ in hardware_params.keys()]
