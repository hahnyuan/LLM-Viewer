from net_parsers.manual.Llama import LlamaParser
from net_parsers.manual.chatglm3 import Chatglm3Parser
from net_parsers.onnx.onnx_parser import OnnxParser
from analyzers.llm_analyzer import LLMAnalyzer
from analyzers.onnx_analyzer import OnnxAnalyzer
from hardwares.hardware_params import hardware_params

avaliable_model_ids_sources = {
    "meta-llama/Llama-2-7b-hf": (LlamaParser, LLMAnalyzer),
    "meta-llama/Llama-2-13b-hf": (LlamaParser, LLMAnalyzer),
    "meta-llama/Llama-2-70b-hf": (LlamaParser, LLMAnalyzer),
    # "meta-llama/Llama-2-13b-hf": {LlamaParser, LLMAnalyzer},
    # "meta-llama/Llama-2-70b-hf": {LlamaParser, LLMAnalyzer},
    "THUDM/chatglm3-6b": (Chatglm3Parser, LLMAnalyzer),
    # "facebook/opt-125m": {"source": "huggingface"},
    # "facebook/opt-1.3b": {"source": "huggingface"},
    # "facebook/opt-2.7b": {"source": "huggingface"},
    # "facebook/opt-6.7b": {"source": "huggingface"},
    # "facebook/opt-30b": {"source": "huggingface"},
    # "facebook/opt-66b": {"source": "huggingface"},
    # "DiT-XL/2": {"source": "DiT"},
    # "DiT-XL/4": {"source": "DiT"},
    "OnnxFile": (OnnxParser, OnnxAnalyzer)
}
avaliable_model_ids = [_ for _ in avaliable_model_ids_sources.keys()]
avaliable_hardwares = [_ for _ in hardware_params.keys()]
