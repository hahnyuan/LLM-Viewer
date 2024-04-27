from hardwares.hardware_params import hardware_params

avaliable_model_ids_sources = {
    "meta-llama/Llama-2-7b-hf": {"source": "huggingface"},
    "meta-llama/Llama-2-13b-hf": {"source": "huggingface"},
    "meta-llama/Llama-2-70b-hf": {"source": "huggingface"},
    "EleutherAI/gpt-j-6B":{"source": "huggingface"},
    "THUDM/chatglm3-6b": {"source": "huggingface"},
    "facebook/opt-125m": {"source": "huggingface"},
    "facebook/opt-1.3b": {"source": "huggingface"},
    "facebook/opt-2.7b": {"source": "huggingface"},
    "facebook/opt-6.7b": {"source": "huggingface"},
    "facebook/opt-30b": {"source": "huggingface"},
    "facebook/opt-66b": {"source": "huggingface"},
    # "DiT-XL/2": {"source": "DiT"},
    # "DiT-XL/4": {"source": "DiT"},
}
avaliable_model_ids = [_ for _ in avaliable_model_ids_sources.keys()]
avaliable_hardwares = [_ for _ in hardware_params.keys()]
