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

# Split the hardware list by category (single source of truth in hardware_params):
# the LLM viewer (cloud serving) shows cloud devices, the VLA viewer (edge
# robotics) shows edge devices. Devices without a category default to cloud.
cloud_hardwares = [k for k, v in hardware_params.items() if v.get("category", "cloud") == "cloud"]
edge_hardwares = [k for k, v in hardware_params.items() if v.get("category") == "edge"]
# avaliable_hardwares is the LLM viewer's list -> cloud only.
avaliable_hardwares = cloud_hardwares
