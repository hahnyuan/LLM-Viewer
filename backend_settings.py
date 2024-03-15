from hardwares.hardware_params import hardware_params

avaliable_model_ids = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "THUDM/chatglm3-6b",
    "facebook/opt-125m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-30b",
    "facebook/opt-66b",
]

avaliable_hardwares = [_ for _ in hardware_params.keys()]
