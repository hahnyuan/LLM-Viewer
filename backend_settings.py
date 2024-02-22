from hardwares.hardware_params import hardware_params

avaliable_model_ids=[
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
]

avaliable_hardwares=[_ for _ in hardware_params.keys()]
