# data source https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units
# Nvidia Ampere https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf
# https://en.wikipedia.org/wiki/Ampere_(microarchitecture)
# Hopper https://www.nvidia.com/en-us/data-center/h100/
# the OPS = sparse OPS/2

hardware_params = {
    "nvidia_V100": {"bandwith": 900e9, "FP16": 112e12, "INT8": 62e12},
    "nvidia_A6000": {"bandwith": 768e9, "FP16": 309.677e12 / 2, "INT8": 309.7e12},
    "nvidia_A100_40G": {"bandwith": 1555e9, "FP16": 312e12, "INT8": 624e12},
    "nvidia_A100_80G": {"bandwith": 2039e9, "FP16": 312e12, "INT8": 624e12},
    "nvidia_H100_SXM": {"bandwith": 3072e9, "FP16": 1979e12, "INT8": 3958e12},
}
