# the OPS = sparse OPS/2

hardware_params = {
    # https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
    # NOTICE: V100 not support INT8 in tensor core, so INT8 performance is not good
    "nvidia_V100": {"bandwith": 900e9, "FP16": 112e12, "INT8": 62e12, "onchip_buffer": 20480e3}, 
    # https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
    "nvidia_A6000": {"bandwith": 768e9, "FP16": 309.677e12 / 2, "INT8": 309.7e12, "onchip_buffer": 21504e3}, 
    # https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
    "nvidia_A100": {"bandwith": 1555e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3}, # use 40G data
    "nvidia_A100_40G": {"bandwith": 1555e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3},
    "nvidia_A100_80G": {"bandwith": 2039e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3},
    # https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper
    "nvidia_H100": {"bandwith": 3072e9, "FP16": 1979e12/2, "INT8": 3958e12/2, "onchip_buffer": 33792e3}, # use SXM data
    "nvidia_H100_SXM": {"bandwith": 3072e9, "FP16": 1979e12/2, "INT8": 3958e12/2, "onchip_buffer": 33792e3},
    "nvidia_H100_PCIe": {"bandwith": 2048e9, "FP16": 1513e12/2, "INT8": 3026e12/2, "onchip_buffer": 29184e3},
}
