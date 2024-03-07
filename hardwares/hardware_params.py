# the OPS = sparse OPS/2

hardware_params = {
    # https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
    # NOTICE: V100 not support INT8 in tensor core, so INT8 performance is not good
    "nvidia_V100": {"bandwidth": 900e9, "FP16": 112e12, "INT8": 62e12, "onchip_buffer": 20480e3}, 
    # https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
    "nvidia_A6000": {"bandwidth": 768e9, "FP16": 309.677e12 / 2, "INT8": 309.7e12, "onchip_buffer": 21504e3}, 
    # https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
    # Ampere's SM has 256KB RF, max 164KB Shared Mem
    "nvidia_A100": {"bandwidth": 1555e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3}, # use 40G data
    "nvidia_A100_40G": {"bandwidth": 1555e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3},
    "nvidia_A100_80G": {"bandwidth": 2039e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3},
    "nvidia_A800_80G_SXM":{"bandwidth":2039e9, "FP16":312e12, "INT8":624e12,"onchip_buffer":27648e3},
    "nvidia_A40": {"bandwidth": 696e9, "FP16": 149.7e12/2, "INT8": 299.3e12/2, "onchip_buffer": 21504e3},
    # https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper
    "nvidia_H100": {"bandwidth": 3072e9, "FP16": 1979e12/2, "INT8": 3958e12/2, "onchip_buffer": 33792e3}, # use SXM data
    "nvidia_H100_SXM": {"bandwidth": 3072e9, "FP16": 1979e12/2, "INT8": 3958e12/2, "onchip_buffer": 33792e3},
    "nvidia_H100_PCIe": {"bandwidth": 2048e9, "FP16": 1513e12/2, "INT8": 3026e12/2, "onchip_buffer": 29184e3},
    # https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf
    # Ada SM has 256 KB Register File, and 128 KB of L1/Shared Memory
    "nvidia_L40": {"bandwidth": 864e9, "FP16": 181e12, "INT8": 362e12, "onchip_buffer": 36352e3},
    # Intel Skylake-X (Skylake-X, Cascade Lake) Intel Xeon Phi (Knights Landing, Knights Mill) Intel Ice Lake, Tiger Lake and Rocket Lake
    # support AVX-512 & FMA (512-bit), they has throughput of 1 cycle
    # https://www.intel.com/content/www/us/en/products/sku/230496/intel-core-i913900k-processor-36m-cache-up-to-5-80-ghz/specifications.html
    "intel_13900k":{"bandwidth":89.6e9, "FP16":8*5.4e9*(512/16),"onchip_buffer": 36e6}
    
}
