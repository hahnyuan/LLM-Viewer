# Peak-throughput note: the OPS values are DENSE = sparse_OPS / 2.
# Units: bandwidth in bytes/s; FP16/FP8/INT8/INT4 in OP/s; onchip_buffer and
# memory_capacity in bytes. For NVIDIA GPUs onchip_buffer is the aggregate
# register-file size (per-SM RF * num_SMs), used as the flash-attention SRAM.
# memory_capacity is the device DRAM (binary GB, i.e. GiB) and gates the
# "does the model fit" verdict; it does not affect roofline throughput.
#
# `category` ("cloud" | "edge") is the single source of truth for which device
# list a UI shows: the LLM viewer (cloud serving) lists "cloud" hardware, the
# VLA viewer (edge robotics) lists "edge" hardware. The analyzer ignores it.

GiB = 1024 ** 3

hardware_params = {
    # NOTICES: For GPU, we use Register File Size as on-chip buffer size
    # https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
    # NOTICE: V100 not support INT8 in tensor core, so INT8 performance is not good
    "nvidia_V100": {"bandwidth": 900e9, "FP16": 112e12, "INT8": 62e12, "onchip_buffer": 20480e3, "memory_capacity": 32 * GiB, "category": "cloud"},
    # https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf
    "nvidia_A6000": {"bandwidth": 768e9, "FP16": 154.8e12, "INT8": 309.7e12, "INT4": 619.4e12, "onchip_buffer": 21504e3, "memory_capacity": 48 * GiB, "category": "cloud"},
    # https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf
    "nvidia_A6000_Ada": {"bandwidth": 960e9, "FP16": 364.2e12, "INT8": 728.5e12, "INT4": 1457e12, "onchip_buffer": 36352e3, "memory_capacity": 48 * GiB, "category": "cloud"},
    # https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
    # Ampere's SM has 256KB RF, max 164KB Shared Mem
    "nvidia_A100": {"bandwidth": 1555e9, "FP16": 312e12, "INT8": 624e12, "INT4": 1248e12, "onchip_buffer": 27648e3, "memory_capacity": 40 * GiB, "category": "cloud"},  # use 40G data
    "nvidia_A100_40G": {"bandwidth": 1555e9, "FP16": 312e12, "INT8": 624e12, "INT4": 1248e12, "onchip_buffer": 27648e3, "memory_capacity": 40 * GiB, "category": "cloud"},
    "nvidia_A100_80G": {"bandwidth": 2039e9, "FP16": 312e12, "INT8": 624e12, "INT4": 1248e12, "onchip_buffer": 27648e3, "memory_capacity": 80 * GiB, "category": "cloud"},
    "nvidia_A800_80G_SXM": {"bandwidth": 2039e9, "FP16": 312e12, "INT8": 624e12, "INT4": 1248e12, "onchip_buffer": 27648e3, "memory_capacity": 80 * GiB, "category": "cloud"},
    "nvidia_A40": {"bandwidth": 696e9, "FP16": 149.7e12, "INT8": 299.3e12, "INT4": 598.6e12, "onchip_buffer": 21504e3, "memory_capacity": 48 * GiB, "category": "cloud"},
    # https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper
    # Hopper tensor cores add FP8 but drop INT4.
    "nvidia_H100": {
        "bandwidth": 3072e9,
        "FP16": 1979e12 / 2,
        "INT8": 3958e12 / 2,
        "onchip_buffer": 33792e3,
        "memory_capacity": 80 * GiB,
        "category": "cloud",
    },  # use SXM data
    "nvidia_H100_SXM": {"bandwidth": 3072e9, "FP16": 1979e12 / 2, "INT8": 3958e12 / 2, "onchip_buffer": 33792e3, "memory_capacity": 80 * GiB, "category": "cloud"},
    "nvidia_H100_PCIe": {"bandwidth": 2048e9, "FP16": 1513e12 / 2, "INT8": 3026e12 / 2, "onchip_buffer": 29184e3, "memory_capacity": 80 * GiB, "category": "cloud"},
    # https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf
    # Ada SM has 256 KB Register File, and 128 KB of L1/Shared Memory
    "nvidia_L40": {"bandwidth": 864e9, "FP16": 181e12, "INT8": 362e12, "INT4": 724e12, "onchip_buffer": 36352e3, "memory_capacity": 48 * GiB, "category": "cloud"},
    # GeForce RTX 4090 (Ada AD102, 128 SMs). Tensor-core dense figures from the
    # Ada whitepaper (sparse/2): FP16 330.3, INT8 660.6, INT4 1321.2. 24GB
    # GDDR6X at 1008 GB/s (384-bit, 21 Gbps). onchip = 128 * 256KB RF.
    "nvidia_RTX4090": {"bandwidth": 1008e9, "FP16": 330.3e12, "INT8": 660.6e12, "INT4": 1321.2e12, "onchip_buffer": 32768e3, "memory_capacity": 24 * GiB, "category": "edge"},
    # GeForce RTX 3090 (Ampere GA102, 82 SMs). Dense tensor figures (sparse/2),
    # FP16 with FP16 accumulate: FP16 142, INT8 284, INT4 568. 24GB GDDR6X at
    # 936 GB/s (384-bit, 19.5 Gbps). onchip = 82 * 256KB RF.
    "nvidia_RTX3090": {"bandwidth": 936e9, "FP16": 142e12, "INT8": 284e12, "INT4": 568e12, "onchip_buffer": 20992e3, "memory_capacity": 24 * GiB, "category": "edge"},
    # GeForce RTX 5090 (Blackwell GB202, 170 SMs). Dense tensor figures derived
    # from the FP4 headline (sparse/2): FP16 419, INT8 838, INT4 1676. 32GB
    # GDDR7 at 1792 GB/s (512-bit, 28 Gbps). onchip = 170 * 256KB RF.
    "nvidia_RTX5090": {"bandwidth": 1792e9, "FP16": 419e12, "INT8": 838e12, "INT4": 1676e12, "onchip_buffer": 43520e3, "memory_capacity": 32 * GiB, "category": "edge"},
    # Intel Skylake-X (Skylake-X, Cascade Lake) Intel Xeon Phi (Knights Landing, Knights Mill) Intel Ice Lake, Tiger Lake and Rocket Lake
    # support AVX-512 & FMA (512-bit), they has throughput of 1 cycle
    # https://www.intel.com/content/www/us/en/products/sku/230496/intel-core-i913900k-processor-36m-cache-up-to-5-80-ghz/specifications.html
    "intel_13900k": {"bandwidth": 89.6e9, "FP16": 8 * 5.4e9 * (512 / 16), "onchip_buffer": 36e6, "memory_capacity": 128 * GiB, "category": "cloud"},  # DRAM is configurable; default assumption

    # === Edge robotics platforms (VLA inference targets) ===================
    # NVIDIA Jetson. AI-perf headline figures are sparse INT8 TOPS; dense =
    # sparse/2, and tensor-core FP16 = INT8/2, INT4 = INT8*2 (Ampere). onchip
    # buffer = num_SMs * 256KB register file. memory is unified LPDDR (capacity
    # also bounds the model, since weights+KV+activations share it with the OS).
    # Jetson Orin NX 16GB: 1024 Ampere CUDA cores (8 SMs), 100 sparse INT8 TOPS,
    # 102.4 GB/s LPDDR5. https://developer.nvidia.com/embedded/jetson-modules
    "jetson_orin_nx_16gb": {"bandwidth": 102.4e9, "FP16": 25e12, "INT8": 50e12, "INT4": 100e12, "onchip_buffer": 8 * 256e3, "memory_capacity": 16 * GiB, "category": "edge"},
    # Jetson AGX Orin 64GB: 2048 Ampere CUDA cores (16 SMs), 275 sparse INT8
    # TOPS, 204.8 GB/s LPDDR5.
    "jetson_agx_orin_64gb": {"bandwidth": 204.8e9, "FP16": 68.75e12, "INT8": 137.5e12, "INT4": 275e12, "onchip_buffer": 16 * 256e3, "memory_capacity": 64 * GiB, "category": "edge"},
    # Jetson AGX Thor (Blackwell): ~2070 sparse FP4 TFLOPS, 128GB LPDDR5X,
    # ~273 GB/s. PRELIMINARY -- dense FP16/INT8/INT4 derived from the FP4
    # headline; verify against the final datasheet before trusting absolutes.
    "jetson_agx_thor": {"bandwidth": 273e9, "FP16": 258e12, "INT8": 517e12, "INT4": 1035e12, "onchip_buffer": 20 * 256e3, "memory_capacity": 128 * GiB, "category": "edge"},
}
