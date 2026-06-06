def evaluate_op(OPs, load_weight, load_act, store_act, load_kv_cache, store_kv_cache, bandwidth, max_OPS):
    """Apply the roofline model to one op's compute/memory breakdown.

    Returns the full per-op result dict (raw byte fields + derived
    arithmetic_intensity / performance / bound / inference_time). Shared by the
    LLM analyzer and the vision-encoder / VLA analyzers so every op is costed
    identically.
    """
    memory_access = load_weight + load_act + store_act + load_kv_cache + store_kv_cache
    arithmetic_intensity, performance, bound = roofline_analyze(bandwidth, max_OPS, OPs, memory_access)
    inference_time = OPs / performance
    return {
        "OPs": OPs,
        "memory_access": memory_access,
        "arithmetic_intensity": arithmetic_intensity,
        "performance": performance,
        "bound": bound,
        "load_weight": load_weight,
        "load_act": load_act,
        "store_act": store_act,
        "load_kv_cache": load_kv_cache,
        "store_kv_cache": store_kv_cache,
        "inference_time": inference_time,
    }


def roofline_analyze(bandwidth, max_OPS, OPs, memory_access):
    # bandwidth is bytes/s
    # memory_access in byte
    # x axis is OPS/byte
    # y axis is OPS/s
    y_max = max_OPS
    memory_access_bytes = memory_access
    turning_point = y_max / bandwidth
    arithmetic_intensity = OPs / memory_access_bytes
    if arithmetic_intensity < turning_point:
        bound = "memory"
        performance = arithmetic_intensity * bandwidth
    else:
        bound = "compute"
        performance = y_max
    if performance==0:
        1==1
        pass
    return arithmetic_intensity, performance, bound
