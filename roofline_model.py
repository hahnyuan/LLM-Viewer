def roofline_analyze(bandwidth, max_OPS, OPs, memory_access):
    # bandwidth is bytes/s
    # memory_access in bit
    # x axis is OPS/byte
    # y axis is OPS/s
    y_max = max_OPS
    memory_access_bytes=memory_access/8
    turning_point = y_max / bandwidth
    arithmetic_intensity=OPs/memory_access_bytes
    if arithmetic_intensity < turning_point:
        bound="memory"
        performance=arithmetic_intensity*bandwidth
    else:
        bound="compute"
        performance=y_max
    return arithmetic_intensity,performance,bound