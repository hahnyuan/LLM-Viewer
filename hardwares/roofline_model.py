

class RooflineModel():
    def __init__(self, peak_flops, peak_memory_bandwidth):
        self.peak_flops = peak_flops
        self.peak_memory_bandwidth = peak_memory_bandwidth

    def roofline(self, op_intensity, memory_intensity):
        if op_intensity < 0 or memory_intensity < 0:
            raise ValueError("Operation intensity and memory intensity must be positive")
        if op_intensity > self.peak_flops:
            raise ValueError("Operation intensity cannot exceed peak FLOPS")
        if memory_intensity > self.peak_memory_bandwidth:
            raise ValueError("Memory intensity cannot exceed peak memory bandwidth")
        return min(op_intensity, memory_intensity)

    def roofline_points(self, op_intensities, memory_intensities):
        return [(op_intensity, self.roofline(op_intensity, memory_intensity)) for op_intensity, memory_intensity in zip(op_intensities, memory_intensities)]

    def roofline_plot(self, op_intensities, memory_intensities, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(op_intensities, memory_intensities, label="Roofline", color="black")
        ax.plot(op_intensities, self.roofline_points(op_intensities, memory_intensities), label="Roofline", color="black", linestyle="--")
        ax.set_xlabel("Operation intensity (FLOPS/byte)")
        ax.set_ylabel("Memory intensity (bytes/FLOP)")
        return ax

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