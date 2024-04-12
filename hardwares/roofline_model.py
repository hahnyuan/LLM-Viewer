from hardwares.hardware_params import hardware_params


class RooflineModel():
    def __init__(self, params):
        self.params=params
        self.bandwidth = params["bandwidth"]
        self.onchip_buffer=params["onchip_buffer"]

    def modify_node(self, node_info, dtype):
        memory_access=node_info["memory_access"]
        OPs=node_info["OPs"]
        if memory_access==0:
            return
        max_OPS=self.params[dtype]
        y_max = max_OPS
        memory_access_bytes = memory_access
        turning_point = y_max / self.bandwidth
        arithmetic_intensity = OPs / memory_access_bytes
        if arithmetic_intensity < turning_point:
            bound = "memory"
            performance = arithmetic_intensity * self.bandwidth
        else:
            bound = "compute"
            performance = y_max
        if performance==0:
            inference_time=memory_access/self.bandwidth
        else:
            inference_time = OPs / performance
        node_info["arithmetic_intensity"]=arithmetic_intensity
        node_info["bound"]=bound
        node_info["performance"]=performance
        node_info["inference_time"]=inference_time

    def run(self,analyze_rsts,dtype):
        if list(analyze_rsts.keys()) == ["layers", "network"]:  # onnx profile_info
            for _, node_info in analyze_rsts["layers"].items():
                self.modify_node(node_info, dtype)
            self.modify_node(analyze_rsts["network"], dtype)
        else:
            for name, item in analyze_rsts.items():
                node, node_info = item
                self.modify_node(node_info, dtype)

def get_roofline_model(hardware_name):
    model=RooflineModel(hardware_params[hardware_name])
    return model