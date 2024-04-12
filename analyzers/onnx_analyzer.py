from .base_analyzer import BaseAnalyzer
from modifier.quantization import QuantAct, QuantWeight
from modifier.memory_access import CalcMemoryAccess


class OnnxAnalyzer(BaseAnalyzer):
    frontend_params_info={
        "model_path": "",
        "x_shape_dict": {},
        "w_bit": {
            "type": "int",
            "min": 1,
            "max": 16,
            "default": 16,
            "description": "Bitwidth for weights"
        },
        "a_bit": {
            "type": "int",
            "min": 1,
            "max": 16,
            "default": 16,
            "description": "Bitwidth for activations"
        },
        "compute_dtype": {
            "type": "select",
            "choices": ["FP16", "INT8"],
            "default": "FP16",
            "description": "Compute data type"
        },
    }

    def analyze(
        self,
        x_shape_dict={},
        w_bit=16,
        a_bit=16,
        compute_dtype="FP16"
    ):
        """profile model

        Args:
            x_shape_dict (_type_): x_shape_dict
            w_bit (int, optional): w_bit. Defaults to 16.
            a_bit (int, optional): a_bit. Defaults to 16.
            compute_dtype (str, optional): compute_dtype. Defaults to "FP16".

        Returns:
            profile_results: return is a dict with the following format:
            {
                "layers": {
                    "node1": {
                            "OPs": "",
                            "input_shape": "",
                            "is_shared_weight": "",
                            "is_shared_input": "",
                            "memory": "",
                            "n_load_weight": "",
                            "n_load_act": "",
                            "n_store_act": "",
                            "output_shape": "",
                            "load_act": "",
                            "store_act": "",
                            "load_weight": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "bound": "",
                            "performance": "",
                            "inference_time": "",
                    }
                    "node2": ...,
                    ...
                }
                "network": {
                            "OPs": "",
                            "n_load_weight": "",
                            "n_load_act": "",
                            "n_store_act": "",
                            "load_act": "",
                            "store_act": "",
                            "load_weight": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "bound": "",
                            "performance": "",
                            "inference_time": "",
                }
            }
        """
        profile_results = self.net_graph.analyze_forward(x_shape_dict)
        
        modifiers=[
            QuantAct(a_bit),  # add info: load_act, load_act
            QuantWeight(w_bit),  # add info: load_weight
            CalcMemoryAccess()  # add info: memory. sum of load_act, load_act, load_weight
        ]
        for modifier in modifiers:
            modifier.run(profile_results)

        # add info: arithmetic_intensity, bound, performance, inference_time
        self.hardware_model.run(profile_results, compute_dtype)

        return profile_results