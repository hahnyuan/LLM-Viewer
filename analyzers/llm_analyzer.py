from analyzers.base_analyzer import BaseAnalyzer
from modifier.quantization import QuantAct, QuantWeight, QuantKV
from modifier.kv_cache import MakeKVLoadStore,AddDecodeKVLoad
from modifier.memory_access import CalcMemoryAccess
from modifier.flashattention import FlashAttention

NETWORK_WISE_NAMES = [
    "OPs",
    "memory_access",
    "inference_time",
]

class LLMAnalyzer(BaseAnalyzer):
    frontend_params_info=[
        {
            "name": "stage",
            "type": "select",
            "choices": ["decode", "prefill",  "chat"],
            "default": "decode",
            "description": "Stage of the model, either prefill or decode"
        },
        {
            "name": "seqlen",
            "type": "int",
            "min": 1,
            "max": 4096,
            "default": 1024,
            "description": "Sequence length in prefill stage, and prefilled sequence length in decode stage"
        },
        {
            "name": "batchsize",
            "type": "int",
            "min": 1,
            "max": 4096,
            "default": 1,
            "description": "Batch size"
        },
        {
            "name": "n_parallel_decode",
            "type": "int",
            "min": 1,
            "max": 64,
            "default": 1,
            "description": "Number of parallel decodes"
        },
        {
            "name": "w_bit",
            "type": "int",
            "min": 1,
            "max": 16,
            "default": 16,
            "description": "Bitwidth for weights"
        },
        {
            "name": "a_bit",
            "type": "int",
            "min": 1,
            "max": 16,
            "default": 16,
            "description": "Bitwidth for activations"
        },
        {
            "name": "kv_bit",
            "type": "int",
            "min": 1,
            "max": 16,
            "default": 16,
            "description": "Bitwidth for key and value cache"
        },
        {
            "name": "use_flashattention",
            "type": "bool",
            "default": False,
            "description": "Whether to use flashattention"
        },
        {
            "name": "compute_dtype",
            "type": "select",
            "choices": ["FP16", "INT8"],
            "default": "FP16",
            "description": "Compute data type"
        }
    ]
        

    def analyze(
        self,
        seqlen=1024,
        batchsize=1,
        stage="decode",
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        use_flashattention=False,
        n_parallel_decode=1,
        compute_dtype="FP16"
    ):
        """
        return is a dict with the following format:
        {
            "layers": {
                "layer_name1": {
                        "OPs": "",
                        "memory_access": "",
                        "arithmetic_intensity": "",
                        "performance": "",
                        "bound": "",
                        "load_weight": "",
                        "load_act": "",
                        "store_act": "",
                        "load_kv_cache": "",
                        "store_kv_cache": "",
                        "inference_time": "",
                        ...
                }
                "layer_name2": ...,
                ...
            }
            "network": {
                "OPs": ...,
                "memory_access": ...,
                "inference_time": ...,
            }
        }
        """
        assert seqlen > 0
        assert batchsize > 0
        if kv_bit is None:
            kv_bit = a_bit

        if stage=="prefill":
            modifiers=[
                MakeKVLoadStore(),
            ]
            x_shape_dict={"input_inds":[batchsize, seqlen]}
        elif stage=="decode":
            modifiers=[
                MakeKVLoadStore(),
                AddDecodeKVLoad(kv_seqlen=seqlen,n_parallel_decode=n_parallel_decode),
            ]
            x_shape_dict={"input_inds":[n_parallel_decode, seqlen]}
        else:
            #TODO write the chat stage
            raise ValueError(f"stage {stage} is not supported")
        if use_flashattention:
            modifiers.append(FlashAttention(self.hardware_model.onchip_buffer))
        modifiers.extend([QuantAct(a_bit),
                QuantWeight(w_bit),
                QuantKV(kv_bit),
                CalcMemoryAccess()])

        layer_results=self.net_graph.analyze_forward(x_shape_dict)

        for modifier in modifiers:
            modifier.run(layer_results)

        self.hardware_model.run(layer_results,compute_dtype)

        # compute total
        network_results = {_:0 for _ in NETWORK_WISE_NAMES}
        for layer_name, (layer,layer_info) in layer_results.items():
            for data_name in NETWORK_WISE_NAMES:
                if data_name in layer_info:
                    network_results[data_name] += layer_info[data_name]

        results={"layers":layer_results,"network":network_results}
        return results

    def set_hooks(self, model):
        self.net_graph = model