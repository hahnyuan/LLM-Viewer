from analyzers.base_analyzer import BaseAnalyzer
from modifier.quantization import QuantAct, QuantWeight, QuantKV
from modifier.kv_cache import MakeKVLoadStore,AddDecodeKVLoad

ALL_DATA_NAMES = [
    "OPs",
    "memory_access",
    "inference_time",
]

class LLMAnalyzer(BaseAnalyzer):
    def analyze(
        self,
        seqlen,
        batchsize,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        use_flashattention=False,
        n_parallel_decode=1,
    ):
        """
        return is a dict with the following format:
        {
            "decode": {
                    "layer_name": {
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
                            "inference_time": ""
                    }
            },
            "prefill": {
                    "layer_name": {
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
                            "inference_time": ""
                    }
            },
            "total_results": {
                "decode": {},
                "prefill": {}
            }
        }
        """
        assert seqlen > 0
        assert batchsize > 0
        self.results = {"decode": {}, "prefill": {}}
        assert use_flashattention==False

        # prefill

        prefill_modifiers=[
            MakeKVLoadStore(),
            QuantAct(a_bit),
            QuantWeight(w_bit),
            QuantKV(kv_bit)
        ]

        x_shape_dict={"input":[batchsize, seqlen]}
        results=self.net_graph.analyze_forward(x_shape_dict)

        for modifier in prefill_modifiers:
            modifier.run(results)
        self.results["prefill"]=results

        

        # decode

        decode_modifiers=[
            MakeKVLoadStore(),
            AddDecodeKVLoad(kv_seqlen=seqlen,n_parallel_decode=n_parallel_decode),
            QuantAct(a_bit),
            QuantWeight(w_bit),
            QuantKV(kv_bit)
        ]

        x_shape_dict={"input":[n_parallel_decode, seqlen]}
        results=self.net_graph.analyze_forward(x_shape_dict)

        for modifier in decode_modifiers:
            modifier.run(results)
        self.results["decode"]=results

        # compute total
        num_hidden_layers=0
        total_results = {"decode": {}, "prefill": {}}
        self.results["total_results"] = total_results
        return self.results

    def set_hooks(self, model):
        self.net_graph = model