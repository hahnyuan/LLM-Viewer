from analyzers.base_analyzer import BaseAnalyzer
from modifier.quantization import QuantAct, QuantWeight, QuantKV
from modifier.memory_access import CalcMemoryAccess
from utils import str_number,str_number_1024,numpy_value_to_python

NETWORK_WISE_NAMES = [
    "OPs",
    "memory_access",
    "inference_time",
    "load_act",
    "load_kv_cache",
    "load_weight",
    "store_act",
    "store_kv_cache"
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
            "name": "batchsize",
            "type": "int",
            "min": 1,
            "max": 128,
            "default": 1,
            "description": "Batch size"
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
             "default": "INT8",
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
        compute_dtype="INT8"
    ):
        """
        return is a dict with the following format:
        {
            "layers": {
                "layer_name1": (node, {
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
                })
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
            modifiers=[]
            x_shape_dict={"input_inds":[batchsize, seqlen]}
            extra_args={"kv_seqlen":0}
        elif stage=="decode":
            modifiers=[]
            x_shape_dict={"input_inds":[batchsize, n_parallel_decode]}
            extra_args={"kv_seqlen":seqlen}
        else:
            #TODO write the chat stage
            raise ValueError(f"stage {stage} is not supported")
        if use_flashattention:
            extra_args["onchip_buffer"]=self.hardware_model.onchip_buffer
        modifiers.extend([QuantAct(a_bit),
                QuantWeight(w_bit),
                QuantKV(kv_bit),
                CalcMemoryAccess()])

        layer_results=self.net_graph.analyze_forward(x_shape_dict,extra_args)

        for modifier in modifiers:
            modifier.run(layer_results)

        self.hardware_model.run(layer_results,compute_dtype)

        # compute total
        network_results = {_:0 for _ in NETWORK_WISE_NAMES}
        for layer_name, (layer,layer_info) in layer_results.items():
            for data_name in NETWORK_WISE_NAMES:
                if data_name in layer_info:
                    network_results[data_name] += layer_info[data_name]
        # compute memory consumption
        n_weight=0
        n_kv_cache=0
        for name, (node, node_info) in layer_results.items():
            n_weight+=node_info.get("n_weight",0)
            n_kv_cache+=node_info.get("n_store_kv_cache",0)+node_info.get("n_load_kv_cache",0)
        max_n_act=self.net_graph.max_n_act
        memory_consumption_weight=n_weight*w_bit/8
        memory_consumption_kv_cache=n_kv_cache*kv_bit/8
        memory_consumption_tmp_act=max_n_act*a_bit/8

        memory_consumption=memory_consumption_weight+memory_consumption_kv_cache+memory_consumption_tmp_act
        network_results.update({
            "memory_consumption": memory_consumption,
            "memory_consumption_weight": memory_consumption_weight,
            "memory_consumption_kv_cache": memory_consumption_kv_cache,
            "memory_consumption_tmp_act": memory_consumption_tmp_act
        })

        results={"layers":layer_results,"network":network_results}
        return results

    def get_ui_graph(self,result):

        module_graphs={}
        network_nodes=[]
        network_edges=[]

        for module in self.net_graph.modules:
            module_name=module.name
            nodes = []
            edges = []


            for input_name in module.input_names:
                if '.' in input_name:
                    source,show_input_name=input_name.split('.')
                else:
                    show_input_name=input_name
                    source="input"
                input_node = {
                    "label": show_input_name,
                    "description": f"from: {source}",
                    "id": input_name,
                    "info":{"layer_type": "Input"}
                }
                nodes.append(input_node)

            for node in module.nodes:
                name=f"{module_name}.{node.name}"
                if name not in result["layers"]:
                    continue
                info=result["layers"][name][1]
                info["layer_type"]=node.__class__.__name__
                numpy_value_to_python(info)
                
                nodes.append({
                    "label": node.name,
                    "id": node.name,
                    "description": f"OPs:{str_number(info['OPs'])}, Access:{str_number_1024(info['memory_access'])}B",
                    "info": info,
                })
                for input_name in node.input_names:
                    edge = {"source": input_name, "target": node.name}
                    edges.append(edge)

            module_graphs[module_name] = {"nodes": nodes, "edges": edges}

            network_node={
                "label": module_name,
                "id": module_name
            }
            network_nodes.append(network_node)
            for input_name in module.input_names:
                if '.' in input_name:
                    source=input_name.split('.')[0]
                    edge = {"source": source, "target": module_name}
                    network_edges.append(edge)
        network_graph={"nodes":network_nodes,"edges":network_edges}
        return network_graph, module_graphs