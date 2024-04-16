from .base_analyzer import BaseAnalyzer
from modifier.quantization import QuantAct, QuantWeight
from modifier.memory_access import CalcMemoryAccess
from utils import str_number,str_number_1024,numpy_value_to_python


class OnnxAnalyzer(BaseAnalyzer):
    frontend_params_info=[
        {
            "name": "model_path",
            "type": "str",
            "default": "data/onnx/light_squeezenet.onnx",
            "description": "Path to the model"
        },
        {
            "name": "input_shape_info",
            "type": "str",
            "default": "",
            "description": "Input shape dict of the onnx which use dynamic input, format: name1:1,2,224,224;name2:1,4,64,64"
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
            "name": "compute_dtype",
            "type": "select",
            "choices": ["FP16", "INT8"],
            "default": "FP16",
            "description": "Compute data type"
        }
    ]

    def analyze(
        self,
        model_path="",
        input_shape_info="",
        w_bit=16,
        a_bit=16,
        compute_dtype="FP16"
    ):
        """profile model

        NOTE: args is same with frontend_params_info.keys()

        Args:
            input_shape_info (str): input_shape_info
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
                    },
                    "node2": ...,
                    ...
                },
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
        input_shape_dict = {}
        if input_shape_info != "":
            for info in input_shape_info.split(";"):
                assert len(info.split(":")) == 2, f"input_shape_info format error: {input_shape_info}"
                name, shape_info = info.split(":")
                shape = [int(i) for i in shape_info.split(",")]
                input_shape_dict[name] = shape

        profile_results = self.net_graph.analyze_forward(input_shape_dict)
        
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

    def get_ui_graph(self,result):
        """create graph to show

        Args:
            result (dict): profile_results

        Returns:
            network_graph:
            {
                "nodes":[{
                    "label":"",
                    "id":"",
                    }
                ],
                "edges":[
                    "source":"",  # module的name
                    "target":"",  # module的name
                ]
            }
            module_graphs:
            {
                key:{  # module name
                    "nodes":[
                        "label":"",  # 用于显示的layer name
                        "description":f"OPs:{str_number(info['OPs'])}, Access:{str_number_1024(info['memory_access'])}B",
                        "id":"",  # layer的name
                        "info":{"layer_type":"", ...}  # 相对于profile信息，多了layer_type字段
                    ],
                    "edges":[{
                        "source":"",  # layer的name
                        "target":"",  # layer的name
                        }
                    ]
                }
            }
        """
        def add_input_module_node_info(module_nodes, tensor_name=None):
            for input_name in self.net_graph.graph.input:
                if tensor_name is not None and input_name != tensor_name:
                    continue
                in_shape = [numpy_value_to_python(i) for i in self.net_graph.graph.tensormap[input_name].shape]
                module_nodes.append({
                    "label":f'input:{input_name}',  # avoid input_tensor's name is same name with node
                    "description": "input tensor",
                    "id":f'input:{input_name}',
                    "info": {'layer_type':'Input', 'input_shape': in_shape}, 
                })
        def add_output_module_node_info(module_nodes, tensor_name=None):
            for output_name in self.net_graph.graph.output:
                if tensor_name is not None and output_name != tensor_name:
                    continue
                out_shape = [numpy_value_to_python(i) for i in self.net_graph.graph.tensormap[output_name].shape]
                module_nodes.append({
                    "label":f'output:{output_name}',
                    "description": "output tensor",
                    "id":f'output:{output_name}',
                    "info": {'layer_type':'Output', 'output_shape': out_shape}, 
                })
        def add_input_module_info(module_graphs):
            for tensor_name in self.net_graph.graph.input:
                module_graphs[tensor_name] = {"nodes": [], "edges":[]}
                module_nodes = module_graphs[tensor_name]["nodes"]
                add_input_module_node_info(module_nodes, tensor_name)
        def add_output_module_info(module_graphs):
            for tensor_name in self.net_graph.graph.output:
                module_graphs[tensor_name] = {"nodes": [], "edges":[]}
                module_nodes = module_graphs[tensor_name]["nodes"]
                add_output_module_node_info(module_nodes, tensor_name)

        def create_network_ui_graph():
            network_nodes=[]
            network_edges=[]
            for name in self.net_graph.graph.input:
                network_nodes.append({"label":name, "id":name})
                network_edges.append({"source":name, "target":"network"})
            network_nodes.append({"label":"network", "id":"network"})
            for name in self.net_graph.graph.output:
                network_nodes.append({"label":name, "id":name})
                network_edges.append({"source":"network", "target":name})
            network_graph={"nodes":network_nodes,"edges":network_edges}
            return network_graph

        network_graph = create_network_ui_graph()

        module_graphs = {"network": {"nodes": [], "edges": []}}


        add_input_module_info(module_graphs)
        add_output_module_info(module_graphs)

        module_nodes, module_edges = module_graphs["network"]["nodes"], module_graphs["network"]["edges"]
        nodemap = self.net_graph.graph.nodemap

        add_input_module_node_info(module_nodes)
        for node_name, info in result["layers"].items():
            node = nodemap[node_name]
            info["layer_type"] = node.op_type
            module_nodes.append({
                "label":node.op_type,
                # "label":node_name,
                "description":f"OPs:{str_number(info['OPs'])}, Access:{str_number_1024(info['memory_access'])}B",
                "id":node_name,
                "info": numpy_value_to_python(info),
            })

            for tensor_name in node.input:
                if tensor_name in self.net_graph.graph.input:
                    module_edges.append({"source":f'input:{tensor_name}', "target":node_name})
            for tensor_name in node.output:
                if tensor_name in self.net_graph.graph.output:
                    module_edges.append({"source":node_name, "target":f'output:{tensor_name}'})

            for output_node in node.nextnodes:
                module_edges.append({"source":node_name, "target":output_node.name})
        add_output_module_node_info(module_nodes)
        return network_graph, module_graphs