from net_graph.module import Module
from net_parsers.onnx.graph import Graph
import typing

class Network:
    def __init__(self, modules: typing.List[Module]):
        self.modules = modules
        self.topo_reorder()
        self.max_n_act=None

    def topo_reorder(self):
        """
        Reorder the modules in topological order.
        """

        visited = {}  
        topo_order = []  

        def dfs(module, visited, topo_order):
            visited[module.name] = True

            for module_input_name in module.input_names:
                input_module = next((m for m in self.modules if m.name == module_input_name), None)
                if input_module and module_input_name not in visited:
                    dfs(input_module, visited, topo_order)

            topo_order.append(module)

        for module in self.modules:
            if module.name not in visited:
                dfs(module, visited, topo_order)

        self.modules = topo_order
            

    def print_graph(self):
        for module in self.modules:
            print(f"=== module {module.name} ===")
            module.print_graph()

    def analyze_forward(self,x_shape_dict,extra_args={}):
        """
        Analyze the forward pass of the model.
        """
        shape_dict={}
        rsts={}
        max_n_act=0
        for module in self.modules:
            module_input_dict={}
            for input_name in module.input_names:
                if input_name in x_shape_dict:
                    module_input_dict[input_name]=x_shape_dict[input_name]
                elif input_name in shape_dict:
                    module_input_dict[input_name]=shape_dict[input_name]
                else:
                    raise ValueError(f"Input shape {input_name} not found")

            rst,module_max_n_act=module.analyze_forward(module_input_dict,extra_args)
            max_n_act=max(max_n_act,module_max_n_act)
            module_name=module.name
            for op_name,op_info in rst.items():
                new_name=f"{module_name}.{op_name}"
                shape_dict[new_name]=op_info[1]["output_shape"]
                rsts[new_name]=op_info
        self.max_n_act=max_n_act
        return rsts


class OnnxNetwork:
    def __init__(self, graph: Graph):
        self.graph = graph   

    def print_graph(self):
        "NOTE:It's not accurate to print by module, although can split name by _ or / or ."
        self.graph.print_graph()

    def analyze_forward(self, x_shape_dict: dict={}):
        """
        Analyze the forward pass of the model.
        x_shape_dict: {name: List(int)}
        """
        if x_shape_dict!={}:
            dummy_input = {name:np.random.randn(*shape) for name,shape in x_shape_dict.items()}
        else:
            dummy_input = None
        self.graph.shape_infer(dummy_input)
        self.graph.profile()
        self.graph.print_profile_info()

        profile_results = {"layers": {name:node.profile_info for name,node in self.graph.nodemap.items()}, 
                            "network": self.graph.graph_profile_info}
        return profile_results
