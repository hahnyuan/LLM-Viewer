from net_graph.module import Module
import typing

class Network:
    def __init__(self, modules: typing.List[Module]):
        self.modules = modules
        self.topo_reorder()

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
        for module in self.modules:
            module_input_dict={}
            for input_name in module.input_names:
                if input_name in x_shape_dict:
                    module_input_dict[input_name]=x_shape_dict[input_name]
                elif input_name in shape_dict:
                    module_input_dict[input_name]=shape_dict[input_name]
                else:
                    raise ValueError(f"Input shape {input_name} not found")

            rst=module.analyze_forward(module_input_dict,extra_args)
            module_name=module.name
            for op_name,op_info in rst.items():
                new_name=f"{module_name}.{op_name}"
                shape_dict[new_name]=op_info[1]["output_shape"]
                rsts[new_name]=op_info
        return rsts

    