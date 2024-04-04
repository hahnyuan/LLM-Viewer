from graph.module import Module
import typing

class Network:
    def __init__(self, modules: typing.List[Module]):
        self.modules = modules
        self.topo_reorder()

    def topo_reorder(self):
        """
        Reorder the modules in topological order.
        """
        for module in self.modules:
            for node in module.nodes:
                if '.' in node.input

        visited = {}  
        topo_order = []  

        def dfs(module, visited, topo_order):
            visited[module.name] = True

            for input_module_name in module.input_module_names:
                input_module = next((m for m in self.modules if m.name == input_module_name), None)
                if input_module and input_module_name not in visited:
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

    def analyze_forward(self,x_shape_dict):
        """
        Analyze the forward pass of the model.
        """
        shape_dict={}
        rsts={}
        for node in self.nodes:
            node_input_shapes=[]
            for input_name in node.input_node_names:
                if input_name in x_shape_dict:
                    node_input_shapes[input_name]=x_shape_dict[input_name]
                elif input_name in shape_dict:
                    node_input_shapes[input_name]=shape_dict[input_name]
                else:
                    raise ValueError(f"Input shape {input_name} not found")

            op_info=node.analyze_node(node_input_shapes)
            shape_dict[node.name]=op_info["output_shape"]
            rsts[node.name]=(node,op_info)
        return rsts

    