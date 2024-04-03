import typing
from typing import Dict

class Node:
    def __init__(self, name: str, input_node_names=[], attrs={}):
        self.name = name
        self.input_node_names = input_node_names
        self.attrs = attrs
        for k,v in attrs.items():
            setattr(self,k,v)


    def analyze_node(self,input_shape) -> typing.Dict:
        rst={
            "OPS":0,
            "n_load_weight":0,
            "n_load_act":0,
            "n_store_act":0,
            "output_shape":[1]
        }
        return rst

    def __repr__(self) -> str:
        return f"{self.name}[{self.__class__.__name__}] input={self.input_node_names} {self.attrs}"

class Model:
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.topo_reorder()

    def topo_reorder(self):
        """
        Reorder the nodes in topological order.
        """
        visited = {}  # Map of node names to whether they have been visited
        topo_order = []  # List to store nodes in topological order

        def dfs(node, visited, topo_order):
            """
            Depth-first search to find the topological order of nodes.
            """
            visited[node.name] = True

            for input_node_name in node.input_node_names:
                input_node = next((n for n in self.nodes if n.name == input_node_name), None)
                if input_node and input_node_name not in visited:
                    dfs(input_node, visited, topo_order)

            topo_order.append(node)

        for node in self.nodes:
            if node.name not in visited:
                dfs(node, visited, topo_order)

        self.nodes = topo_order
            

    def print_graph(self):
        for node in self.nodes:
            print(node)

    def analyze_forward(self,x_shape_dict):
        """
        Analyze the forward pass of the model.
        """
        shape_dict={}
        shape_dict.update(x_shape_dict)
        rsts={}
        for node in self.nodes:
            for input_name in node.input_node_names:
                assert input_name in shape_dict, f"Input shape {input_name} not found"
            input_shapes=[shape_dict[input_name] for input_name in node.input_node_names]

            op_info=node.analyze_node(input_shapes)
            shape_dict[node.name]=op_info["output_shape"]
            rsts[node.name]=(node,op_info)
        return rsts
