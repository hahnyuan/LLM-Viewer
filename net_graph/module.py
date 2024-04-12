import typing
import numpy as np

class Node:
    def __init__(self, name: str, input_names=[], attrs={}):
        self.name = name
        self.input_names = input_names
        self.attrs = attrs
        for k,v in attrs.items():
            setattr(self,k,v)


    def analyze_node(self,input_shapes,extra_args) -> typing.Dict:
        rst={
            "OPs":0,
            "n_load_weight":0,
            "n_load_act":0,
            "n_store_act":0,
            "output_shape":[1]
        }
        return rst

    def __repr__(self) -> str:
        return f"{self.name}[{self.__class__.__name__}] input={self.input_names} {self.attrs}"

class Module:
    def __init__(self, nodes=[], name="module"):
        self.name=name
        self.input_names=self.get_module_input_names(nodes)
        self.nodes=self.get_node_topo_order(nodes)
        self.max_n_act=None

    def get_module_input_names(self,nodes):
        all_input=set()
        
        for node in nodes:
            for input_name in node.input_names:
                all_input.add(input_name)
        input_names=all_input-set([node.name for node in nodes])
        return input_names

    def get_node_topo_order(self,nodes):
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

            for input_node_name in node.input_names:
                input_node = next((n for n in nodes if n.name == input_node_name), None)
                if input_node and input_node_name not in visited:
                    dfs(input_node, visited, topo_order)

            topo_order.append(node)

        for node in nodes:
            if node.name not in visited:
                dfs(node, visited, topo_order)

        return topo_order
            

    def print_graph(self):
        for node in self.nodes:
            print(node)

    def analyze_forward(self,x_shape_dict,extra_args={}):
        """
        Analyze the forward pass of the model.
        """
        node_output_nodes={}
        # get output nodes
        for node in self.nodes:
            for input_name in node.input_names:
                if input_name in node_output_nodes:
                    node_output_nodes[input_name].append(node)
                else:
                    node_output_nodes[input_name]=[node]

        shape_dict={}
        rsts={}
        remain_shape_dict={key:value for key,value in x_shape_dict.items()}
        max_n_act=sum([np.prod(shape) for shape in x_shape_dict.values()])
        
        for node in self.nodes:
            node_input_shapes=[]
            for input_name in node.input_names:
                if input_name in x_shape_dict:
                    node_input_shapes.append(x_shape_dict[input_name])
                elif input_name in shape_dict:
                    node_input_shapes.append(shape_dict[input_name])
                else:
                    raise ValueError(f"Input shape {input_name} not found")

            op_info=node.analyze_node(node_input_shapes,extra_args)
            shape_dict[node.name]=op_info["output_shape"]
            rsts[node.name]=(node,op_info)

            # process act
            remain_shape_dict[node.name]=op_info["output_shape"]
            for input_name in node.input_names:
                node_output_nodes[input_name].remove(node)
                if len(node_output_nodes[input_name])==0:
                    remain_shape_dict.pop(input_name)
            now_act=sum([np.prod(shape) for shape in remain_shape_dict.values()])
            max_n_act=max(max_n_act,now_act)
        self.max_n_act=max_n_act
        return rsts,max_n_act

    def __repr__(self) -> str:
        return f"{self.name}({self.input_names})"