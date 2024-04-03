from graph.model import Node
import numpy as np

class LinearNode(Node):
    """
    attr:
    - in_features: int
    - out_features: int
    """
    def analyze_node(self,input_shape):
        output_shape=input_shape[:-1]+[self.out_features]
        rst={
            "OPS": np.prod(input_shape)*self.out_features,
            "n_load_weight": self.in_features*self.out_features,
            "n_load_act": np.prod(input_shape),
            "n_store_act": np.prod(output_shape),
            "output_shape": output_shape
        }
        return rst