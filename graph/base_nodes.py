from graph.model import Node
import numpy as np

class Linear(Node):
    """
    attr:
    - out_features: int
    """
    def analyze_node(self,input_shape):
        output_shape=input_shape[:-1]+[self.out_features]
        rst={
            "OPS": np.prod(input_shape)*self.out_features,
            "n_load_weight": input_shape[-1]*self.out_features,
            "n_load_act": np.prod(input_shape),
            "n_store_act": np.prod(output_shape),
            "output_shape": output_shape
        }
        return rst

class Input(Node):
    def analyze_node(self):
        rst={
            "OPS":0,
            "n_load_weight":0,
            "n_load_act":0,
            "n_store_act":0,
            "output_shape":self.shape
        }
        return rst
    
class MatMul(Node):
    def analyze_node(self,a_shape,b_shape):
        assert len(a_shape)>=2
        assert len(b_shape)>=2
        assert a_shape[-1]==b_shape[-2]
        output_shape=a_shape[:-1]+b_shape[-1:]
        rst={
            "OPS": b_shape[-1]*np.prod(output_shape),
            "n_load_weight": 0,
            "n_load_act": np.prod(b_shape)+np.prod(a_shape),
            "n_store_act": np.prod(output_shape),
            "output_shape": output_shape
        }
        return rst

class Add(Node):
    def analyze_node(self,a_shape,b_shape):
        assert a_shape==b_shape
        output_shape=[max(a,b) for a,b in zip(a_shape,b_shape)]
        rst={
            "OPS": max(np.prod(a_shape),np.prod(b_shape)),
            "n_load_weight": 0,
            "n_load_act": np.prod(a_shape)+np.prod(b_shape),
            "n_store_act": max(np.prod(a_shape),np.prod(b_shape)),
            "output_shape": output_shape
        }
        return rst
    
class Softmax(Node):
    def analyze_node(self,input_shape):
        output_shape=input_shape
        rst={
            "OPS": np.prod(input_shape)*5,
            "n_load_weight": 0,
            "n_load_act": np.prod(input_shape),
            "n_store_act": np.prod(input_shape),
            "output_shape": output_shape
        }
        return rst

class Norm(Node):
    def analyze_node(self,input_shape):
        output_shape=input_shape
        rst={
            "OPS": np.prod(input_shape)*7,
            "n_load_weight": 0,
            "n_load_act": np.prod(input_shape),
            "n_store_act": np.prod(input_shape),
            "output_shape": output_shape
        }
        return rst
    
class Activation(Node):
    def analyze_node(self,input_shape):
        output_shape=input_shape
        rst={
            "OPS": np.prod(input_shape)*2,
            "n_load_weight": 0,
            "n_load_act": np.prod(input_shape),
            "n_store_act": np.prod(input_shape),
            "output_shape": output_shape
        }
        return rst