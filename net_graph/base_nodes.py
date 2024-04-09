from typing import Dict
from net_graph.module import Node
import numpy as np

class Linear(Node):
    """
    attr:
    - out_features: int
    """
    def analyze_node(self,input_shapes):
        input_shape=input_shapes[0]

        output_shape=input_shape[:-1]+[self.out_features]
        rst={
            "OPs": np.prod(input_shape)*self.out_features,
            "n_weight": input_shape[-1]*self.out_features,
            "n_load_weight": input_shape[-1]*self.out_features,
            "n_load_act": np.prod(input_shape),
            "n_store_act": np.prod(output_shape),
            "output_shape": output_shape
        }
        return rst

class Embedding(Node):
    """
    attr:
    - out_features: int
    - vocab_size: int
    """
    def analyze_node(self,input_shapes):
        input_shape=input_shapes[0]
        out_features=self.out_features

        output_shape=input_shape+[out_features]
        rst={
            "OPs": 0,
            "n_weight": np.prod(self.vocab_size*self.out_features),
            "n_load_weight": np.prod(output_shape),
            "n_load_act": np.prod(input_shape),
            "n_store_act": np.prod(output_shape),
            "output_shape": output_shape
        }
        return rst
    
class MatMul(Node):
    def analyze_node(self,input_shapes):
        a_shape,b_shape=input_shapes
        if self.attrs.get("transpose_a",False):
            a_shape=a_shape[:-2]+[a_shape[-1],a_shape[-2]]
        if self.attrs.get("transpose_b",False):
            b_shape=b_shape[:-2]+[b_shape[-1],b_shape[-2]]
        assert len(a_shape)>=2
        assert len(b_shape)>=2
        assert a_shape[-1]==b_shape[-2]
        output_shape=a_shape[:-1]+b_shape[-1:]
        rst={
            "OPs": b_shape[-1]*np.prod(output_shape),
            "n_load_weight": 0,
            "n_load_act": np.prod(b_shape)+np.prod(a_shape),
            "n_store_act": np.prod(output_shape),
            "output_shape": output_shape
        }
        return rst


class Add(Node):
    def analyze_node(self,input_shapes):
        a_shape,b_shape=input_shapes
        assert a_shape==b_shape
        output_shape=[max(a,b) for a,b in zip(a_shape,b_shape)]
        rst={
            "OPs": max(np.prod(a_shape),np.prod(b_shape)),
            "n_load_weight": 0,
            "n_load_act": np.prod(a_shape)+np.prod(b_shape),
            "n_store_act": max(np.prod(a_shape),np.prod(b_shape)),
            "output_shape": output_shape
        }
        return rst
    
class Softmax(Node):
    def analyze_node(self,input_shapes):
        input_shape=input_shapes[0]
        output_shape=input_shape
        rst={
            "OPs": np.prod(input_shape)*5,
            "n_load_weight": 0,
            "n_load_act": np.prod(input_shape),
            "n_store_act": np.prod(input_shape),
            "output_shape": output_shape
        }
        return rst

class Norm(Node):
    def analyze_node(self,input_shapes):
        input_shape=input_shapes[0]
        output_shape=input_shape
        rst={
            "OPs": np.prod(input_shape)*7,
            "n_load_weight": 0,
            "n_load_act": np.prod(input_shape),
            "n_store_act": np.prod(input_shape),
            "output_shape": output_shape
        }
        return rst
    
class Activation(Node):
    def analyze_node(self,input_shapes):
        input_shape=input_shapes[0]
        output_shape=input_shape
        rst={
            "OPs": np.prod(input_shape)*2,
            "n_load_weight": 0,
            "n_load_act": np.prod(input_shape),
            "n_store_act": np.prod(input_shape),
            "output_shape": output_shape
        }
        return rst
    
    
class ReshapeTranspose(Node):
    def analyze_node(self,input_shapes):
        input_shape=input_shapes[0]
        output_shape=[]
        for i in self.attrs["shape"]:
            if type(i)==str:
                output_shape.append(eval(i))
            else:
                output_shape.append(i)
        assert np.prod(input_shape)==np.prod(output_shape)
        # no load and store, because we assume the reshape is fused to other operations
        # you shold understand this is a theoretical assumption
        rst={
            "OPs": 0,
            "n_load_weight": 0,
            "n_load_act": 0,
            "n_store_act": 0,
            "output_shape": output_shape
        }
        return rst