from typing import Dict
from net_graph.module import Node
import numpy as np
import math

# NOTICE: a MAC(multiply-accumulate) should be counted as 2 OPs

class Linear(Node):
    """
    attr:
    - out_features: int
    """
    def analyze_node(self,input_shapes, extra_args):
        input_shape=input_shapes[0]

        output_shape=input_shape[:-1]+[self.out_features]
        rst={
            "OPs": np.prod(input_shape)*self.out_features*2,
            "n_weight": input_shape[-1]*self.out_features,
            "n_load_weight": input_shape[-1]*self.out_features,
            "n_load_act": np.prod(input_shape),
            "n_store_act": np.prod(output_shape),
            "output_shape": output_shape
        }
        return rst

class LinearWithStoreKVCache(Node):
    """
    attr:
    - out_features: int
    """
    def analyze_node(self,input_shapes, extra_args):
        input_shape=input_shapes[0]
        output_shape=input_shape[:-1]+[self.out_features]
        rst={
            "OPs": np.prod(input_shape)*self.out_features*2,
            "n_weight": input_shape[-1]*self.out_features,
            "n_load_weight": input_shape[-1]*self.out_features,
            "n_load_act": np.prod(input_shape),
            "n_store_kv_cache": np.prod(output_shape),
            "output_shape": output_shape
        }
        return rst

class Embedding(Node):
    """
    attr:
    - out_features: int
    - vocab_size: int
    """
    def analyze_node(self,input_shapes, extra_args):
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
    def analyze_node(self,input_shapes, extra_args):
        a_shape,b_shape=input_shapes
        assert len(a_shape)>=2
        assert len(b_shape)>=2
        assert a_shape[-1]==b_shape[-2]
        output_shape=a_shape[:-1]+b_shape[-1:]
        rst={
            "OPs": a_shape[-1]*np.prod(output_shape)*2,
            "n_load_weight": 0,
            "n_load_act": np.prod(b_shape)+np.prod(a_shape),
            "n_store_act": np.prod(output_shape),
            "output_shape": output_shape
        }
        return rst

class MatmulWithLoadKVCache(Node):
    def analyze_node(self, input_shapes, extra_args):
        a_shape, b_shape = input_shapes
        kv_seqlen=extra_args.get("kv_seqlen",0)
        assert len(a_shape) >= 2
        assert len(b_shape) >= 2
        kv_cache_shape=[_ for _ in b_shape]
        kv_cache_shape[self.attrs["concat_dim"]]=kv_seqlen
        new_b_shape=[_ for _ in b_shape]
        new_b_shape[self.attrs["concat_dim"]]+=kv_seqlen
        assert a_shape[-1] == new_b_shape[-2]
        output_shape = a_shape[:-1] + new_b_shape[-1:]
        rst = {
            "OPs": a_shape[-1] * np.prod(output_shape) * 2,
            "n_load_weight": 0,
            "n_load_act": np.prod(b_shape) + np.prod(a_shape),
            "n_store_act": np.prod(output_shape),
            "n_load_kv_cache": np.prod(kv_cache_shape),
            "output_shape": output_shape
        }
        return rst


class Add(Node):
    def analyze_node(self,input_shapes, extra_args):
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
    def analyze_node(self,input_shapes, extra_args):
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
    def analyze_node(self,input_shapes, extra_args):
        input_shape=input_shapes[0]
        output_shape=input_shape
        rst={
            "OPs": np.prod(input_shape)*7,
            "n_load_act": np.prod(input_shape),
            "n_store_act": np.prod(input_shape),
            "output_shape": output_shape
        }
        return rst
    
class Activation(Node):
    def analyze_node(self,input_shapes, extra_args):
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
    def analyze_node(self,input_shapes, extra_args):
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
    
class FlashAttention(Node):
    def analyze_node(self, input_shapes, extra_args):
        """
        input: q_reshape, k_reshape, v_reshape
        This analyze FlashAttention V2
        M is on-chip SRAM size
        block size Br=min(ceil(M/4d),d) and Bc=ceil(M/4d)
        Divide Q/O into Tr=ceil(q_length/Br) and K into Tc=ceil(kv_length/Bc) blocks
        for i in Tr
            load Qi
            On chip, initialize Oi
            for j in Tc
                load Kj,Vj
                compute Si=Qi Kj^T
                compute mi=max(mi_old,rowmax(Si)) Pi=exp(Si-mi) li=exp(mi_old-mi)li_old+rowsum(Pi)
                compute Oi=diag(exp(mi_old-mi))^-1 Oi_old + Pi Vj
            Oi=diag(li)^-1 Oi
            Write Oi
        """
        q_shape=input_shapes[0]
        k_shape=input_shapes[1]
        v_shape=input_shapes[2]
        batch_size=input_shapes[0][0]
        n_heads=input_shapes[0][1]
        onchip_buffer=extra_args["onchip_buffer"]
        kv_length=extra_args.get("kv_seqlen")
        head_size=input_shapes[0][-1]
        q_length=input_shapes[0][-2]
        all_kv_length=input_shapes[1][-1]+kv_length
        block_size_r = min(
            math.ceil(onchip_buffer / (4 * head_size)), q_length
        )
        block_size_c=min(math.ceil(onchip_buffer / (4 * head_size)),all_kv_length)
        n_blocks_r = math.ceil(q_length / block_size_r)
        n_blocks_c = math.ceil(all_kv_length / block_size_c)
        extra_OPs=0
        extra_OPs=block_size_r+block_size_r*block_size_c # max
        extra_OPs+=block_size_r*block_size_c*2 # exp
        extra_OPs+=block_size_r*block_size_c+block_size_r*3 # li=exp(mi_old-mi)li_old+rowsum(Pi)
        extra_OPs+=block_size_r*block_size_c*2+block_size_r*3 # diag(exp(mi_old-mi))^-1 Oi_old
        extra_OPs*=n_blocks_c*n_blocks_r
        extra_OPs+=n_blocks_r*block_size_r*head_size*2 # O inittialize and div
        extra_OPs*=(batch_size*n_heads)

        qk_matmul_OPs=q_shape[-1] * np.prod(q_shape[:-1])*all_kv_length * 2
        output_shape=[batch_size,n_heads,q_shape[-2],v_shape[-1]]
        sv_matmul_OPs=np.prod(output_shape)*all_kv_length*2
        rst={
            "OPs": qk_matmul_OPs+sv_matmul_OPs+extra_OPs,
            "n_load_weight": 0,
            "n_load_act": np.prod(q_shape)+n_blocks_r*(np.prod(k_shape)+np.prod(v_shape)),
            "n_store_act": np.prod(output_shape)*n_blocks_r,
            "n_load_kv_cache":n_blocks_r*(batch_size*n_heads*head_size*kv_length*2),
            "output_shape": output_shape
        }
        return rst