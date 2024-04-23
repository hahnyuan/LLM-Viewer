from typing import Dict
from net_graph.module import Node
import numpy as np
import math
from net_parsers.onnx.node import _conv_output_shape, RESIZE_LINEAR_MACS, RESIZE_CUBIC_MACS


# NOTICE: a MAC(multiply-accumulate) should be counted as 2 OPs

# 
# GLM
# QWen
# DIT

class Conv2d(Node):
    """
    attr:
    - out_features: int
    """
    def analyze_node(self,input_shapes, extra_args):
        input_shape=input_shapes[0]

        output_h = _conv_output_shape(input_shape[2], 2*self.padding[0], self.ksize[0], self.stride[0],
                                self.dilations[0])
        output_w = _conv_output_shape(input_shape[3], 2*self.padding[1], self.ksize[1], self.stride[1],
                                self.dilations[1])
        output_shape = [input_shape[0], self.out_channels, output_h, output_w]

        rst={
            "OPs": np.prod(output_shape)*self.ksize[0]*self.ksize[1]*input_shape[1]*2//self.groups,
            "n_weight": self.ksize[0]*self.ksize[1]*input_shape[1]*self.out_channels,
            "n_load_weight": self.ksize[0]*self.ksize[1]*input_shape[1]*self.out_channels,
            "n_load_act": np.prod(input_shape),
            "n_store_act": np.prod(output_shape),
            "output_shape": output_shape
        }
        return rst

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
        assert len(a_shape)==len(b_shape)
        assert a_shape==b_shape or [a%b for a,b in zip(a_shape,b_shape)]==[0]*len(a_shape)
        output_shape=[max(a,b) for a,b in zip(a_shape,b_shape)]
        rst={
            "OPs": max(np.prod(a_shape),np.prod(b_shape)),
            "n_load_weight": 0,
            "n_load_act": np.prod(a_shape)+np.prod(b_shape),
            "n_store_act": max(np.prod(a_shape),np.prod(b_shape)),
            "output_shape": output_shape
        }
        return rst

class Concat(Node):
    def analyze_node(self, input_shapes, extra_args):
        a_shape,b_shape=input_shapes
        cat_dim = self.attrs['dim']
        assert len(a_shape)==len(b_shape)
        assert [a_shape[i]==b_shape[i] for i in range(len(a_shape)) if i!=cat_dim]==[1]*(len(a_shape)-1)

        output_shape = []
        for i in range(len(a_shape)):
            if i==cat_dim:
                output_shape.append(a_shape[i]+b_shape[i])
            else:
                output_shape.append(a_shape[i])

        rst={
            "OPs": 0,
            "n_load_weight": 0,
            "n_load_act": np.prod(a_shape)+np.prod(b_shape),
            "n_store_act": np.prod(output_shape),
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
    """RMS Norm
    x = x / tf.sqrt(x.pow(2).mean + eps)  # ops: 6
    x = x * gamma  # ops: 1
    """
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

class GroupNorm(Node):
    """
    mean = x.mean  # ops: 2
    var = (x-mean).pow(2).mean  # ops: 4
    x = (x - mean) / tf.sqrt(var + eps)  # ops: 4
    x = x * gamma + beta  # ops: 2
    """
    def analyze_node(self,input_shapes, extra_args):
        input_shape=input_shapes[0]
        output_shape=input_shape
        rst={
            "OPs": np.prod(input_shape)*12,
            "n_load_act": np.prod(input_shape),
            "n_store_act": np.prod(input_shape),
            "output_shape": output_shape
        }
        return rst

class LayerNorm(Node):
    """
    mean = x.mean  # ops: 2
    var = (x-mean).pow(2).mean  # ops: 4
    x = (x - mean) / tf.sqrt(var + eps)  # ops: 4
    x = x * gamma + beta  # ops: 2
    """
    def analyze_node(self,input_shapes, extra_args):
        input_shape=input_shapes[0]
        output_shape=input_shape
        rst={
            "OPs": np.prod(input_shape)*12,
            "n_load_act": np.prod(input_shape),
            "n_store_act": np.prod(input_shape),
            "output_shape": output_shape
        }
        return rst

class SiLU(Node):
    """
    x = x * sigmoid(x), sigmoid=1/(1+e^-x)  # ops: 4
    """
    def analyze_node(self,input_shapes, extra_args):
        input_shape=input_shapes[0]
        output_shape=input_shape
        rst={
            "OPs": np.prod(input_shape)*4,
            "n_load_act": np.prod(input_shape),
            "n_store_act": np.prod(input_shape),
            "output_shape": output_shape
        }
        return rst

class GELU(Node):
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))  # ops: 5
    x = x*cdf  # ops: 1
    """
    def analyze_node(self,input_shapes, extra_args):
        input_shape=input_shapes[0]
        output_shape=input_shape
        rst={
            "OPs": np.prod(input_shape)*6,
            "n_load_act": np.prod(input_shape),
            "n_store_act": np.prod(input_shape),
            "output_shape": output_shape
        }
        return rst

class GEGLU(Node):
    """
    from stable diffusion:
        x = nn.Linear(x)
        x, gate = chunk(chunk_num, dim=chunk_dim)
        x * GELU(gate)
    ops: linear + GELU + mul
    """
    def analyze_node(self,input_shapes, extra_args):
        input_shape=input_shapes[0]
        # print(self.attrs['out_features'])

        output_shape=input_shape[:-1]+[self.attrs['out_features']//self.attrs['chunk_num']]
        rst={
            "OPs": np.prod(input_shape)*self.attrs['out_features']*2 + np.prod(input_shape)*6 + np.prod(input_shape),
            "n_load_weight": input_shape[-1]*self.attrs['out_features'],
            "n_load_act": np.prod(input_shape),
            "n_store_act": np.prod(output_shape)*self.attrs['chunk_num'],
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
    
class Upsample(Node):
    def analyze_node(self,input_shapes, extra_args):
        op_mac = 0
        if self.attrs['mode']=='nearest':
            op_mac = 0
        elif self.attrs['mode']=='bilinear':
            op_mac = RESIZE_LINEAR_MACS
        elif self.attrs['mode']=='cubic':
            op_mac = RESIZE_CUBIC_MACS
        
        input_shape=input_shapes[0]
        output_shape=input_shape[:2]+[input_shape[2]*self.attrs['ratio'][0], input_shape[3]*self.attrs['ratio'][1]]
        rst={
            "OPs": np.prod(output_shape)*op_mac,
            "n_load_act": np.prod(input_shape),
            "n_store_act": np.prod(output_shape),
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