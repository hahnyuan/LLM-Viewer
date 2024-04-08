from modifier.base_modifier import BaseModifier
import numpy as np
import math

class FlashAttention(BaseModifier):
    """
    Transform qk_matmul softmax sv_matmul into one fused_attention layer
    """
    def __init__(self,onchip_buffer,kv_bit, qk_matmul="qk_matmul", sv_matmul="sv_matmul", softmax="softmax") -> None:
        super().__init__()
        self.onchip_buffer=onchip_buffer
        self.kv_bit=kv_bit
        self.qk_matmul = qk_matmul
        self.sv_matmul = sv_matmul
        self.softmax = softmax
        
        
    def run(self,analyze_rsts):
        n_load_q=None
        qk_matmul_OPs=None
        sv_matmul_OPs=None
        softmax_OPs=None
        n_load_kv_cache=0
        raise NotImplementedError("There may be some problem in KV cache understanding")
        
        for name, (node, node_info) in analyze_rsts.items():
            if '.' in name:
                name=name.split('.')[1]
            if name==self.qk_matmul:
                qk_matmul_OPs=node_info["OPs"]
                n_load_q=node_info["n_load_act"]
                n_load_kv_cache+=node_info["n_load_kv_cache"]
            if name==self.softmax:
                softmax_OPs=node_info["OPs"]
                s_numel=np.prod(node_info["output_shape"])
            if name==self.sv_matmul:
                sv_matmul_OPs=node_info["OPs"]
                head_size=node_info["output_shape"][1]
                block_size_r = min(
                    math.ceil(self.onchip_buffer / ((self.kv_bit/8) * head_size)), head_size
                )
                n_blocks_r = math.ceil(1 / block_size_r)
                fused_attention_node_info={
                    "OPs": qk_matmul_OPs+softmax_OPs+sv_matmul_OPs,
                    "n_load_weight": 0,
                    "n_load_act": n_load_q,
                    "n_store_act": node_info["n_store_act"]*2,  # initialize O and save O
                    "n_load_kv_cache":n_blocks_r,
                    "output_shape": node_info["output_shape"]
                }
                node_info["n_load_kv_cache"] = node_info["n_load_act"]-s_numel
                node_info["n_load_act"] = s_numel
                
