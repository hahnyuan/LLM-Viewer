from modifier.base_modifier import BaseModifier
import numpy as np
import math

class FlashAttention(BaseModifier):
    """
    Transform qk_matmul softmax sv_matmul into one fused_attention layer
    FlashAttention V2 Source: https://arxiv.org/pdf/2307.08691.pdf
    FlashDecoding Source: https://crfm.stanford.edu/2023/10/12/flashdecoding.html
    """
    def __init__(self,onchip_buffer, qk_matmul="qk_matmul", sv_matmul="sv_matmul", softmax="softmax") -> None:
        super().__init__()
        self.onchip_buffer=onchip_buffer
        self.qk_matmul = qk_matmul
        self.sv_matmul = sv_matmul
        self.softmax = softmax
        
        
    def run(self,analyze_rsts):
        """
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
        n_load_q=None
        qk_matmul_OPs=None
        sv_matmul_OPs=None
        q_length=None
        kv_length=None
        n_load_kv_cache=0
        replace_node_pair=[]
        old_node_names=[]
        new_node_input_names=[]
        for raw_name, (node, node_info) in analyze_rsts.items():
            if '.' in raw_name:
                name=raw_name.split('.')[1]
            else:
                name=raw_name
            if name==self.qk_matmul:
                qk_matmul_OPs=node_info["OPs"]
                n_load_q=node_info["n_load_act"]
                n_load_kv_cache+=node_info["n_load_kv_cache"]
                old_node_names.append(raw_name)
                new_node_input_names.extend(node.input_names)
            if name==self.softmax:
                q_length=node_info["output_shape"][-2]
                kv_length=node_info["output_shape"][-1]
                old_node_names.append(raw_name)
            if name==self.sv_matmul:
                batch_size=node_info["output_shape"][0]
                n_heads=node_info["output_shape"][1]
                sv_matmul_OPs=node_info["OPs"]
                head_size=node_info["output_shape"][-1]
                n_load_kv_cache+=node_info["n_load_kv_cache"]
                old_node_names.append(raw_name)
                new_node_input_names.append(node.input_names[1])

                # build new node
                block_size_r = min(
                    math.ceil(self.onchip_buffer / (4 * head_size)), head_size
                )
                block_size_c=math.ceil(self.onchip_buffer / (4 * head_size))
                n_blocks_r = math.ceil(q_length / block_size_r)
                n_blocks_c = math.ceil(kv_length / block_size_c)
                extra_OPs=block_size_r+block_size_r*block_size_c # max
                extra_OPs+=block_size_r*block_size_c*2 # exp
                extra_OPs+=block_size_r*block_size_c+block_size_r*3 # li=exp(mi_old-mi)li_old+rowsum(Pi)
                extra_OPs+=block_size_r*block_size_c*2+block_size_r*3 # diag(exp(mi_old-mi))^-1 Oi_old
                extra_OPs*=n_blocks_c*n_blocks_r
                extra_OPs+=n_blocks_r*block_size_r*head_size*2 # O inittialize and div
                extra_OPs*=(batch_size*n_heads)
                fused_attention_node_info={
                    "OPs": qk_matmul_OPs+sv_matmul_OPs+extra_OPs,
                    "n_load_weight": 0,
                    "n_load_act": n_load_q,
                    "n_store_act": node_info["n_store_act"]*n_blocks_r,
                    "n_load_kv_cache":n_blocks_r*n_load_kv_cache,
                    "output_shape": node_info["output_shape"]
                }
                replace_node_pair.append(([_ for _ in old_node_names],fused_attention_node_info, [_ for _ in new_node_input_names]))
                old_node_names.clear()
                new_node_input_names.clear()
        for old_node_names,new_node_info,new_node_input_names in replace_node_pair:
            for old_node_name in old_node_names:
                analyze_rsts.pop(old_node_name)
            new_name=old_node_names[0].replace(self.qk_matmul,"fused_attention")
            analyze_rsts[new_name]=(None,new_node_info)


            
                
