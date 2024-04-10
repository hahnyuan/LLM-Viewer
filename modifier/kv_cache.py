from modifier.base_modifier import BaseModifier
import numpy as np

class MakeKVLoadStore(BaseModifier):
    def __init__(self, qk_matmul="qk_matmul", sv_matmul="sv_matmul", softmax="softmax", q_proj="q_proj", k_proj="k_proj",v_proj="v_proj") -> None:
        super().__init__()
        self.qk_matmul = qk_matmul
        self.sv_matmul = sv_matmul
        self.softmax = softmax
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        
    def run(self,analyze_rsts):
        q_numel=None
        s_numel=None
        for name, (node, node_info) in analyze_rsts.items():
            if '.' in name:
                name=name.split('.')[1]
            if name==self.q_proj:
                q_numel=np.prod(node_info["output_shape"])
            if name==self.k_proj or name==self.v_proj:
                node_info["n_store_kv_cache"] = node_info["n_store_act"]
                node_info["n_store_act"] = 0
            if name==self.qk_matmul:
                node.analyze_node(node.input_shapes)
                node_info["n_load_kv_cache"] = node_info["n_load_act"]-q_numel
                node_info["n_load_act"] = q_numel
                node_info["n_store_kv_cache"] = node_info["n_store_act"]
            if name==self.softmax:
                s_numel=np.prod(node_info["output_shape"])
            if name==self.sv_matmul:
                node_info["n_load_kv_cache"] = node_info["n_load_act"]-s_numel
                node_info["n_load_act"] = s_numel

class AddDecodeKVLoad(BaseModifier):
    def __init__(self, kv_seqlen,n_parallel_decode) -> None:
        super().__init__()
        self.kv_seqlen = kv_seqlen
        self.n_parallel_decode = n_parallel_decode
    
    def run(self,analyze_rsts):
        for name, (node, node_info) in analyze_rsts.items():
            if "n_load_kv_cache" in node_info:
                node_info["n_load_kv_cache"] += node_info["n_load_kv_cache"]//self.n_parallel_decode * self.kv_seqlen
                node_info["OPs"]*=self.kv_seqlen
