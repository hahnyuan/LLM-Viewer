from modifier.base_modifier import BaseModifier
import numpy as np

default_memory_keys=["load_act","load_weight","store_act","load_kv_cache","store_kv_cache"]

class CalcMemoryAccess(BaseModifier):
    def __init__(self, keys=default_memory_keys) -> None:
        super().__init__()
        self.keys = keys
        
    def run(self,analyze_rsts):
        for name, (node, node_info) in analyze_rsts.items():
            memory_access=0
            for key in self.keys:
                if key in node_info:
                    memory_access+=node_info[key]
            node_info["memory_access"]=memory_access
                
