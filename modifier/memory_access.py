from modifier.base_modifier import BaseModifier
import numpy as np

default_memory_keys=["load_act","load_weight","store_act","load_kv_cache","store_kv_cache"]

class CalcMemoryAccess(BaseModifier):
    def __init__(self, keys=default_memory_keys) -> None:
        super().__init__()
        self.keys = keys

    def modify_node(self, node_info):
        memory_access=0
        for key in self.keys:
            if key in node_info:
                memory_access+=node_info[key]
        node_info["memory_access"]=memory_access

    def run(self,analyze_rsts):
        if list(analyze_rsts.keys()) == ["layers", "network"]:  # onnx profile_info
            for _, node_info in analyze_rsts["layers"].items():
                self.modify_node(node_info)
            self.modify_node(analyze_rsts["network"])
        else:
            for name, item in analyze_rsts.items():
                node, node_info = item
                self.modify_node(node_info)
                
