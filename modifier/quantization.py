from modifier.base_modifier import BaseModifier

class QuantAct(BaseModifier):
    def __init__(self, bitwidth) -> None:
        super().__init__()
        self.bitwidth=bitwidth
        

    def modify_node(self, node, node_info):
        node_info["load_act"] = node_info.get("n_load_act",0) * self.bitwidth / 8
        node_info["store_act"] = node_info.get("n_store_act",0) * self.bitwidth / 8
    
    def run(self,analyze_rsts):
        for name, (node, node_info) in analyze_rsts.items():
            self.modify_node(node, node_info)


class QuantWeight(QuantAct):
    def modify_node(self, node, node_info):
        node_info["load_weight"] = node_info.get("n_load_weight",0) * self.bitwidth / 8

class  QuantKV(QuantAct):
    def modify_node(self, node, node_info):
        if "n_load_kv_cache" in node_info:
            node_info["load_kv_cache"] = node_info.get("n_load_kv_cache",0) * self.bitwidth / 8
        if "n_store_kv_cache" in node_info:
            node_info["store_kv_cache"] = node_info.get("n_store_kv_cache",0) * self.bitwidth / 8