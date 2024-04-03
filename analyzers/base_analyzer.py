
class BaseAnalyzer():
    def __init__(self, net_graph, hardware_model) -> None:
        """
        net_graph can be either Model or ModelPipeline
        """
        self.net_graph = net_graph
        self.hardware_model = hardware_model

    def node_analyze(
        self,
        node
    ):
        op_info=node.get_op_info()
        inference_time,extra_info=self.hardware_model.analyze(op_info)
        self.results[stage][name] = {
            "OPs": OPs,
            "memory_access": memory_access,
            "arithmetic_intensity": arithmetic_intensity,
            "performance": performance,
            "bound": bound,
            "load_weight": load_weight,
            "load_act": load_act,
            "store_act": store_act,
            "load_kv_cache": load_kv_cache,
            "store_kv_cache": store_kv_cache,
            "inference_time": inference_time,
        }
    
    def graph_anlyze(self):
        for node in self.net_graph.nodes:
            self.node_analyze(node)
        return self.results
        
    def set_hooks(self, model):
        self.model = model

    def analyze(self,**kwargs):
        raise NotImplementedError