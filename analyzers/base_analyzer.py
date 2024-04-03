
class BaseAnalyzer():
    def __init__(self, net_graph, hardware_model) -> None:
        """
        net_graph can be either Model or ModelPipeline
        """
        self.net_graph = net_graph
        self.hardware_model = hardware_model

        
    def set_hooks(self, model):
        pass

    def analyze(self,**kwargs):
        raise NotImplementedError