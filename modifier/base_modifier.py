
class BaseModifier():
    def __init__(self) -> None:
        pass
    
    def run(self,analyze_rsts):
        for name, (node, node_info) in analyze_rsts.items():
            # modify
            pass