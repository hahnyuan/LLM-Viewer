import os
import numpy as np

from ..base_parser import BaseParser
from net_graph.network import OnnxNetwork


class OnnxParser(BaseParser):
    def __init__(self, model_id, args: dict):
        super().__init__(model_id, args)
        import onnx
        from .base_graph import OnnxParseConfig
        from .graph import Graph
        
        self.model_config={"constant_folding": args.get('constant_folding',True),
                            "node_rename": args.get('node_rename',False),}
        self.cfg = OnnxParseConfig(self.model_config)

        self.model_path = args.get("model_path", "")
        print(f'model_path: {self.model_path}')
        assert os.path.exists(self.model_path), "model path not exist"
        self.mproto = onnx.load_model(self.model_path)
        
        if args.get("onnx_sim", False):
            from onnxsim import simplify
            print("==========simplify onnx begin===========")
            model_sim, check = simplify(self.mproto)
            assert check, "Simplified ONNX model could not be validated"
            print("==========simplify onnx over===========")
            self.mproto = model_sim
            self.graph = Graph(model_sim.graph, self.cfg)
        else:
            self.graph = Graph(self.mproto.graph, self.cfg)    

    def parse(self):
        return OnnxNetwork(self.graph)

    def save_model(self, f: str, shape_only: bool = False, no_shape: bool = False):
        self.graph.save_model(f, shape_only=shape_only, rawmodel=self.mproto, no_shape=no_shape)
