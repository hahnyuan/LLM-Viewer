import os
import onnx
import numpy as np
from onnxsim import simplify

from net_graph.network import OnnxNetwork
from .base_graph import OnnxParseConfig
from .graph import Graph


class OnnxParser:
    def __init__(self, model_path: [str], model_config={}):
        self.cfg = OnnxParseConfig(model_config)
        self.modelname = model_path.split('/')[-1]

        m = onnx.load_model(model_path)

        self.valid = True

        # # convert model
        # print("==========simplify onnx begin===========")
        # model_sim, check = simplify(m)
        # assert check, "Simplified ONNX model could not be validated"
        # print("==========simplify onnx over===========")
        # self.mproto = model_sim
        # self.graph = Graph(model_sim.graph, self.cfg)

        self.mproto = m
        self.graph = Graph(m.graph, self.cfg)

    def save_model(self, f: str, shape_only: bool = False, no_shape: bool = False):
        self.graph.save_model(f, shape_only=shape_only, rawmodel=self.mproto, no_shape=no_shape)


def get_onnx_network_graph(file_path, model_config={}):
    # NOTE if onnx input has dynamic axis, dummy_input is necessary in model_config
    m = OnnxParser(file_path, model_config=model_config)
    return OnnxNetwork(m.graph)


