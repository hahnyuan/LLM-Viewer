import os
from analyzers.onnx_analyzer import OnnxAnalyzer
from hardwares.roofline_model import get_roofline_model
from net_parsers.onnx.onnx_parser import get_onnx_network_graph

def test_onnx_analyzer():
    root_path = "data/onnx"

    # model_path = f'{root_path}/vae_decoder_only.onnx'
    for file in os.listdir(root_path):
        if file.endswith(".onnx"):
            model_path = f'{root_path}/{file}'
            model_config = {"constant_folding": True}
            # model_config = {"constant_folding": False}
            x_shape_dict = {}
            # test_onnx_analyzer(model_path, model_config, x_shape_dict)
            network = get_onnx_network_graph(model_path, model_config)
            hardware_model=get_roofline_model("nvidia_A6000")

            analyzer=OnnxAnalyzer(network,hardware_model)
            profile_info=analyzer.analyze(x_shape_dict=x_shape_dict,
                                w_bit=16,
                                a_bit=16,
                                compute_dtype="FP16")
            print(profile_info)
    # network.print_graph()

