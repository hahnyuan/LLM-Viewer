import os
from analyzers.onnx_analyzer import OnnxAnalyzer
from hardwares.roofline_model import get_roofline_model
from net_parsers.onnx.onnx_parser import OnnxParser


def test_onnx_analyzer():
    root_path = "data/onnx"
    model_path = f'{root_path}/resnet50-v2-7.onnx'
    model_config = {"constant_folding": True}
    network = OnnxParser(model_id=None, args={"model_path":model_path, **model_config}).parse()
    hardware_model=get_roofline_model("nvidia_A6000")

    analyzer=OnnxAnalyzer(network,hardware_model)
    profile_info=analyzer.analyze(
                        model_path="",
                        input_shape_info="data:1,3,224,224",
                        w_bit=16,
                        a_bit=16,
                        compute_dtype="FP16")
    # print(profile_info)
    analyzer.get_ui_graph(profile_info)

def test_dir_onnx_analyzer():
    root_path = "data/onnx"
    for file in os.listdir(root_path):
        if file.endswith(".onnx"):
            model_path = f'{root_path}/{file}'
            model_config = {"constant_folding": True}
            network = OnnxParser(model_id=None, args={"model_path":model_path, **model_config}).parse()
            hardware_model=get_roofline_model("nvidia_A6000")

            analyzer=OnnxAnalyzer(network,hardware_model)
            profile_info=analyzer.analyze(
                                model_path="",
                                input_shape_info="",
                                # input_shape_info="data:1,3,224,224",
                                w_bit=16,
                                a_bit=16,
                                compute_dtype="FP16")
            # print(profile_info)
            analyzer.get_ui_graph(profile_info)
            # network.print_graph()