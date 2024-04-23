import os
from analyzers.onnx_analyzer import OnnxAnalyzer
from hardwares.roofline_model import get_roofline_model
from net_parsers.onnx.onnx_parser import OnnxParser


def test_onnx_analyzer():
    root_path = "data/onnx"
    # model_path = f'{root_path}/diffusion_zero_param.onnx'
    # model_path = f'/data01/user/luqiang/projects/onnx_files/sd_1_4/unet/model.onnx'
    # model_path = f'{root_path}/diffusion_zero_param_sim.onnx'  # NOTE 权重全为0时，onnxsim会折叠非常多，不适合
    # model_path = f'{root_path}/vae_decoder_only.onnx'
    # model_path = f'{root_path}/vae_decoder_only_sim.onnx'
    model_config = {"constant_folding": True}
    # model_config = {"constant_folding": False}
    network = OnnxParser(model_id=None, args={"model_path":model_path, **model_config}).parse()
    hardware_model=get_roofline_model("nvidia_A6000")

    analyzer=OnnxAnalyzer(network,hardware_model)
    profile_info=analyzer.analyze(
                        model_path="",
                        input_shape_info="timestep:1;sample:1,3,512,512;encoder_hidden_states:1,1024,768",  # sd_1_4: unet
                        w_bit=16,
                        a_bit=16,
                        compute_dtype="FP16")
    print(profile_info['network'])
    analyzer.get_ui_graph(profile_info)

def test_sd_onnx_analyzer():
    root_path = "data/onnx"
    model_path = f'{root_path}/sd_1_4/unet/model.onnx'
    model_config = {"constant_folding": True}
    # model_config = {"constant_folding": False}
    network = OnnxParser(model_id=None, args={"model_path":model_path, **model_config}).parse()
    hardware_model=get_roofline_model("nvidia_A6000")

    analyzer=OnnxAnalyzer(network,hardware_model)
    profile_info=analyzer.analyze(
                        model_path="",
                        input_shape_info="timestep:1;sample:1,3,512,512;encoder_hidden_states:1,1024,768",  # sd_1_4: unet
                        w_bit=16,
                        a_bit=16,
                        compute_dtype="FP16")
    print(profile_info['network'])
    analyzer.get_ui_graph(profile_info)

def test_llama_onnx_analyzer():
    root_path = "data/onnx"
    model_path = f'{root_path}/llama-onnx/model.onnx'
    model_config = {"constant_folding": True}
    network = OnnxParser(model_id=None, args={"model_path":model_path, **model_config}).parse()
    hardware_model=get_roofline_model("nvidia_A6000")

    analyzer=OnnxAnalyzer(network,hardware_model)
    profile_info=analyzer.analyze(
                        model_path="",
                        input_shape_info="input_ids:1,1024;attention_mask:1,5;position_ids:1,1024;past_key_values.0.key:1,32,4,128;past_key_values.0.value:1,32,4,128;past_key_values.1.key:1,32,4,128;past_key_values.1.value:1,32,4,128;past_key_values.2.key:1,32,4,128;past_key_values.2.value:1,32,4,128;past_key_values.3.key:1,32,4,128;past_key_values.3.value:1,32,4,128;past_key_values.4.key:1,32,4,128;past_key_values.4.value:1,32,4,128;past_key_values.5.key:1,32,4,128;past_key_values.5.value:1,32,4,128;past_key_values.6.key:1,32,4,128;past_key_values.6.value:1,32,4,128;past_key_values.7.key:1,32,4,128;past_key_values.7.value:1,32,4,128;past_key_values.8.key:1,32,4,128;past_key_values.8.value:1,32,4,128;past_key_values.9.key:1,32,4,128;past_key_values.9.value:1,32,4,128;past_key_values.10.key:1,32,4,128;past_key_values.10.value:1,32,4,128;past_key_values.11.key:1,32,4,128;past_key_values.11.value:1,32,4,128;past_key_values.12.key:1,32,4,128;past_key_values.12.value:1,32,4,128;past_key_values.13.key:1,32,4,128;past_key_values.13.value:1,32,4,128;past_key_values.14.key:1,32,4,128;past_key_values.14.value:1,32,4,128;past_key_values.15.key:1,32,4,128;past_key_values.15.value:1,32,4,128;past_key_values.16.key:1,32,4,128;past_key_values.16.value:1,32,4,128;past_key_values.17.key:1,32,4,128;past_key_values.17.value:1,32,4,128;past_key_values.18.key:1,32,4,128;past_key_values.18.value:1,32,4,128;past_key_values.19.key:1,32,4,128;past_key_values.19.value:1,32,4,128;past_key_values.20.key:1,32,4,128;past_key_values.20.value:1,32,4,128;past_key_values.21.key:1,32,4,128;past_key_values.21.value:1,32,4,128;past_key_values.22.key:1,32,4,128;past_key_values.22.value:1,32,4,128;past_key_values.23.key:1,32,4,128;past_key_values.23.value:1,32,4,128;past_key_values.24.key:1,32,4,128;past_key_values.24.value:1,32,4,128;past_key_values.25.key:1,32,4,128;past_key_values.25.value:1,32,4,128;past_key_values.26.key:1,32,4,128;past_key_values.26.value:1,32,4,128;past_key_values.27.key:1,32,4,128;past_key_values.27.value:1,32,4,128;past_key_values.28.key:1,32,4,128;past_key_values.28.value:1,32,4,128;past_key_values.29.key:1,32,4,128;past_key_values.29.value:1,32,4,128;past_key_values.30.key:1,32,4,128;past_key_values.30.value:1,32,4,128;past_key_values.31.key:1,32,4,128;past_key_values.31.value:1,32,4,128;",
                        w_bit=16,
                        a_bit=16,
                        compute_dtype="FP16")
    analyzer.get_ui_graph(profile_info)

def test_resnet50_onnx_analyzer():
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
    # root_path = "/data01/model_zoo/swin_transformer"
    # root_path = "/data01/model_zoo/bevdet"
    # root_path = "/data01/model_zoo/bevformer"
    # root_path = "/data01/model_zoo/yolop"
    # root_path = "/data01/model_zoo/detr"
    # root_path = "/data01/model_zoo/shufflenet"
    # root_path = "/data01/model_zoo/pointpillars"
    # root_path = "/data01/model_zoo/resnet"
    # root_path = "/data01/model_zoo/unet"
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
                                w_bit=16,
                                a_bit=16,
                                compute_dtype="FP16")
            print(profile_info['network'])
            analyzer.get_ui_graph(profile_info)
            # network.print_graph()
