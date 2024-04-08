from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import importlib
import os
from hardwares.hardware_params import hardware_params
from model_analyzer import ModelAnalyzer
from utils import str_number,str_number_1024
import re
from backend_settings import avaliable_model_ids_sources
from hardwares.roofline_model import get_roofline_model
import numpy as np

config_cache = {}

def numpy_value_to_python(d):
    for key in d:
        if isinstance(d[key],np.ndarray):
            d[key]=d[key].tolist()
        if isinstance(d[key],np.int64):
            d[key]=int(d[key])
        if isinstance(d[key],np.int32):
            d[key]=int(d[key])
        if isinstance(d[key],np.float32):
            d[key]=float(d[key])
        if isinstance(d[key],np.float64):
            d[key]=float(d[key])
    return d

def get_analyer(model_id, hardware, config_path) -> ModelAnalyzer:
    config = f"{model_id}_{hardware}_{config_path}"
    if config not in config_cache:
        config_cache[config] = ModelAnalyzer(
            model_id,
            hardware,
            config_path,
            source=avaliable_model_ids_sources[model_id]["source"],
        )
    return config_cache[config]


def get_quant_bit(dtype):
    if isinstance(dtype,(int,float)):
        return dtype
    if dtype == "FP16":
        return 16
    elif dtype == "INT8":
        return 8
    elif dtype == "INT4":
        return 4
    elif "bit" in dtype:
        bitwidth = int(re.findall(r"\d+", dtype)[0])
        return bitwidth
    else:
        raise ValueError(f"Unsupported dtype:{dtype}")


def analyze_get_ui_graph(model_id, hardware, inference_config):

    network_func, analyzer_cls = avaliable_model_ids_sources[model_id]
    network = network_func(model_id)
    hardware_model=get_roofline_model(hardware)
    analyzer=analyzer_cls(network,hardware_model)

    # Roofline model
    stage = inference_config["stage"]
    w_bit = get_quant_bit(inference_config["w_quant"])
    a_bit = get_quant_bit(inference_config["a_quant"])
    kv_bit = get_quant_bit(inference_config["kv_quant"])
    seq_length = int(inference_config["seq_length"])
    batch_size = int(inference_config["batch_size"])
    n_parallel_decode = int(inference_config["n_parallel_decode"])
    use_flashattention = bool(inference_config["use_flashattention"])
    gen_length = int(inference_config["gen_length"])

    result = analyzer.analyze(
        seqlen=seq_length,
        batchsize=batch_size,
        stage=stage,
        w_bit=w_bit,
        a_bit=a_bit,
        kv_bit=kv_bit,
        use_flashattention=use_flashattention,
        n_parallel_decode=n_parallel_decode,
    )
    hardware_info = hardware_model.params

    nodes = []
    edges = []

    def write_to_node(name, OPs, memory_access, info, input_names=[]):
        node = {
            "label": name,
            "id": name,
            "description": f"OPs:{str_number(OPs)}, Access:{str_number_1024(memory_access)}B",
            "info": info,
        }
        nodes.append(node)
        for input_name in input_names:
            edge = {"source": input_name, "target": name}
            edges.append(edge)

    
    # TODO: select module in GUI
    module=network.modules[1]
    prefix="transformer_layer0"

    for input_name in module.input_names:
        if '.' in input_name:
            source,show_input_name=input_name.split('.')
        else:
            show_input_name=input_name
            source="input"
        input_node = {
            "label": show_input_name,
            "description": f"from: {source}",
            "id": input_name,
        }
        nodes.append(input_node)

    for node in module.nodes:
        name=f"{prefix}.{node.name}"
        info=result["layers"][name][1]
        numpy_value_to_python(info)
            
        write_to_node(node.name, info["OPs"], info["memory_access"], info, node.input_names)

    network_results = numpy_value_to_python(result["network"])
    return nodes, edges, network_results, hardware_info
