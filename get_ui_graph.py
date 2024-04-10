from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import importlib
import os
from hardwares.hardware_params import hardware_params
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

# def get_analyer(model_id, hardware, config_path) -> ModelAnalyzer:
#     config = f"{model_id}_{hardware}_{config_path}"
#     if config not in config_cache:
#         config_cache[config] = ModelAnalyzer(
#             model_id,
#             hardware,
#             config_path,
#             source=avaliable_model_ids_sources[model_id]["source"],
#         )
#     return config_cache[config]


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


def analyze_get_ui_graph(model_id, hardware, frontend_params_info):

    kwargs={}
    for param in frontend_params_info:
        value=param.get("value",param["default"])
        if param["type"]=="int":
            value=int(value)
        kwargs[param["name"]]=value

    network_func, analyzer_cls = avaliable_model_ids_sources[model_id]
    if "use_flashattention" in kwargs:
        network = network_func(model_id,use_flashattention=kwargs["use_flashattention"])
    else:
        network = network_func(model_id)
    hardware_model=get_roofline_model(hardware)
    analyzer=analyzer_cls(network,hardware_model)

    # Roofline model
    result = analyzer.analyze(
        **kwargs
    )
    hardware_info = hardware_model.params
    if "compute_dtype" in kwargs:
        hardware_info["max_OPS"]=hardware_info[kwargs["compute_dtype"]]
    else:
        hardware_info["max_OPS"]=hardware_info["FP16"]


    module_graphs={}
    network_nodes=[]
    network_edges=[]

    for module in network.modules:
        module_name=module.name
        nodes = []
        edges = []


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
            name=f"{module_name}.{node.name}"
            if name not in result["layers"]:
                continue
            info=result["layers"][name][1]
            info["layer_type"]=node.__class__.__name__
            numpy_value_to_python(info)
            
            nodes.append({
                "label": node.name,
                "id": node.name,
                "description": f"OPs:{str_number(info['OPs'])}, Access:{str_number_1024(info['memory_access'])}B",
                "info": info,
            })
            for input_name in node.input_names:
                edge = {"source": input_name, "target": node.name}
                edges.append(edge)

        module_graphs[module_name] = {"nodes": nodes, "edges": edges}

        network_node={
            "label": module_name,
            "id": module_name
        }
        network_nodes.append(network_node)
        for input_name in module.input_names:
            if '.' in input_name:
                source=input_name.split('.')[0]
                edge = {"source": source, "target": module_name}
                network_edges.append(edge)
    network_graph={"nodes":network_nodes,"edges":network_edges}
    network_results = numpy_value_to_python(result["network"])
    return network_graph,module_graphs, network_results, hardware_info
