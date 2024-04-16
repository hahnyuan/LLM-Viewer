from hardwares.hardware_params import hardware_params
import re
from backend_settings import avaliable_model_ids_sources
from hardwares.roofline_model import get_roofline_model
from utils import numpy_value_to_python

config_cache = {}

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

    parser_cls, analyzer_cls = avaliable_model_ids_sources[model_id]
    network = parser_cls(model_id, kwargs).parse()
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

    network_graph, module_graphs=analyzer.get_ui_graph(result)
    network_results = numpy_value_to_python(result["network"])
    return network_graph, module_graphs, network_results, hardware_info
