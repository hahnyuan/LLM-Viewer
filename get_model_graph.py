from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import importlib
import os
from hardwares.hardware_params import hardware_params
from model_analyzer import ModelAnalyzer
from utils import str_number

config_cache={}

def get_analyer(model_id,hardware,config_path)->ModelAnalyzer:
    config=f"{model_id}_{hardware}_{config_path}"
    if config not in config_cache:
        config_cache[config]=ModelAnalyzer(model_id,hardware,config_path)
    return config_cache[config]



# def get_model_config(model_id,config_path):
#     if model_id not in config_cache:
#         model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
#         config = importlib.import_module(config_path.replace("/", ".").replace(".py", ""))
#         config_cache[model_id] = model_config,config
#     return config_cache[model_id]

def get_model_graph(model_id,hardware,config_path,inference_config):
    
    # Roofline model
    w_bit = 16
    a_bit = 16
    kv_bit=16
    analyzer=get_analyer(model_id,hardware,config_path)
    result=analyzer.analyze(seqlen=2048,batchsize=1,w_bit=w_bit,a_bit=a_bit,kv_bit=kv_bit)
    stage=inference_config['stage']
    result=result[stage]
        
    nodes=[{"label":"input", "id": "input", "panels":[{"title":"OPs","value":"0"},{"title":"Access","value":"0"}]}]
    edges=[]

    def write_to_node(name,OPs,memory_access,input_names=[]):
        node={"label":name, "id": name, "description":f"OPs:{str_number(OPs)}, Access:{str_number(memory_access)}", "panels":[{"title":"OPs","value":str_number(OPs)},{"title":"Access","value":str_number(memory_access)}]}
        nodes.append(node)
        for input_name in input_names:
            edge={"source":input_name,"target":name}
            edges.append(edge)

    for name, input_names in analyzer.config.transformer_layer_graph.items():
        if name in ["input","output"]:
            OPs=0
            memory_access=0
        else:            
            OPs=result[name]['OPs']
            memory_access=result[name]['memory_access']
        write_to_node(name,OPs,memory_access,input_names)
    return nodes,edges