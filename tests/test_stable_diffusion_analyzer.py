from net_graph.module import Module,Node
from net_parsers.manual.stable_diffusion import StableDiffusionParser
from analyzers.stable_diffusion_analyzer import StableDiffusionAnalyzer
from hardwares.roofline_model import get_roofline_model
from types import SimpleNamespace

def test_dummy_llm():
    network=StableDiffusionParser("meta-llama/Llama-2-7b-hf", {}).parse()
    hardware_model=get_roofline_model("nvidia_A6000")
    analyzer=StableDiffusionAnalyzer(network,hardware_model)
    rst=analyzer.analyze(256,1)
    network_graph, module_graphs=analyzer.get_ui_graph(rst)
    print(rst)

def test_dummy_manual_sd():
    params = {
        "batchsize": 1,
        "latent_size": 64,
        "tinme_steps": 20,
        "conditional_guidance": True,
        "w_bit": 16,
        "a_bit": 16,
        "compute_dtype": "FP16"
    }
    network=StableDiffusionParser("stable_diffusion", params).parse()
    hardware_model=get_roofline_model("nvidia_A6000")
    analyzer=StableDiffusionAnalyzer(network,hardware_model)
    rst=analyzer.analyze(**params)
    # print(rst)