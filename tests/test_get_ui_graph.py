from get_ui_graph import analyze_get_ui_graph
from analyzers.llm_analyzer import LLMAnalyzer
import copy

def test_dummy_llm():
    inference_config = copy.deepcopy(LLMAnalyzer.frontend_params_info)
    model_id = "meta-llama/Llama-2-70b-hf"
    hardware = "nvidia_A6000"
    network_graph, modules, network_results, hardware_info = analyze_get_ui_graph(
        model_id,
        hardware,
        inference_config,
    )
    print(modules)
    print(network_results)
    print(hardware_info)