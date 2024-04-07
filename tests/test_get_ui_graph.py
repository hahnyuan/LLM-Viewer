from get_ui_graph import analyze_get_ui_graph

def test_dummy_llm():
    inference_config = {
        "hidden_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "intermediate_size": 512,
        "vocab_size": 1000,
        "num_hidden_layers": 1,
        "stage": "decode",
        "w_quant": 16,
        "a_quant": 16,
        "kv_quant": 16,
        "seq_length": 256,
        "batch_size": 1,
        "n_parallel_decode": 1,
        "use_flashattention": False,
        "gen_length": 1,
    }
    model_id = "georgesung/llama2_7b_chat_uncensored"
    hardware = "nvidia_A6000"
    nodes, edges, total_results, hardware_info = analyze_get_ui_graph(
        model_id,
        hardware,
        inference_config,
    )
    print(nodes)
    print(edges)
    print(total_results)
    print(hardware_info)