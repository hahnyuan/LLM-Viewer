def get_num_attention_heads(model_params):
    return getattr(model_params, "num_attention_heads")

def get_hidden_size(model_params):
    return getattr(model_params, "hidden_size")

def get_num_key_value_heads(model_params):
    if getattr(model_params,"multi_query_attention"):
        return getattr(model_params, "multi_query_group_num")
    else:
        return getattr(model_params, "num_attention_heads")

def get_num_hidden_layers(model_params):
    return getattr(model_params, "num_layers")

def get_intermediate_size(model_params):
    return getattr(model_params, "ffn_hidden_size")

def get_vocab_size(model_params):
    return getattr(model_params, "padded_vocab_size")

def get_linear_layers(model_params):
    hidden_size=get_hidden_size(model_params)
    intermediate_size=get_intermediate_size(model_params)
    key_value_heads=get_num_key_value_heads(model_params)
    attention_heads=get_num_attention_heads(model_params)
    return {
        "q_proj":[hidden_size, hidden_size],
        "k_proj":[hidden_size, hidden_size*key_value_heads/attention_heads],
        "v_proj":[hidden_size, hidden_size*key_value_heads/attention_heads],
        "out_proj":[hidden_size, hidden_size],
        "gate_proj":[hidden_size, intermediate_size],
        "up_proj":[hidden_size,intermediate_size],
        "down_proj":[intermediate_size, hidden_size]
    }

from configs.Llama import flashattention_transformer_layer_graph,transformer_layer_graph