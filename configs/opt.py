from transformers import OPTForCausalLM

def get_num_attention_heads(config):
    return getattr(config, "num_attention_heads")

def get_hidden_size(config):
    return getattr(config, "hidden_size")

def get_num_key_value_heads(config):
    return getattr(config, "num_attention_heads")

def get_num_hidden_layers(config):
    return getattr(config, "num_hidden_layers")

def get_intermediate_size(config):
    return getattr(config, "ffn_dim")

def get_vocab_size(config):
    return getattr(config, "vocab_size")

def get_lm_haed_layer(config):
    return {
        'lm_haead':[get_hidden_size(config),get_vocab_size(config)]
    }

def get_linear_layers(config):
    hidden_size=get_hidden_size(config)
    intermediate_size=get_intermediate_size(config)
    key_value_heads=get_num_key_value_heads(config)
    attention_heads=get_num_attention_heads(config)
    return {
        "q_proj":[hidden_size, hidden_size,"self_attn_norm"],
        "k_proj":[hidden_size, hidden_size*key_value_heads/attention_heads,"self_attn_norm"],
        "v_proj":[hidden_size, hidden_size*key_value_heads/attention_heads,"self_attn_norm"],
        "out_proj":[hidden_size, hidden_size,"sv_matmul"],
        "gate_proj":[hidden_size, intermediate_size,"mlp_norm"],
        "up_proj":[hidden_size,intermediate_size,"mlp_norm"],
        "down_proj":[intermediate_size, hidden_size,"mlp_act"],
    }
