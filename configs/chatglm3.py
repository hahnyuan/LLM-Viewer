def get_num_attention_heads(config):
    return getattr(config, "num_attention_heads")

def get_hidden_size(config):
    return getattr(config, "hidden_size")

def get_num_key_value_heads(config):
    if getattr(config,"multi_query_attention"):
        return getattr(config, "multi_query_group_num")
    else:
        return getattr(config, "num_attention_heads")

def get_num_hidden_layers(config):
    return getattr(config, "num_layers")

lm_head_name="output_layer"
