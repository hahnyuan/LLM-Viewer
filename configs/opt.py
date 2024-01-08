# num_attention_heads_config = "num_attention_heads"
# hidden_size_config = "hidden_size"
# num_key_value_heads_config = "num_attention_heads"
# num_hidden_layers_config = "num_hidden_layers"

def get_num_attention_heads(config):
    return getattr(config, "num_attention_heads")

def get_hidden_size(config):
    return getattr(config, "hidden_size")

def get_num_key_value_heads(config):
    return getattr(config, "num_attention_heads")

def get_num_hidden_layers(config):
    return getattr(config, "num_hidden_layers")