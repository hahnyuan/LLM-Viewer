from net_parsers.manual.Llama import get_llama_network_graph
from transformers import AutoConfig


def get_chatglm_network_graph(model_id, manual_params=None,use_flashattention=False):
    if manual_params is not None:
        p=manual_params
    else:
        p=AutoConfig.from_pretrained(model_id,trust_remote_code=True)
    
    if getattr(p,"multi_query_attention"):
        p.num_key_value_heads= getattr(p, "multi_query_group_num")
    else:
        p.num_key_value_heads= getattr(p, "num_attention_heads")
    p.num_hidden_layers=getattr(p, "num_layers")
    p.intermediate_size=getattr(p, "ffn_hidden_size")
    p.vocab_size=getattr(p, "padded_vocab_size")
    
    network=get_llama_network_graph(model_id, manual_params=p, use_flashattention=use_flashattention)
    return network


