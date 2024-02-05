from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import importlib
import os
from hardwares.hardware_params import hardware_params
from roofline_model import roofline_analyze

def get_model_graph(model_id,batchsize,seqlen,config_path,decode=True):
    # Roofline model
    w_bit = 16
    a_bit = 16
    w_byte = w_bit / 8
    a_byte = a_bit / 8

    model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    config = importlib.import_module(config_path.replace("/", ".").replace(".py", ""))
    num_attention_heads = config.get_num_attention_heads(model_config)
    hidden_size = config.get_hidden_size(model_config)
    num_key_value_heads = config.get_num_key_value_heads(model_config)
    num_hidden_layers = config.get_num_hidden_layers(model_config)

    def str_number(num):
        if num > 1e12:
            return f"{num/1e12:.0f}T"
        elif num > 1e9:
            return f"{num/1e9:.0f}G"
        elif num > 1e6:
            return f"{num/1e6:.0f}M"
        elif num > 1e3:
            return f"{num/1e3:.0f}K"
        elif num < 10:
            return f"{num:.2f}"
        else:
            return f"{num:.0f}"
        
    nodes=[{"label":"input", "id": "input", "panels":[{"title":"OPs","value":"0"},{"title":"Access","value":"0"}]}]
    edges=[]

    def write_to_node(name,OPs,load_weight,load_act,store_act,load_kv_cache,store_kv_cache,input_names=[]):
        memory_access=load_weight+load_act+store_act+load_kv_cache+store_kv_cache
        node={"label":name, "id": name, "description":f"OPs:{str_number(OPs)}, Access:{str_number(memory_access)}", "panels":[{"title":"OPs","value":str_number(OPs)},{"title":"Access","value":str_number(memory_access)}]}
        nodes.append(node)
        for input_name in input_names:
            edge={"source":input_name,"target":name}
            edges.append(edge)

    for name, (ic,oc,input_names) in config.get_linear_layers(model_config).items():
        # for linear layers
        is_kv_proj = name in ["k_proj","v_proj"]
        is_normal_proj = not is_kv_proj
        if decode:
            write_to_node(
                name,
                OPs=ic*oc * batchsize * 2,
                load_weight=ic*oc * w_byte,
                load_act=ic * batchsize * a_byte,
                store_act=0 if is_kv_proj else oc * batchsize * a_byte,
                load_kv_cache=0,
                store_kv_cache=(
                    0 if is_normal_proj else oc * batchsize * a_byte
                ),
                input_names=[input_names]
            )
        else:
            # for prefill
            write_to_node(
                name,
                OPs=ic*oc * batchsize * seqlen * 2,
                load_weight=ic*oc * w_byte,
                load_act=ic * batchsize * seqlen * a_byte,
                store_act=(
                    0
                    if is_kv_proj
                    else oc * batchsize * seqlen * a_byte
                ),
                load_kv_cache=0,
                store_kv_cache=(
                    0
                    if is_normal_proj
                    else oc * batchsize * seqlen * a_byte
                ),
                input_names=[input_names]
            )

    # for attention
    head_size = hidden_size // num_attention_heads
    # for decode
    if decode:
        name = f"qk_matmul"
        write_to_node(
            name,
            OPs=seqlen * head_size * num_attention_heads * batchsize * 2,
            load_weight=0,
            load_act=(1) * head_size * batchsize * num_attention_heads * a_byte,
            store_act=1 * seqlen * batchsize * num_attention_heads * a_byte,
            load_kv_cache=(seqlen) * head_size * batchsize * num_attention_heads * a_byte,
            store_kv_cache=0,
            input_names=["q_proj","k_proj"]

        )
        name = f"sv_matmul"
        write_to_node(
            name,
            OPs=1 * head_size * seqlen * num_attention_heads * batchsize * 2,
            load_weight=0,
            load_act=(1 * seqlen * batchsize * num_attention_heads) * a_byte,
            store_act=1 * head_size * batchsize * num_attention_heads * a_byte,
            load_kv_cache=(seqlen * head_size * batchsize * num_attention_heads) * a_byte,
            store_kv_cache=0,
            input_names=["softmax","v_proj"]
        )

        name = f"softmax"
        # max sub exp sum div
        write_to_node(
            name,
            OPs=batchsize * num_attention_heads * seqlen * 1 * 5,
            load_weight=0,
            load_act=batchsize * num_attention_heads * seqlen * 1 * a_byte,
            store_act=batchsize * num_attention_heads * seqlen * 1 * a_byte,
            load_kv_cache=0,
            store_kv_cache=0,
            input_names=["qk_matmul"]
        )

        name = f"self_attn_norm"
        # sum sub pow sum div mul add
        write_to_node(
            name,
            OPs=batchsize * hidden_size * 1 * 7,
            load_weight=0,
            load_act=batchsize * hidden_size * 1 * a_byte,
            store_act=batchsize * hidden_size * 1 * a_byte,
            load_kv_cache=0,
            store_kv_cache=0,
            input_names=["input"]
        )

        name = f"self_attn_add"
        write_to_node(
            name,
            OPs=batchsize * hidden_size * 1,
            load_weight=0,
            load_act=batchsize * hidden_size * 1 * a_byte,
            store_act=batchsize * hidden_size * 1 * a_byte,
            load_kv_cache=0,
            store_kv_cache=0,
            input_names=["input","out_proj"]
        )

        name = f"mlp_norm"
        # sum sub pow sum div mul add
        write_to_node(
            name,
            OPs=batchsize * hidden_size * 1 * 7,
            load_weight=0,
            load_act=batchsize * hidden_size * 1 * a_byte,
            store_act=batchsize * hidden_size * 1 * a_byte,
            load_kv_cache=0,
            store_kv_cache=0,
            input_names=["self_attn_add"]
        )

        name = f"mlp_add"
        write_to_node(
            name,
            OPs=batchsize * hidden_size * 1,
            load_weight=0,
            load_act=batchsize * hidden_size * 1 * a_byte,
            store_act=batchsize * hidden_size * 1 * a_byte,
            load_kv_cache=0,
            store_kv_cache=0,
            input_names=["self_attn_add","down_proj"]
        )
        name = f"mlp_act"
        write_to_node(
            name,
            OPs=batchsize * hidden_size * 1,
            load_weight=0,
            load_act=batchsize * hidden_size * 1 * a_byte,
            store_act=batchsize * hidden_size * 1 * a_byte,
            load_kv_cache=0,
            store_kv_cache=0,
            input_names=["up_proj","gate_proj"]
        )

    # for prefill
    else:
        name = f"qk_matmul"
        write_to_node(
            name,
            OPs=seqlen * seqlen * head_size * num_attention_heads * batchsize * 2,
            load_weight=0,
            load_act=0,
            store_act=seqlen * seqlen * batchsize * num_attention_heads * a_byte,
            load_kv_cache=seqlen * head_size * batchsize * num_attention_heads * 2 * a_byte,
            store_kv_cache=0,
        )
        name = f"sv_matmul"
        write_to_node(
            name,
            OPs=seqlen * head_size * seqlen * num_attention_heads * batchsize * 2,
            load_weight=0,
            load_act=seqlen * seqlen * batchsize * num_attention_heads * a_byte,
            store_act=seqlen * head_size * batchsize * num_attention_heads * a_byte,
            load_kv_cache=seqlen * head_size * batchsize * num_attention_heads * a_byte,
            store_kv_cache=0,
        )
        name = f"softmax"
        write_to_node(
            name,
            OPs=batchsize * num_attention_heads * seqlen * seqlen * 5,
            load_weight=0,
            load_act=batchsize * num_attention_heads * seqlen * seqlen * a_byte,
            store_act=batchsize * num_attention_heads * seqlen * seqlen * a_byte,
            load_kv_cache=0,
            store_kv_cache=0,
        )
        name = f"norm"
        write_to_node(
            name,
            OPs=batchsize * hidden_size * seqlen * 7,
            load_weight=0,
            load_act=batchsize * hidden_size * seqlen * a_byte,
            store_act=batchsize * hidden_size * seqlen * a_byte,
            load_kv_cache=0,
            store_kv_cache=0,
        )
        name = f"add"
        write_to_node(
            name,
            OPs=batchsize * hidden_size * seqlen * 1,
            load_weight=0,
            load_act=batchsize * hidden_size * seqlen * a_byte,
            store_act=batchsize * hidden_size * seqlen * a_byte,
            load_kv_cache=0,
            store_kv_cache=0,
        )
    return nodes,edges