from graph.module import Module
from graph.base_nodes import *
from graph.network import Network

def get_network_graph(network_params):
    p=network_params

    modules=[]
    embedding=Module(name="embedding", nodes=[
    Embedding("embedding", ["input_inds"], {"out_features":p.hidden_size, "p.vocab_size":p.vocab_size})])
    modules.append(embedding)

    for i in range(p.num_hidden_layers):
        if i==0:
            input_name="embedding.embedding"
        else:
            input_name=f"transformer_layer{i-1}.mlp_add"
        transformer_layer=Module(name=f"transformer_layer{i}",nodes=[
            Linear("q_proj", [input_name], {"out_features":p.hidden_size}),
            Linear("k_proj", [input_name], {"out_features":p.hidden_size//p.num_attention_heads*p.num_key_value_heads}),
            Linear("v_proj", [input_name], {"out_features":p.hidden_size//p.num_attention_heads*p.num_key_value_heads}),
            ReshapeTranspose("q_reshape", ["q_proj"], {"shape":["input_shape[0]",p.num_attention_heads,"input_shape[2]", p.hidden_size//p.num_attention_heads]}),
            ReshapeTranspose("k_reshape", ["k_proj"], {"shape":["input_shape[0]",p.num_key_value_heads,p.hidden_size//p.num_attention_heads,"input_shape[2]"]}),
            ReshapeTranspose("v_reshape", ["v_proj"], {"shape":["input_shape[0]",p.num_key_value_heads,"input_shape[2]", p.hidden_size//p.num_attention_heads]}),
            MatMul("qk_matmul", ["q_reshape", "k_reshape"]),
            Softmax("softmax", ["qk_matmul"]),
            MatMul("sv_matmul", ["softmax", "v_reshape"]),
            ReshapeTranspose("sv_reshape", ["sv_matmul"], {"shape":["input_shape[0]","input_shape[2]", p.hidden_size]}),
            Linear("out_proj", ["sv_reshape"], {"out_features":p.hidden_size}),
            Add("attn_add", [input_name, "out_proj"]),
            Norm("mlp_norm", ["attn_add"]),
            Linear("gate_proj", ["mlp_norm"], {"out_features":p.intermediate_size}),
            Linear("up_proj", ["mlp_norm"], {"out_features":p.intermediate_size}),
            Activation("mlp_act", ["up_proj", "gate_proj"]),
            Linear("down_proj", ["mlp_act"], {"out_features":p.hidden_size}),
            Add("mlp_add", ["attn_add", "down_proj"])
        ])
        modules.append(transformer_layer)

    lm_head=Module(name="lm_head", nodes=[
        Norm("lm_head_norm", [f"transformer_layer{p.num_hidden_layers-1}.mlp_add"]),
        Linear("lm_head", ["lm_head_norm"], {"out_features":p.vocab_size})
    ])
    modules.append(lm_head)

    return Network(modules=modules)
