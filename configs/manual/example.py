from graph.module import Module
from graph.base_nodes import *
from graph.network import Network

def get_network_graph(network_params):
    p=network_params

    modules=[]
    embedding=Module(name="embedding", nodes=[
    Embedding("embedding", ["input_inds"], {"out_features":p.hidden_size, "p.vocab_size":p.vocab_size})])
    modules.append(embedding)

    lm_head=Module(name="lm_head", nodes=[
        Norm("lm_head_norm", [f"embedding.embedding"]),
        Linear("lm_head", ["lm_head_norm"], {"out_features":p.vocab_size})
    ])
    modules.append(lm_head)

    return Network(modules=modules)
