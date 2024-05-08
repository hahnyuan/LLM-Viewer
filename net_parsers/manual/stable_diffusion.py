from ..base_parser import BaseParser
from net_graph.module import Module
from net_graph.base_nodes import *
from net_graph.network import Network
from transformers import AutoConfig
from types import SimpleNamespace


class StableDiffusionParser(BaseParser):
    "create based on stable diffusion 1.4, part of diffusion UNetModel"
    def __init__(self, model_id, args: dict):
        super().__init__(model_id, args)

        self.input_blocks_num = 12
        self.mid_blocks_num = 3
        self.out_blocks_num = 12

        self.time_embed_hidden_channels = 1280
        self.time_embed_out_channels = 1280
        self.hidden_channels = 320
        self.hidden_channels_ffn = 1280
        self.num_heads = 8  # attention

        self.batch_size = 2*self.args.get('batchsize', 1) if self.args.get("conditional_guidance", True) else self.args.get('batchsize', 1)
        self.latent_size = self.args.get('latent_size', 64)
    
    def build_res_block(self, input_name, output_name, hidden_channels, with_skip_connection=False):
        out_name_before_skip = "out_layers_out" if with_skip_connection else output_name
        nodes = [
            # --ResBlock--
            # in_layers
            GroupNorm("in_layers_0", [input_name], {"groups": 32, "channels": hidden_channels}),
            SiLU("in_layers_1", ["in_layers_0"]),
            Conv2d("in_layers_out", ["in_layers_1"], {"out_channels": hidden_channels, "ksize": (3,3), "stride": (1,1), "dilations": (1,1), "padding":(1,1), "groups":1}),
            # emb_layers
            SiLU("emb_layers_0", ["time_embed.time_embed_out"]),
            Linear("emb_layers_out", ["emb_layers_0"], {"out_features": hidden_channels}),
            ReshapeTranspose("emb_layers_out_reshape", ["emb_layers_out"], {"shape":["input_shape[0]", "input_shape[1]", 1, 1]}),
            # add
            Add("x_emb_add", ["in_layers_out", "emb_layers_out_reshape"]),
            # out_layers
            GroupNorm("out_layers_0", ["x_emb_add"], {"groups": 32, "channels": hidden_channels}),
            SiLU("out_layers_1", ["out_layers_0"]),
            Conv2d(out_name_before_skip, ["out_layers_1"], {"out_channels": hidden_channels, "ksize": (3,3), "stride": (1,1), "dilations": (1,1), "padding":(1,1), "groups":1}),
        ]
        if with_skip_connection:
            nodes += [
                Conv2d("skip_connection", [out_name_before_skip], {"out_channels": hidden_channels, "ksize": (1,1), "stride": (1,1), "dilations": (1,1), "padding":(0,0), "groups":1}),
                Add(output_name, ["skip_connection", out_name_before_skip]),
            ]
        return nodes

    def build_spatial_transformer(self, input_name, output_name, hidden_channels, input_context_emb, hidden_channels_ffn, size_ratio):
        nodes = [
            # --SpatialTransformer--
            GroupNorm("spatial_transformer_0", [input_name], {"groups": 32, "channels": hidden_channels}),
            Conv2d("spatial_transformer_1", ["spatial_transformer_0"], {"out_channels": hidden_channels, "ksize": (1,1), "stride": (1,1), "dilations": (1,1), "padding":(0,0), "groups":1}),
            ReshapeTranspose("spatial_transformer_1_reshape", ["spatial_transformer_1"], {"shape":["input_shape[0]","input_shape[2]*input_shape[3]", hidden_channels]}),
            # norm1
            LayerNorm("spatial_transformer_norm1", ["spatial_transformer_1_reshape"]),
            # CrossAttention_0
            Linear("self_attn_to_q", ["spatial_transformer_norm1"], {"out_features": hidden_channels}),
            Linear("self_attn_to_k", ["spatial_transformer_norm1"], {"out_features": hidden_channels}),
            Linear("self_attn_to_v", ["spatial_transformer_norm1"], {"out_features": hidden_channels}),
            ReshapeTranspose("self_attn_to_q_reshape", ["self_attn_to_q"], {"shape":[self.batch_size*self.num_heads, "input_shape[1]", "input_shape[2]//self.num_heads"], "num_heads":self.num_heads}),
            ReshapeTranspose("self_attn_to_k_reshape", ["self_attn_to_k"], {"shape":[self.batch_size*self.num_heads, "input_shape[2]//self.num_heads", "input_shape[1]"], "num_heads":self.num_heads}),
            ReshapeTranspose("self_attn_to_v_reshape", ["self_attn_to_v"], {"shape":[self.batch_size*self.num_heads, "input_shape[1]", "input_shape[2]//self.num_heads"], "num_heads":self.num_heads}),

            MatMul("self_attn_qk_matmul", ["self_attn_to_q_reshape", "self_attn_to_k_reshape"], {"out_features": hidden_channels}),
            Softmax("self_attn_softmax", ["self_attn_qk_matmul"]),
            MatMul("self_attn_smv_matmul", ["self_attn_softmax", "self_attn_to_v_reshape"]),
            ReshapeTranspose("self_attn_smv_matmul_reshape", ["self_attn_smv_matmul"], {"shape":[self.batch_size, "input_shape[1]", "input_shape[2]*self.num_heads"], "num_heads":self.num_heads}),
            Linear("self_attn_out", ["self_attn_smv_matmul_reshape"], {"out_features": hidden_channels}),
            # add
            Add("self_attn_out_add", ["self_attn_out", "spatial_transformer_1_reshape"]),
            # norm2
            LayerNorm("spatial_transformer_norm2", ["self_attn_out_add"]),
            # CrossAttention_1
            Linear("cross_attn_to_q", ["spatial_transformer_norm2"], {"out_features": hidden_channels}),
            Linear("cross_attn_to_k", [input_context_emb], {"out_features": hidden_channels}),
            Linear("cross_attn_to_v", [input_context_emb], {"out_features": hidden_channels}),
            ReshapeTranspose("cross_attn_to_q_reshape", ["cross_attn_to_q"], {"shape":[self.batch_size*self.num_heads, "input_shape[1]", "input_shape[2]//self.num_heads"], "num_heads":self.num_heads}),
            ReshapeTranspose("cross_attn_to_k_reshape", ["cross_attn_to_k"], {"shape":[self.batch_size*self.num_heads, "input_shape[2]//self.num_heads", "input_shape[1]"], "num_heads":self.num_heads}),
            ReshapeTranspose("cross_attn_to_v_reshape", ["cross_attn_to_v"], {"shape":[self.batch_size*self.num_heads, "input_shape[1]", "input_shape[2]//self.num_heads"], "num_heads":self.num_heads}),

            MatMul("cross_attn_qk_matmul", ["cross_attn_to_q_reshape", "cross_attn_to_k_reshape"], {"out_features": hidden_channels}),
            Softmax("cross_attn_softmax", ["cross_attn_qk_matmul"]),
            MatMul("cross_attn_smv_matmul", ["cross_attn_softmax", "cross_attn_to_v_reshape"]),
            ReshapeTranspose("cross_attn_smv_matmul_reshape", ["cross_attn_smv_matmul"], {"shape":[self.batch_size, "input_shape[1]", "input_shape[2]*self.num_heads"], "num_heads":self.num_heads}),
            Linear("cross_attn_out", ["cross_attn_smv_matmul_reshape"], {"out_features": hidden_channels}),
            # add
            Add("cross_attn_out_add", ["cross_attn_out", "self_attn_out_add"]),
            # norm3
            LayerNorm("spatial_transformer_norm3", ["cross_attn_out_add"]),
            # ffn
            GEGLU("ffn_geglu", ["spatial_transformer_norm3"], {"chunk_num":2, "chunk_dim":-1, "out_features": hidden_channels_ffn*2}),
            Linear("ffn_out", ["ffn_geglu"], {"out_features": hidden_channels}),
            # add
            Add("ffn_out_add", ["ffn_out", "cross_attn_out_add"]),
            # proj_out
            ReshapeTranspose("ffn_out_add_reshape", ["ffn_out_add"], {"shape":["input_shape[0]", hidden_channels, self.latent_size*size_ratio, self.latent_size*size_ratio]}),
            Conv2d("proj_out", ["ffn_out_add_reshape"], {"out_channels": hidden_channels, "ksize": (1,1), "stride": (1,1), "dilations": (1,1), "padding":(0,0), "groups":1}),
            # add
            Add(output_name, ["proj_out", input_name]),
        ]
        return nodes

    def build_input_blocks(self, modules, input_latent, input_context_emb):
        for i in range(self.input_blocks_num):
            if i == 0:
                nodes = [
                    Conv2d(f"input_block_{i}_out", [input_latent], {"out_channels": self.hidden_channels, "ksize": (3,3), "stride": (1,1), "dilations": (1,1), "padding":(1,1), "groups":1})
                ]
            else:
                input_name = f"input_block_{i-1}.input_block_{i-1}_out"  # module_name+layer_name
                output_name = f"input_block_{i}_out"
                size_ratio = 1
                hidden_channels = self.hidden_channels
                hidden_channels_ffn = self.hidden_channels_ffn
                if i in [4,5,6]:
                    hidden_channels = 2*self.hidden_channels
                elif i in [7,8,9,10,11]:
                    hidden_channels = 4*self.hidden_channels
                
                if i in [1,2,4,5,7,8,10,11]:
                    if i in [4,5]:
                        size_ratio = 1/2
                        hidden_channels_ffn = 2*self.hidden_channels_ffn
                    elif i in [7,8]:
                        size_ratio = 1/4
                        hidden_channels_ffn = 4*self.hidden_channels_ffn
                    
                    res_block_output_name = "skip_connection_out"
                    with_skip_connection = True if i in [4,7] else False
                    if i in [10,11]:
                        res_block_output_name = output_name
                        nodes = self.build_res_block(input_name, res_block_output_name, hidden_channels, with_skip_connection=with_skip_connection)
                    else:
                        nodes = self.build_res_block(input_name, res_block_output_name, hidden_channels, with_skip_connection=with_skip_connection)
                        nodes += self.build_spatial_transformer(res_block_output_name, output_name, hidden_channels, input_context_emb, hidden_channels_ffn, size_ratio)
                elif i in [3,6,9]:
                    nodes = [
                        Conv2d(output_name, [input_name], {"out_channels": hidden_channels, "ksize": (3,3), "stride": (2,2), "dilations": (1,1), "padding":(1,1), "groups":1})
                    ]

            input_block_ = Module(name=f"input_block_{i}", nodes=nodes)
            modules.append(input_block_)     
        return modules

    def build_mid_blocks(self, modules, input_context_emb):
        for i in range(self.mid_blocks_num):
            output_name = f"mid_block_{i}_out"
            size_ratio = 1/8
            hidden_channels = 4*self.hidden_channels
            hidden_channels_ffn = 4*self.hidden_channels_ffn
            if i in [0, 2]:
                input_name = "input_block_11.input_block_11_out" if i==0 else f"mid_block_{i-1}.mid_block_{i-1}_out"
                nodes = self.build_res_block(input_name, output_name, hidden_channels, with_skip_connection=False)
            elif i in [1]:
                input_name = f"mid_block_{i-1}.mid_block_{i-1}_out"
                nodes = self.build_spatial_transformer(input_name, output_name, hidden_channels, input_context_emb, hidden_channels_ffn, size_ratio)
            
            mid_block_ = Module(name=f"mid_block_{i}", nodes=nodes)
            modules.append(mid_block_)     
        return modules

    def build_output_blocks(self, modules, input_context_emb):
        for i in range(self.out_blocks_num):
            output_name = f"output_block_{i}_out"
            if i in [0,1,2]:
                size_ratio = 1/8
            elif i in [3,4,5]:
                size_ratio = 1/4
            elif i in [6,7,8]:
                size_ratio = 1/2
            else:
                size_ratio = 1

            if i in [6, 7, 8]:
                hidden_channels = 2*self.hidden_channels
                hidden_channels_ffn = 2*self.hidden_channels_ffn
            elif i in [9, 10, 11]:
                hidden_channels = self.hidden_channels
                hidden_channels_ffn = self.hidden_channels_ffn
            else:
                hidden_channels = 4*self.hidden_channels
                hidden_channels_ffn = 4*self.hidden_channels_ffn
            
            input_name = "mid_block_2.mid_block_2_out" if i==0 else f"output_block_{i-1}.output_block_{i-1}_out"
            nodes =[
                Concat("concat", [input_name, f"input_block_{self.out_blocks_num-1-i}.input_block_{self.out_blocks_num-1-i}_out"], {"dim": 1}),
            ]
            res_block_output_name = "skip_connection_out"
            if i in [0, 1]:
                res_block_output_name = output_name
            nodes += self.build_res_block("concat", res_block_output_name, hidden_channels, with_skip_connection=True)
            
            if i in [3, 4, 6, 7, 9, 10, 11]:
                nodes += self.build_spatial_transformer(res_block_output_name, output_name, hidden_channels, input_context_emb, hidden_channels_ffn, size_ratio)
            elif i in [2, 5, 8]:
                nodes += self.build_spatial_transformer(res_block_output_name, "spatial_transformer_out", hidden_channels, input_context_emb, hidden_channels_ffn, size_ratio)
                nodes += [
                    Upsample("upsample", ["spatial_transformer_out"], {"ratio": (2,2), "mode": "nearest"}),
                    Conv2d(output_name, ["upsample"], {"out_channels": hidden_channels, "ksize": (3,3), "stride": (1,1), "dilations": (1,1), "padding":(1,1), "groups":1}),
                ]
            output_block_ = Module(name=f"output_block_{i}", nodes=nodes)
            modules.append(output_block_)     
        return modules

    def parse(self):
        """
        input_latent: (batch_size, 4, latent_size, latent_size)
        input_time_emb: (batch_size, 320), repeat from shape: (batch_size)
        input_context_emb: (batch_size, 77, 768)

        NOTE create model include each step, such as add, concat... so profile_info maybe more than other profile tool
        """
        input_latent = "input_latent"
        input_time_emb = "input_time_emb"
        input_context_emb = "input_context_emb"
        
        modules = []
        embedding = Module(name="time_embed", nodes=[
            Linear("time_embed_0", [input_time_emb], {"out_features": self.time_embed_hidden_channels}),
            Norm("time_embed_norm", ["time_embed_0"]),
            Linear("time_embed_out", ["time_embed_norm"], {"out_features": self.time_embed_out_channels}),
        ])
        modules.append(embedding)
        
        modules = self.build_input_blocks(modules, input_latent, input_context_emb)
        modules = self.build_mid_blocks(modules, input_context_emb)
        modules = self.build_output_blocks(modules, input_context_emb)

        out_module = Module(name="out", nodes=[
            GroupNorm("norm", [f"output_block_{self.out_blocks_num-1}.output_block_{self.out_blocks_num-1}_out"], {"groups": 32, "channels": self.hidden_channels}),
            SiLU("silu", ["norm"]),
            Conv2d("model_out", ["silu"], {"out_channels": self.hidden_channels, "ksize": (3,3), "stride": (1,1), "dilations": (1,1), "padding":(1,1), "groups":1}),
        ])
        modules.append(out_module)
        return Network(modules=modules)