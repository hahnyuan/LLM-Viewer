"""Config module for a standard Vision Transformer (ViT) encoder.

A ViT encoder is a plain transformer run as a single bidirectional forward pass
over the image patch tokens (no causal mask, no KV cache, no lm_head). It is
analyzed by reusing ModelAnalyzer.analyze() with seqlen = number of patch
tokens and reading the "prefill" results. The patch-embedding front-end and the
projector to the LLM are handled by the VLA orchestrator (vla.py), not here.

Differences from the decoder LLM config (e.g. configs/Llama.py):
  * MLP is a non-gated fc1 -> act -> fc2 (no gate_proj), so the linear set is
    {q_proj, k_proj, v_proj, out_proj, fc1, fc2}.
  * post_process returns [] (no lm_head).

Expects model_params (see model_params/vision_encoders.py) with attributes:
  hidden_size, num_attention_heads, num_hidden_layers, intermediate_size,
  image_size, patch_size, num_channels, has_cls_token.
"""


def get_num_attention_heads(model_params):
    return getattr(model_params, "num_attention_heads")


def get_hidden_size(model_params):
    return getattr(model_params, "hidden_size")


def get_num_key_value_heads(model_params):
    # ViT uses standard multi-head attention (no GQA).
    return getattr(model_params, "num_attention_heads")


def get_norm_layers(model_params):
    return ["attn_norm", "mlp_norm"]


def get_num_hidden_layers(model_params):
    return getattr(model_params, "num_hidden_layers")


def get_intermediate_size(model_params):
    return getattr(model_params, "intermediate_size")


def get_num_image_tokens(model_params):
    """Number of patch tokens fed to the transformer (+1 for a CLS token)."""
    grid = getattr(model_params, "image_size") // getattr(model_params, "patch_size")
    tokens = grid * grid
    if getattr(model_params, "has_cls_token", False):
        tokens += 1
    return tokens


def get_patch_dim(model_params):
    """Flattened length of one patch = channels * patch_size^2."""
    return getattr(model_params, "num_channels") * getattr(model_params, "patch_size") ** 2


def get_linear_layers(model_params, tp_size: int):
    hidden_size = get_hidden_size(model_params)
    intermediate_size = get_intermediate_size(model_params)

    if tp_size > 1:
        assert hidden_size % tp_size == 0
        assert intermediate_size % tp_size == 0

    return {
        "q_proj": [hidden_size, hidden_size // tp_size],
        "k_proj": [hidden_size, hidden_size // tp_size],
        "v_proj": [hidden_size, hidden_size // tp_size],
        "out_proj": [hidden_size // tp_size, hidden_size],
        "fc1": [hidden_size, intermediate_size // tp_size],
        "fc2": [intermediate_size // tp_size, hidden_size],
    }


def post_process(model_params, args):
    # No language-model head on a vision encoder.
    return []


# name, input_names -- used by the frontend graph (get_model_graph)
transformer_layer_graph = {
    "input": [],
    "attn_norm": ["input"],
    "q_proj": ["attn_norm"],
    "k_proj": ["attn_norm"],
    "v_proj": ["attn_norm"],
    "qk_matmul": ["q_proj", "k_proj"],
    "softmax": ["qk_matmul"],
    "sv_matmul": ["softmax", "v_proj"],
    "out_proj": ["sv_matmul"],
    "attn_add": ["input", "out_proj"],
    "mlp_norm": ["attn_add"],
    "fc1": ["mlp_norm"],
    "mlp_act": ["fc1"],
    "fc2": ["mlp_act"],
    "mlp_add": ["attn_add", "fc2"],
    "output": ["mlp_add"],
}

flashattention_transformer_layer_graph = {
    "input": [],
    "attn_norm": ["input"],
    "q_proj": ["attn_norm"],
    "k_proj": ["attn_norm"],
    "v_proj": ["attn_norm"],
    "fused_attention": ["q_proj", "k_proj", "v_proj"],
    "out_proj": ["fused_attention"],
    "attn_add": ["input", "out_proj"],
    "mlp_norm": ["attn_add"],
    "fc1": ["mlp_norm"],
    "mlp_act": ["fc1"],
    "fc2": ["mlp_act"],
    "mlp_add": ["attn_add", "fc2"],
    "output": ["mlp_add"],
}
