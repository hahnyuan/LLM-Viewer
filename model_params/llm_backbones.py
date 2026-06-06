"""LLM-backbone presets used by VLAs, as local params (no HF download / gating).

Used via ModelAnalyzer(model_id, hw, "configs/Llama.py", source="llm_backbones").
Gemma's per-op cost structure matches the Llama config (gated MLP, RMSNorm x2,
MQA via num_key_value_heads), and for these models head_dim == hidden/heads so
the q/k/v projection dims the Llama config derives are exact.
"""

from easydict import EasyDict


model_params = {
    # Gemma 2B -- the language backbone inside PaliGemma-3B (pi0's VLM).
    # hidden 2048, 18 layers, 8 query heads, 1 KV head (MQA), head_dim 256,
    # intermediate 16384, vocab 256000.
    "gemma_2b": EasyDict(
        hidden_size=2048,
        num_attention_heads=8,
        num_key_value_heads=1,
        num_hidden_layers=18,
        intermediate_size=16384,
        vocab_size=256000,
    ),
}
