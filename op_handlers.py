"""Per-op cost handlers for the roofline analyzer.

Each handler computes the compute/memory cost of a single op type and returns
the keyword arguments consumed by `ModelAnalyzer._analyze_to_results`:

    OPs, load_weight, load_act, store_act, load_kv_cache, store_kv_cache

Handlers are looked up by `op_type` in the `OP_HANDLERS` registry, so adding a
new op type (e.g. linear attention, a diffusion action head, a ViT patch-embed)
means writing a handler and registering it -- no edits to the analyzer's
orchestration. See design_docs/vla_roofline_analyzer.md.

Stage handling
--------------
The decode and prefill stages differ only in the *query sequence length*
processed in one step, so handlers take it explicitly instead of branching on a
"stage" string:

    q_seqlen  : number of query tokens processed this step
                (decode = 1, prefill = full prompt length; >1 decode is also
                 valid, e.g. speculative decoding / parallel action chunks)
    kv_seqlen : number of key/value tokens attended over (context length)

This module is a faithful extraction of the cost math that previously lived
inline in `ModelAnalyzer.analyze()`; results stay bit-identical and are guarded
by the golden harness (tests/golden_analyzer.py).
"""

import math


class OpContext:
    """Stage-invariant context shared by every op handler in one analysis.

    Carries model dimensions, byte widths, and the hardware on-chip buffer size.
    The stage-varying quantities (q_seqlen, kv_seqlen) and op-specific params
    (e.g. a linear layer's in/out channels) are passed to the handler directly.
    """

    def __init__(
        self,
        batchsize,
        a_byte,
        w_byte,
        kv_byte,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        head_size,
        onchip_buffer,
    ):
        self.batchsize = batchsize
        self.a_byte = a_byte
        self.w_byte = w_byte
        self.kv_byte = kv_byte
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_size = head_size
        self.onchip_buffer = onchip_buffer


def _result(OPs=0, load_weight=0, load_act=0, store_act=0, load_kv_cache=0, store_kv_cache=0):
    return {
        "OPs": OPs,
        "load_weight": load_weight,
        "load_act": load_act,
        "store_act": store_act,
        "load_kv_cache": load_kv_cache,
        "store_kv_cache": store_kv_cache,
    }


def linear(ctx, q_seqlen, kv_seqlen, ic, oc, is_kv_proj):
    """A projection / matmul against a weight matrix of shape (ic, oc)."""
    is_normal_proj = not is_kv_proj
    bs = ctx.batchsize
    return _result(
        OPs=ic * oc * bs * q_seqlen * 2,
        load_weight=ic * oc * ctx.w_byte,
        load_act=ic * bs * q_seqlen * ctx.a_byte,
        store_act=0 if is_kv_proj else oc * bs * q_seqlen * ctx.a_byte,
        load_kv_cache=0,
        store_kv_cache=(0 if is_normal_proj else oc * bs * q_seqlen * ctx.kv_byte),
    )


def qk_matmul(ctx, q_seqlen, kv_seqlen, query_act_heads=None):
    """Q @ K^T scores, shape [batch, heads, q_seqlen, kv_seqlen]."""
    # `query_act_heads`: head count used when sizing the Q-activation load.
    # The Q activation physically has `num_attention_heads` heads, which is the
    # default here. The legacy code, however, used `num_key_value_heads` in the
    # prefill stage (and attention-heads in decode) -- a discrepancy that only
    # surfaces under GQA. The analyzer passes the legacy value explicitly to
    # preserve bit-identical behavior; drop that override to make it consistent.
    if query_act_heads is None:
        query_act_heads = ctx.num_attention_heads
    head_size = ctx.head_size
    nah = ctx.num_attention_heads
    nkvh = ctx.num_key_value_heads
    bs = ctx.batchsize
    return _result(
        OPs=q_seqlen * kv_seqlen * head_size * nah * bs * 2,
        load_act=q_seqlen * head_size * bs * query_act_heads * ctx.a_byte,
        store_act=q_seqlen * kv_seqlen * bs * nah * ctx.a_byte,
        load_kv_cache=kv_seqlen * head_size * bs * nkvh * ctx.kv_byte,
    )


def sv_matmul(ctx, q_seqlen, kv_seqlen):
    """softmax(scores) @ V, producing [batch, heads, q_seqlen, head_size]."""
    head_size = ctx.head_size
    nah = ctx.num_attention_heads
    nkvh = ctx.num_key_value_heads
    bs = ctx.batchsize
    return _result(
        OPs=q_seqlen * head_size * kv_seqlen * nah * bs * 2,
        load_act=q_seqlen * kv_seqlen * bs * nah * ctx.a_byte,
        store_act=q_seqlen * head_size * bs * nah * ctx.a_byte,
        load_kv_cache=kv_seqlen * head_size * bs * nkvh * ctx.kv_byte,
    )


def softmax(ctx, q_seqlen, kv_seqlen):
    # max sub exp sum div -> 5 ops per element of the [q_seqlen, kv_seqlen] scores
    nah = ctx.num_attention_heads
    bs = ctx.batchsize
    n = bs * nah * kv_seqlen * q_seqlen
    return _result(OPs=n * 5, load_act=n * ctx.a_byte, store_act=n * ctx.a_byte)


def fused_attention(ctx, q_seqlen, kv_seqlen):
    # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
    head_size = ctx.head_size
    nah = ctx.num_attention_heads
    nkvh = ctx.num_key_value_heads
    bs = ctx.batchsize
    a_byte = ctx.a_byte
    block_size_r = min(math.ceil(ctx.onchip_buffer / (ctx.kv_byte * head_size)), head_size)
    n_blocks_r = math.ceil(q_seqlen / block_size_r)
    qk_matmul_OPs = q_seqlen * kv_seqlen * head_size * nah * bs * 2
    sv_matmul_OPs = q_seqlen * head_size * kv_seqlen * nah * bs * 2
    softmax_OPs = bs * nah * kv_seqlen * q_seqlen * 5
    q_numel = q_seqlen * head_size * bs * nah * a_byte
    o_numel = q_seqlen * head_size * bs * nah * a_byte
    return _result(
        OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
        load_weight=0,
        load_act=q_numel,
        store_act=o_numel * 2,  # initialize O and save O
        load_kv_cache=n_blocks_r * kv_seqlen * head_size * bs * nkvh * ctx.kv_byte * 2,
        store_kv_cache=0,
    )


def patch_embed(ctx, q_seqlen, kv_seqlen, num_patches, patch_dim):
    """Vision patch embedding: split the image into `num_patches` patches and
    project each (flattened length `patch_dim` = channels * patch_size^2) to
    ctx.hidden_size. Equivalent to a conv with kernel=stride=patch_size,
    expressed as a matmul; runs once per image (not per layer)."""
    d = ctx.hidden_size
    bs = ctx.batchsize
    return _result(
        OPs=num_patches * patch_dim * d * 2 * bs,
        load_weight=patch_dim * d * ctx.w_byte,
        load_act=bs * num_patches * patch_dim * ctx.a_byte,  # the input image
        store_act=bs * num_patches * d * ctx.a_byte,
    )


def linear_attention(ctx, q_seqlen, kv_seqlen):
    """O(L) linear attention (e.g. SARA-RT / linear transformers).

    Instead of forming the L*L softmax score matrix, accumulate a per-head
    d_head*d_head KV summary over the kv tokens, then apply it to each query:
    cost is linear in sequence length, with no L^2 scores or softmax.
    """
    hs = ctx.head_size
    nah = ctx.num_attention_heads
    nkvh = ctx.num_key_value_heads
    bs = ctx.batchsize
    build_kv_OPs = kv_seqlen * hs * hs * nah * bs * 2  # KV summary from K,V
    apply_q_OPs = q_seqlen * hs * hs * nah * bs * 2     # Q @ KV-summary
    return _result(
        OPs=build_kv_OPs + apply_q_OPs,
        load_act=q_seqlen * hs * bs * nah * ctx.a_byte,    # Q
        store_act=q_seqlen * hs * bs * nah * ctx.a_byte,   # O
        load_kv_cache=kv_seqlen * hs * bs * nkvh * ctx.kv_byte * 2,  # K and V
    )


def norm(ctx, q_seqlen, kv_seqlen):
    # sum sub pow sum div mul add -> 7 ops per element
    n = ctx.batchsize * ctx.hidden_size * q_seqlen
    return _result(OPs=n * 7, load_act=n * ctx.a_byte, store_act=n * ctx.a_byte)


def add(ctx, q_seqlen, kv_seqlen):
    # residual add
    n = ctx.batchsize * ctx.hidden_size * q_seqlen
    return _result(OPs=n * 1, load_act=n * ctx.a_byte, store_act=n * ctx.a_byte)


def act(ctx, q_seqlen, kv_seqlen):
    # mlp activation / gating (reads two inputs, writes one)
    n = ctx.batchsize * ctx.hidden_size * q_seqlen
    return _result(OPs=n * 2, load_act=n * ctx.a_byte * 2, store_act=n * ctx.a_byte)


# Registry: op_type -> handler. Extend this to add new op types.
OP_HANDLERS = {
    "linear": linear,
    "patch_embed": patch_embed,
    "qk_matmul": qk_matmul,
    "sv_matmul": sv_matmul,
    "softmax": softmax,
    "fused_attention": fused_attention,
    "linear_attention": linear_attention,
    "norm": norm,
    "add": add,
    "act": act,
}
