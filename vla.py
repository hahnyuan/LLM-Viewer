"""VLA (Vision-Language-Action) multi-phase roofline orchestration.

A single VLA inference step on an edge device composes three phases:

    1. vision encode : patch_embed (once) + ViT transformer over image tokens
    2. projector     : map vision features into the LLM token space
    3. LLM prefill   : backbone forward over [image tokens + text tokens]

(The action-generation phase -- autoregressive token decode vs diffusion /
flow-matching chunk -- is added in a later step; see
design_docs/vla_roofline_analyzer.md.)

Each phase reuses the golden-locked op cost machinery: the ViT and LLM are
analyzed by ModelAnalyzer (prefill stage), and the patch-embed / projector ops
go through op_handlers + roofline_model.evaluate_op directly. The orchestrator
also produces a memory-fit verdict against the device's memory_capacity --
the headline question for edge deployment.
"""

import importlib
from dataclasses import dataclass

import configs.vit as vit_cfg
from model_analyzer import ModelAnalyzer
from op_handlers import OpContext, OP_HANDLERS
from roofline_model import evaluate_op
from hardwares.hardware_params import hardware_params


# Default action-expert dims (~300M, pi0/Gemma-expert scale) when a flow/parallel
# VLA preset doesn't specify its own.
DEFAULT_ACTION_EXPERT = {"hidden": 1024, "heads": 8, "kv_heads": 8, "layers": 18, "intermediate": 4096}


@dataclass
class VLAConfig:
    """A VLA = one vision encoder + one LLM backbone (+ projector + action head)."""

    name: str
    vision_model_id: str           # key in model_params/vision_encoders.py
    llm_model_id: str              # HF id or model_params key
    llm_source: str = "huggingface"
    llm_config_file: str = None    # None -> ModelAnalyzer auto-search

    # Action head: "ar" (autoregressive discrete tokens, e.g. OpenVLA),
    # "flow" (diffusion / flow-matching chunk, e.g. pi0/Octo), or "parallel"
    # (single-pass action expert).
    action_head: str = "ar"
    # autoregressive head:
    action_horizon: int = 1        # actions per chunk
    tokens_per_action: int = 7     # discrete tokens per action (e.g. 7-DoF)
    # flow / diffusion / parallel head:
    num_flow_steps: int = 10       # denoising/flow steps ("parallel" forces 1)
    action_chunk: int = 50         # action tokens processed by the expert
    action_expert: dict = None     # expert dims; None -> DEFAULT_ACTION_EXPERT
    expert_attention: str = "quadratic"  # "quadratic" or "linear" (SARA-RT)


def _phase(time, OPs, weight=0.0, **extra):
    d = {"time": time, "OPs": OPs, "weight": weight}
    d.update(extra)
    return d


def chunk_horizon(cfg):
    """Actions PREDICTED per inference -- the chunk length that drives compute.

    One forward pass emits this many actions (action_horizon for AR heads, the
    action-chunk length for flow/parallel). How many are actually executed
    before replanning is a separate deployment choice (see control_budget's
    exec_horizon).
    """
    return cfg.action_horizon if cfg.action_head == "ar" else cfg.action_chunk


def control_budget(cfg, total_time, control_hz, exec_horizon=None):
    """Deployment verdict under action chunking + receding-horizon replanning.

    The policy predicts `chunk_horizon` actions per inference (that's what costs
    compute / latency). Of those, `exec_horizon` are played out open-loop before
    the next inference; the rest are discarded (receding horizon). Inference must
    keep up at 1 per exec_horizon control periods, so:

        budget = exec_horizon / control_hz

    A pipeline slower than one control step is still real-time as long as it
    commits enough actions per inference. exec_horizon defaults to the full chunk
    (pure open-loop) and is capped at chunk_horizon (can't execute more than was
    predicted). Assumes inference overlaps execution of the previous chunk.
    """
    ch = chunk_horizon(cfg)
    if exec_horizon in (None, 0, ""):
        eh = ch
    else:
        eh = max(1, min(int(exec_horizon), ch))
    period = 1.0 / control_hz
    budget = eh * period
    return {
        "hz": control_hz,
        "chunk_horizon": ch,     # predicted (drives latency)
        "exec_horizon": eh,      # executed open-loop (drives the budget)
        "period": period,
        "budget": budget,
        "ok": total_time <= budget,
        "achievable_hz": (eh / total_time) if total_time else 0.0,
    }


def _transformer_forward(
    dims, n_layers, q_seqlen, kv_seqlen, batchsize, a_byte, w_byte, kv_byte,
    bandwidth, max_OPS, onchip_buffer=0, use_flashattention=False,
    gated_mlp=True, attention_type="quadratic",
):
    """Cost of one forward pass of an arbitrary transformer, summed over layers.

    Generalizes the per-layer op walk to arbitrary query/kv lengths (q != kv,
    e.g. chunked/cross attention) and to linear attention -- capabilities the
    LLM-centric analyze() doesn't expose. Returns (time, OPs, weight) for the
    whole stack. Used for the flow/parallel action expert.
    """
    hidden = dims["hidden"]
    heads = dims["heads"]
    kv_heads = dims.get("kv_heads", heads)
    inter = dims["intermediate"]
    ctx = OpContext(
        batchsize, a_byte, w_byte, kv_byte,
        hidden_size=hidden, num_attention_heads=heads, num_key_value_heads=kv_heads,
        head_size=hidden // heads, onchip_buffer=onchip_buffer,
    )
    kvh_dim = hidden * kv_heads // heads
    if gated_mlp:
        lin = {
            "q_proj": (hidden, hidden), "k_proj": (hidden, kvh_dim), "v_proj": (hidden, kvh_dim),
            "out_proj": (hidden, hidden), "gate_proj": (hidden, inter), "up_proj": (hidden, inter),
            "down_proj": (inter, hidden),
        }
    else:
        lin = {
            "q_proj": (hidden, hidden), "k_proj": (hidden, kvh_dim), "v_proj": (hidden, kvh_dim),
            "out_proj": (hidden, hidden), "fc1": (hidden, inter), "fc2": (inter, hidden),
        }

    time = ops = weight = mem = 0.0

    def add(res):
        nonlocal time, ops, weight, mem
        ev = evaluate_op(**res, bandwidth=bandwidth, max_OPS=max_OPS)
        time += ev["inference_time"]
        ops += res["OPs"]
        weight += res["load_weight"]
        mem += ev["memory_access"]

    for name, (ic, oc) in lin.items():
        add(OP_HANDLERS["linear"](ctx, q_seqlen, kv_seqlen, ic=ic, oc=oc, is_kv_proj=name in ("k_proj", "v_proj")))
    if attention_type == "linear":
        add(OP_HANDLERS["linear_attention"](ctx, q_seqlen, kv_seqlen))
    elif use_flashattention:
        add(OP_HANDLERS["fused_attention"](ctx, q_seqlen, kv_seqlen))
    else:
        add(OP_HANDLERS["qk_matmul"](ctx, q_seqlen, kv_seqlen))
        add(OP_HANDLERS["sv_matmul"](ctx, q_seqlen, kv_seqlen))
        add(OP_HANDLERS["softmax"](ctx, q_seqlen, kv_seqlen))
    add(OP_HANDLERS["norm"](ctx, q_seqlen, kv_seqlen))
    add(OP_HANDLERS["norm"](ctx, q_seqlen, kv_seqlen))
    add(OP_HANDLERS["add"](ctx, q_seqlen, kv_seqlen))
    add(OP_HANDLERS["add"](ctx, q_seqlen, kv_seqlen))
    add(OP_HANDLERS["act"](ctx, q_seqlen, kv_seqlen))
    return time * n_layers, ops * n_layers, weight * n_layers, mem * n_layers


def analyze_vla(
    vla_cfg,
    hardware,
    num_text_tokens=256,
    num_images=1,
    batchsize=1,
    w_bit=16,
    a_bit=16,
    kv_bit=None,
    use_flashattention=False,
):
    """Analyze one observe->prefill VLA step. Returns a phase/total breakdown
    plus a memory-fit verdict for the given hardware.

    num_images: number of camera views. They are encoded through the shared
    vision tower (batched -> weights counted once, compute scales with N) and
    contribute N * tokens_per_image to the LLM prefix (and the action expert's
    cross-attention).
    """
    if kv_bit is None:
        kv_bit = a_bit
    w_byte, a_byte, kv_byte = w_bit / 8, a_bit / 8, kv_bit / 8

    # --- Vision encoder (N camera views, batched through one tower) --------
    vmp = importlib.import_module("model_params.vision_encoders").model_params[vla_cfg.vision_model_id]
    n_img = vit_cfg.get_num_image_tokens(vmp)          # tokens per image
    n_patches = n_img - (1 if getattr(vmp, "has_cls_token", False) else 0)
    patch_dim = vit_cfg.get_patch_dim(vmp)
    vision_hidden = vit_cfg.get_hidden_size(vmp)
    vision_heads = vit_cfg.get_num_attention_heads(vmp)
    vision_batch = batchsize * num_images               # N images share the tower weights
    total_img_tokens = num_images * n_img               # image tokens in the LLM prefix

    v_an = ModelAnalyzer(vla_cfg.vision_model_id, hardware, "configs/vit.py", source="vision_encoders")
    v_res = v_an.analyze(
        seqlen=n_img, batchsize=vision_batch, w_bit=w_bit, a_bit=a_bit, kv_bit=kv_bit,
        use_flashattention=use_flashattention,
    )
    v_pf = v_res["total_results"]["prefill"]
    bandwidth, max_OPS, _ = v_an.get_hardware_info()

    # patch embedding runs once per image (not per layer)
    v_ctx = OpContext(
        vision_batch, a_byte, w_byte, kv_byte,
        hidden_size=vision_hidden, num_attention_heads=vision_heads,
        num_key_value_heads=vision_heads, head_size=vision_hidden // vision_heads,
        onchip_buffer=0,
    )
    pe = OP_HANDLERS["patch_embed"](v_ctx, None, None, num_patches=n_patches, patch_dim=patch_dim)
    pe_eval = evaluate_op(**pe, bandwidth=bandwidth, max_OPS=max_OPS)

    # --- LLM backbone prefill over [all image tokens + text tokens] -------
    l_an = ModelAnalyzer(vla_cfg.llm_model_id, hardware, vla_cfg.llm_config_file, source=vla_cfg.llm_source)
    llm_hidden = l_an.config.get_hidden_size(l_an.model_params)
    prefix_len = total_img_tokens + num_text_tokens
    l_res = l_an.analyze(
        seqlen=prefix_len, batchsize=batchsize, w_bit=w_bit, a_bit=a_bit, kv_bit=kv_bit,
        use_flashattention=use_flashattention,
    )
    l_pf = l_res["total_results"]["prefill"]

    # --- Projector: vision_hidden -> llm_hidden, per image token ----------
    p_ctx = OpContext(
        batchsize, a_byte, w_byte, kv_byte,
        hidden_size=llm_hidden, num_attention_heads=1, num_key_value_heads=1,
        head_size=1, onchip_buffer=0,
    )
    proj = OP_HANDLERS["linear"](p_ctx, total_img_tokens, total_img_tokens, ic=vision_hidden, oc=llm_hidden, is_kv_proj=False)
    proj_eval = evaluate_op(**proj, bandwidth=bandwidth, max_OPS=max_OPS)

    # --- Action generation ------------------------------------------------
    head = vla_cfg.action_head
    if head == "ar":
        # Autoregressive discrete action tokens via the LLM backbone decode.
        n_action_tokens = vla_cfg.action_horizon * vla_cfg.tokens_per_action
        l_dec = l_res["total_results"]["decode"]  # one decode step at ~prefix_len context
        action = _phase(
            n_action_tokens * l_dec["inference_time"], n_action_tokens * l_dec["OPs"],
            weight=0.0,  # reuses the resident LLM weights
            memory_access=n_action_tokens * l_dec["memory_access"],
            mode="ar", action_tokens=n_action_tokens,
        )
    elif head in ("flow", "parallel"):
        # Separate action expert run num_steps times over the action chunk.
        steps = 1 if head == "parallel" else vla_cfg.num_flow_steps
        dims = vla_cfg.action_expert or DEFAULT_ACTION_EXPERT
        chunk = vla_cfg.action_chunk
        # The expert's action queries (chunk) cross-attend to the cached VLM
        # prefix KV AND self-attend over the chunk -> attention kv length is
        # prefix_len + chunk. The expert only projects its own action tokens
        # (q_seqlen = chunk); the prefix K/V come from the VLM prefill (read from
        # cache, not re-projected here). Approximation: prefix KV is costed in
        # the expert's head space, not the (larger) VLM head space.
        expert_kv_len = prefix_len + chunk
        per_t, per_ops, expert_weight, per_mem = _transformer_forward(
            dims, dims["layers"], chunk, expert_kv_len, batchsize, a_byte, w_byte, kv_byte,
            bandwidth, max_OPS, use_flashattention=use_flashattention,
            gated_mlp=True, attention_type=vla_cfg.expert_attention,
        )
        action = _phase(
            steps * per_t, steps * per_ops,
            weight=expert_weight,  # expert weights resident once (reloaded each step in time)
            memory_access=steps * per_mem,
            mode=head, steps=steps, chunk=chunk, expert_attention=vla_cfg.expert_attention,
        )
    else:
        raise ValueError(f"unknown action_head {head!r}")

    # --- Aggregate phases -------------------------------------------------
    phases = {
        "patch_embed": _phase(pe_eval["inference_time"], pe["OPs"], pe["load_weight"],
                              memory_access=pe_eval["memory_access"]),
        "vision_encoder": _phase(v_pf["inference_time"], v_pf["OPs"], v_pf["memory_consumption_weight"],
                                 memory_access=v_pf["memory_access"]),
        "projector": _phase(proj_eval["inference_time"], proj["OPs"], proj["load_weight"],
                            memory_access=proj_eval["memory_access"]),
        "llm_prefill": _phase(
            l_pf["inference_time"], l_pf["OPs"], l_pf["memory_consumption_weight"],
            memory_access=l_pf["memory_access"], kv_cache=l_pf["memory_consumption_kv_cache"],
        ),
        "action_generation": action,
    }
    total_time = sum(p["time"] for p in phases.values())
    total_weight = sum(p["weight"] for p in phases.values())
    kv_cache = phases["llm_prefill"]["kv_cache"]
    # phases run sequentially -> activation memory is reused, so peak is the max
    act_peak = max(v_pf["memory_consumption_tmp_act"], l_pf["memory_consumption_tmp_act"])
    peak_memory = total_weight + kv_cache + act_peak

    capacity = hardware_params[hardware].get("memory_capacity")
    fits = (peak_memory <= capacity) if capacity else None

    return {
        "config": vla_cfg.name,
        "hardware": hardware,
        "num_images": num_images,
        "tokens_per_image": n_img,
        "num_image_tokens": total_img_tokens,
        "prefix_len": prefix_len,
        "phases": phases,
        "total_time": total_time,
        "total_weight": total_weight,
        "kv_cache": kv_cache,
        "act_peak": act_peak,
        "peak_memory": peak_memory,
        "memory_capacity": capacity,
        "fits_in_memory": fits,
        "hardware_info": {"bandwidth": bandwidth, "max_OPS": max_OPS},
    }
