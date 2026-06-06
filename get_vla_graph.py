"""Backend adapter: turn analyze_vla() output into the VLA dashboard payload.

Kept separate from get_model_graph.py (the LLM viewer's adapter) so the two
products don't conflate. Returns a JSON-friendly per-phase breakdown plus the
edge-deployment verdicts (memory-fit, control-frequency budget).
"""

import dataclasses

from vla import analyze_vla, control_budget
from model_params.vla_models import VLA_MODELS
from get_model_graph import get_quant_bit

# UI-overridable action-head fields (the rest of a preset is fixed).
_INT_OVERRIDES = ("action_horizon", "tokens_per_action", "num_flow_steps", "action_chunk")
_STR_OVERRIDES = ("action_head", "expert_attention")
_PHASE_META = ("mode", "steps", "chunk", "action_tokens", "expert_attention")


def get_vla_graph(vla_model_id, hardware, vla_config):
    cfg = VLA_MODELS[vla_model_id]

    # Apply UI overrides on top of the preset.
    overrides = {}
    for k in _INT_OVERRIDES:
        v = vla_config.get(k)
        if v not in (None, ""):
            overrides[k] = int(v)
    for k in _STR_OVERRIDES:
        v = vla_config.get(k)
        if v not in (None, ""):
            overrides[k] = v
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    w_bit = get_quant_bit(vla_config["w_quant"])
    a_bit = get_quant_bit(vla_config["a_quant"])
    kv_bit = get_quant_bit(vla_config["kv_quant"])
    num_text_tokens = int(vla_config.get("num_text_tokens", 256))
    num_images = int(vla_config.get("num_images", 1))
    batch_size = int(vla_config.get("batch_size", 1))
    use_flashattention = bool(vla_config.get("use_flashattention", False))

    r = analyze_vla(
        cfg, hardware,
        num_text_tokens=num_text_tokens, num_images=num_images, batchsize=batch_size,
        w_bit=w_bit, a_bit=a_bit, kv_bit=kv_bit,
        use_flashattention=use_flashattention,
    )

    hw = r["hardware_info"]
    bandwidth, max_OPS = hw["bandwidth"], hw["max_OPS"]
    turning_point = max_OPS / bandwidth  # arithmetic intensity where compute-bound begins

    total = r["total_time"]
    phases = []
    for name, p in r["phases"].items():
        mem = p.get("memory_access", 0)
        ai = (p["OPs"] / mem) if mem else 0.0
        phase = {
            "name": name,
            "time": p["time"],
            "OPs": p["OPs"],
            "memory_access": mem,
            "arithmetic_intensity": ai,
            "bound": "compute" if ai >= turning_point else "memory",
            "performance": (p["OPs"] / p["time"]) if p["time"] else 0.0,  # effective OPS
            "weight": p.get("weight", 0),
            "share": (p["time"] / total if total else 0),
        }
        for mk in _PHASE_META:
            if mk in p:
                phase[mk] = p[mk]
        phases.append(phase)

    payload = {
        "config": r["config"],
        "hardware": hardware,
        "num_image_tokens": r["num_image_tokens"],
        "prefix_len": r["prefix_len"],
        "action_head": cfg.action_head,
        "hardware_info": {"bandwidth": bandwidth, "max_OPS": max_OPS, "turning_point": turning_point},
        "phases": phases,
        "total_time": total,
        "steps_per_sec": (1.0 / total if total else 0.0),
        "memory": {
            "weights": r["total_weight"],
            "kv_cache": r["kv_cache"],
            "act_peak": r["act_peak"],
            "peak": r["peak_memory"],
            "capacity": r["memory_capacity"],
            "fits": r["fits_in_memory"],
        },
    }

    control_hz = vla_config.get("control_hz")
    if control_hz not in (None, ""):
        eh = vla_config.get("exec_horizon")
        eh = int(eh) if eh not in (None, "") else None
        payload["control"] = control_budget(cfg, total, float(control_hz), exec_horizon=eh)

    return payload
