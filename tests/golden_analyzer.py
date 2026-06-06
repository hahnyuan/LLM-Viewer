"""Golden-output regression harness for ModelAnalyzer.

Runs ModelAnalyzer over a fixed matrix of (model, hardware, config) cases,
flattens every numeric / string output, and either:

  * writes the golden snapshot          --> `python tests/golden_analyzer.py --update`
  * compares against the saved snapshot  --> `python tests/golden_analyzer.py`

The purpose is to lock the *current* analyzer numbers before refactoring
`ModelAnalyzer.analyze()` into op-type handlers, so the refactor can be proven
to produce identical results.

Cases are restricted to models that load without gated HF access:
  * facebook/opt-125m                     (huggingface source, MHA)
  * TinyLlama/TinyLlama-1.1B-Chat-v1.0    (huggingface source, GQA: 32 vs 4 heads)
  * DiT-S/2                               (local model_params source, no network)

The TinyLlama (GQA) cases matter because attention cost has a head-count path
that differs between MHA and GQA; the MHA-only models can't exercise it.
"""

import argparse
import json
import math
import os
import sys

# allow running from anywhere: make repo root importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from model_analyzer import ModelAnalyzer  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_analyzer.json")

# Default numeric comparison tolerance. A faithful refactor should reproduce
# results to well within this; it only absorbs last-bit float reassociation.
DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 0.0


# Each case fully specifies one analyzer run. `kind` selects the entry point.
CASES = [
    # --- opt-125m (huggingface source) -------------------------------------
    {
        "name": "opt125m/A6000/b1_s1024_fp16",
        "model_id": "facebook/opt-125m",
        "hardware": "nvidia_A6000",
        "source": "huggingface",
        "kind": "analyze",
        "params": {"seqlen": 1024, "batchsize": 1},
    },
    {
        "name": "opt125m/A6000/b1_s1024_fp16_flash",
        "model_id": "facebook/opt-125m",
        "hardware": "nvidia_A6000",
        "source": "huggingface",
        "kind": "analyze",
        "params": {"seqlen": 1024, "batchsize": 1, "use_flashattention": True},
    },
    {
        "name": "opt125m/A6000/b16_s2048_fp16",
        "model_id": "facebook/opt-125m",
        "hardware": "nvidia_A6000",
        "source": "huggingface",
        "kind": "analyze",
        "params": {"seqlen": 2048, "batchsize": 16},
    },
    {
        "name": "opt125m/A6000/b1_s1024_w8a8kv8",
        "model_id": "facebook/opt-125m",
        "hardware": "nvidia_A6000",
        "source": "huggingface",
        "kind": "analyze",
        "params": {"seqlen": 1024, "batchsize": 1, "w_bit": 8, "a_bit": 8, "kv_bit": 8},
    },
    {
        "name": "opt125m/H100/b1_s2048_fp16",
        "model_id": "facebook/opt-125m",
        "hardware": "nvidia_H100",
        "source": "huggingface",
        "kind": "analyze",
        "params": {"seqlen": 2048, "batchsize": 1},
    },
    {
        "name": "opt125m/A6000/generate_p128_g64",
        "model_id": "facebook/opt-125m",
        "hardware": "nvidia_A6000",
        "source": "huggingface",
        "kind": "generate",
        "params": {"prompt_len": 128, "gen_len": 64, "batchsize": 1},
    },
    # --- TinyLlama (huggingface source, GQA) -------------------------------
    {
        "name": "tinyllama/A6000/b1_s1024_fp16",
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "hardware": "nvidia_A6000",
        "source": "huggingface",
        "kind": "analyze",
        "params": {"seqlen": 1024, "batchsize": 1},
    },
    {
        "name": "tinyllama/A6000/b1_s1024_fp16_flash",
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "hardware": "nvidia_A6000",
        "source": "huggingface",
        "kind": "analyze",
        "params": {"seqlen": 1024, "batchsize": 1, "use_flashattention": True},
    },
    {
        "name": "tinyllama/A6000/b1_s512_w8a8kv8",
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "hardware": "nvidia_A6000",
        "source": "huggingface",
        "kind": "analyze",
        "params": {"seqlen": 512, "batchsize": 1, "w_bit": 8, "a_bit": 8, "kv_bit": 8},
    },
    # --- Edge hardware (Jetson) + INT4 path --------------------------------
    {
        "name": "opt125m/orin_nx/b1_s1024_w4a4kv4",  # exercises INT4 throughput tier
        "model_id": "facebook/opt-125m",
        "hardware": "jetson_orin_nx_16gb",
        "source": "huggingface",
        "kind": "analyze",
        "params": {"seqlen": 1024, "batchsize": 1, "w_bit": 4, "a_bit": 4, "kv_bit": 4},
    },
    {
        "name": "tinyllama/orin_nx/b1_s2048_fp16",
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "hardware": "jetson_orin_nx_16gb",
        "source": "huggingface",
        "kind": "analyze",
        "params": {"seqlen": 2048, "batchsize": 1},
    },
    # --- Vision encoder (ViT) + VLA orchestration --------------------------
    {
        "name": "clip_vit_l/A6000/b1_s577_fp16",
        "model_id": "clip_vit_large_p14_336",
        "hardware": "nvidia_A6000",
        "source": "vision_encoders",
        "config_file": "configs/vit.py",
        "kind": "analyze",
        "params": {"seqlen": 577, "batchsize": 1},  # (336/14)^2 + 1 CLS
    },
    {
        "name": "clip_vit_l/orin_nx/b1_s577_w8a8",
        "model_id": "clip_vit_large_p14_336",
        "hardware": "jetson_orin_nx_16gb",
        "source": "vision_encoders",
        "config_file": "configs/vit.py",
        "kind": "analyze",
        "params": {"seqlen": 577, "batchsize": 1, "w_bit": 8, "a_bit": 8, "kv_bit": 8},
    },
    {
        "name": "vla/tinyvla_demo/orin_nx/w4a8kv8",  # AR action head
        "model_id": "tinyvla_demo",
        "hardware": "jetson_orin_nx_16gb",
        "kind": "vla",
        "params": {"num_text_tokens": 16, "batchsize": 1, "w_bit": 4, "a_bit": 8, "kv_bit": 8},
    },
    {
        "name": "vla/tinyvla_demo+flow/orin_nx/w4a8kv8",  # flow head as a config override
        "model_id": "tinyvla_demo",
        "hardware": "jetson_orin_nx_16gb",
        "kind": "vla",
        "overrides": {"action_head": "flow"},
        "params": {"num_text_tokens": 16, "batchsize": 1, "w_bit": 4, "a_bit": 8, "kv_bit": 8},
    },
    {
        "name": "vla/tinyvla_demo+flow+linear/orin_nx/w4a8kv8",  # flow + linear-attention expert
        "model_id": "tinyvla_demo",
        "hardware": "jetson_orin_nx_16gb",
        "kind": "vla",
        "overrides": {"action_head": "flow", "expert_attention": "linear"},
        "params": {"num_text_tokens": 16, "batchsize": 1, "w_bit": 4, "a_bit": 8, "kv_bit": 8},
    },
    {
        "name": "vla/pi0_like/agx_orin/w8a8kv8",  # PaliGemma (Gemma-2B) + flow expert, local params
        "model_id": "pi0_like",
        "hardware": "jetson_agx_orin_64gb",
        "kind": "vla",
        "params": {"num_text_tokens": 16, "batchsize": 1, "w_bit": 8, "a_bit": 8, "kv_bit": 8},
    },
    # --- DiT-S/2 (local model_params source) -------------------------------
    {
        "name": "DiT-S2/A6000/b1_s256_fp16",
        "model_id": "DiT-S/2",
        "hardware": "nvidia_A6000",
        "source": "DiT",
        "kind": "analyze",
        "params": {"seqlen": 256, "batchsize": 1},
    },
]


def flatten(obj, prefix=""):
    """Flatten a nested dict into {path: leaf} for numeric/string/bool leaves."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}/{k}" if prefix else str(k)))
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        out[prefix] = obj
    else:
        # unexpected type -> stringify so a change is at least visible
        out[prefix] = f"<{type(obj).__name__}>{obj!r}"
    return out


def run_case(case):
    if case["kind"] == "vla":
        import dataclasses
        from vla import analyze_vla
        from model_params.vla_models import VLA_MODELS
        cfg = VLA_MODELS[case["model_id"]]
        if case.get("overrides"):  # action_head / expert_attention / etc. as config, not a variant preset
            cfg = dataclasses.replace(cfg, **case["overrides"])
        return flatten(analyze_vla(cfg, case["hardware"], **case["params"]))
    analyzer = ModelAnalyzer(
        case["model_id"], case["hardware"], case.get("config_file"), source=case["source"]
    )
    if case["kind"] == "analyze":
        result = analyzer.analyze(**case["params"])
    elif case["kind"] == "generate":
        result = analyzer.analyze_generate_task(**case["params"])
    else:
        raise ValueError(f"unknown kind {case['kind']}")
    return flatten(result)


def numbers_match(a, b, rtol, atol):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isinf(a) or math.isinf(b):
            return a == b
        return abs(a - b) <= atol + rtol * abs(b)
    return a == b


def compare(golden, current, rtol, atol):
    """Return list of (key, golden_value, current_value) mismatches."""
    mismatches = []
    keys = sorted(set(golden) | set(current))
    for k in keys:
        if k not in golden:
            mismatches.append((k, "<MISSING>", current[k]))
        elif k not in current:
            mismatches.append((k, golden[k], "<MISSING>"))
        elif not numbers_match(golden[k], current[k], rtol, atol):
            mismatches.append((k, golden[k], current[k]))
    return mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="write the golden snapshot instead of comparing")
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--verbose", action="store_true", help="show every mismatch (default caps at 20 per case)")
    args = parser.parse_args()

    current = {}
    for case in CASES:
        print(f"[run] {case['name']}", flush=True)
        current[case["name"]] = run_case(case)

    if args.update:
        with open(GOLDEN_PATH, "w") as f:
            json.dump(current, f, indent=2, sort_keys=True)
        n_vals = sum(len(v) for v in current.values())
        print(f"\nWrote golden snapshot: {GOLDEN_PATH}")
        print(f"  {len(current)} cases, {n_vals} values")
        return 0

    if not os.path.exists(GOLDEN_PATH):
        print(f"\nERROR: golden snapshot not found at {GOLDEN_PATH}. Run with --update first.")
        return 2

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    total_mismatches = 0
    failed_cases = 0
    for name in sorted(set(golden) | set(current)):
        if name not in golden:
            print(f"[NEW CASE] {name} (not in golden — run --update)")
            failed_cases += 1
            continue
        if name not in current:
            print(f"[MISSING CASE] {name} (in golden but not produced)")
            failed_cases += 1
            continue
        mismatches = compare(golden[name], current[name], args.rtol, args.atol)
        if mismatches:
            failed_cases += 1
            total_mismatches += len(mismatches)
            print(f"[FAIL] {name}: {len(mismatches)} mismatch(es)")
            shown = mismatches if args.verbose else mismatches[:20]
            for key, g, c in shown:
                print(f"    {key}: golden={g!r} current={c!r}")
            if not args.verbose and len(mismatches) > len(shown):
                print(f"    ... ({len(mismatches) - len(shown)} more; use --verbose)")
        else:
            print(f"[PASS] {name}")

    print()
    if failed_cases:
        print(f"RESULT: FAIL — {failed_cases} case(s), {total_mismatches} value mismatch(es)")
        return 1
    print(f"RESULT: PASS — all {len(current)} cases match golden")
    return 0


if __name__ == "__main__":
    sys.exit(main())
