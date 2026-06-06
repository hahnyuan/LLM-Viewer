# VLA Edge-Inference Roofline Analyzer — Design Plan

**Status:** Draft / in progress
**Last updated:** 2026-06-03
**Owner:** Pradeep Kadubandi

## 1. Goal (500-ft view)

Turn LLM-Viewer into a **roofline analyzer for VLA (Vision-Language-Action) models running inference on edge robotics devices**. Training is assumed to happen off-device, so the focus is purely inference: per-layer arithmetic intensity, memory traffic, roofline-bound classification, end-to-end latency, and peak-memory footprint — evaluated against the constrained compute/bandwidth/capacity of edge hardware.

The headline question the tool should answer for a given (hardware, model, optimization, inference) configuration:

> *Does end-to-end inference fit the control-frequency latency budget AND fit in device memory on this edge platform — and which ops are the bottleneck?*

## 2. Assessment of the existing codebase

The repo is small with a cleaner architecture than its rough edges suggest. Four layers:

| Layer | Files | Role | Reuse |
|---|---|---|---|
| **Roofline core** | `roofline_model.py` | Pure fn: `(bw, peak_OPS, OPs, bytes) → (intensity, perf, bound)` | As-is |
| **Analyzer** | `model_analyzer.py` | Walks a transformer layer, computes OPs + memory traffic per op for prefill/decode, aggregates ×num_layers, tracks peak memory | Refactor |
| **Architecture descriptors** | `configs/*.py`, `hardwares/hardware_params.py`, `model_params/*.py` | Per-model config modules expose layer-dim getters + a `transformer_layer_graph` DAG; hardware is a dict of `{bandwidth, FP16, INT8, onchip_buffer}` | Extend |
| **Surfaces** | `analyze_cli.py`, `backend_app.py` (Flask), `frontend/` (Vue + G6) | CLI + REST + interactive per-node roofline graph | Reuse/extend |

### The key architectural fact

The *architecture* (which ops exist, how they connect) lives in pluggable config modules, **but the per-op cost math is hardcoded inside `ModelAnalyzer.analyze()`** (~250 lines). Config DAG node names (`qk_matmul`, `gate_proj`, …) are just strings that must match what `analyze()` already knows how to cost. Adding a genuinely new op type (linear attention, diffusion denoiser, conv patch-embed) means editing `analyze()`, not just adding a config.

**That single coupling is the only real architectural obstacle to the VLA goal.** Everything else transfers directly.

## 3. Decision: build on top + one focused refactor

**Chosen approach: build on top of the existing repo, with a targeted refactor of `analyze()`.** (Alternatives considered: minimal in-place extension; full from-scratch rebuild.)

Rationale:
- The expensive, hard-to-rebuild parts are already correct and reusable: roofline plumbing, peak-memory accounting via the DAG, HF `AutoConfig` ingestion, and especially the **Vue + G6 per-node roofline visualizer** (which is essentially the end goal).
- The one blocker — the monolithic `analyze()` — is ~250 lines; refactoring into dispatchable op handlers is days-scale, not a rewrite.
- From-scratch would re-derive all the plumbing and frontend to gain a cleaner `analyze()` achievable with one refactor.

**Trade-off accepted:** the code has rough edges (typos like `avaliable_`, commented-out blocks, zero tests, tight coupling). The refactor therefore ships with a golden-output test harness so refactors can be proven number-for-number identical before extending.

**The core refactor:** move per-op cost from `analyze()` into **op-type handlers**; DAG nodes carry an `op_type`; `analyze()` becomes a dispatcher walking the DAG. New op types become new handlers + config modules — no surgery on a 250-line method each time.

## 4. The four input axes

### (a) Edge hardware platforms (top 3)

Robotics-edge is overwhelmingly NVIDIA Jetson. Roofline needs peak compute per dtype, peak memory bandwidth, on-chip SRAM, and — uniquely at the edge — **unified-memory capacity** and **power envelope** (peak TOPS is only reachable within the power budget; sustained robotics inference is often capacity/bandwidth-bound, not compute-bound).

| Platform | Peak compute (dense) | Mem BW | Capacity | Notes |
|---|---|---|---|---|
| **Jetson AGX Orin 64GB** | ~85 FP16 TFLOPS / ~170 INT8 TOPS | 204.8 GB/s LPDDR5 | 64 GB unified | Default VLA robotics target today |
| **Jetson AGX Thor** (2025, Blackwell) | 1000s of FP4/FP8 TFLOPS | ~273 GB/s | 128 GB | Successor; adds FP8/FP4 (relevant to quant axis) |
| **Jetson Orin NX 16GB** | ~50 INT8 TOPS | 102 GB/s | 16 GB | Small robots; capacity-constrained → **first validation target** |

(Secondary: Qualcomm QRB5165/Hexagon NPU; Apple M-series research rigs.)

Hardware dict needs added fields: **`memory_capacity`** (flag "won't fit"), optional `FP8`/`INT4` peak, optional `power_W`.

### (b) Model architectures & per-module choices

VLA = vision encoder → VLM backbone → action head. Three pluggable knobs:

- **Vision encoder:** SigLIP ViT · DINOv2 ViT · dual SigLIP+DINOv2 (OpenVLA). Cost driver = image resolution → num patch tokens.
- **Backbone attention:** standard **quadratic softmax** · **linear attention** (SARA-RT-style) · sliding-window/local. Per-op-handler choice (qk/sv matmul cost O(L²) → O(L)).
- **Action head / decoder:** **autoregressive discrete tokens** (OpenVLA; gen = horizon × tokens/action, iterative decode) · **diffusion / flow-matching chunk** (π0, Octo, Diffusion Policy; cost = `num_steps × forward over chunk tokens`) · **parallel single-pass action expert**.

Reference models to seed configs: **OpenVLA** (SigLIP+DINOv2, Llama-2-7B, AR discrete), **π0 / π0-FAST** (PaliGemma + flow-matching expert / FAST tokenizer), **Octo** (ViT + diffusion head).

### (c) HF config leverage — the repo already does this

Pattern: `AutoConfig.from_pretrained(model_id)` supplies raw dims; a hand-written `configs/<Arch>.py` maps those into layer shapes + the DAG. The DiT path (`model_params/DiT.py`) shows the non-HF fallback: a local param dict. **For VLA, extend the same pattern** — a config module pulls dims from sub-model HF configs (vision tower + text config) and exposes the architecture knobs from (b). Knobs not in any HF config (attention type, action-head type, chunk size, denoising steps) become CLI/UI parameters, exactly as `--use_flashattention` and `--tp-size` already are.

### (d) Inference / optimization config

Already present: batch_size, seqlen, gen_length, w/a/kv quant bits, flash-attention, tp_size, stage.

Add for VLA: image resolution → num vision tokens, action chunk size, action horizon, num denoising/flow steps, and **control frequency (Hz)** — converting latency into the answer that matters: *does end-to-end latency fit the control budget on this device?*

**Control budget under action chunking (two distinct horizons, see `vla.py` `chunk_horizon`/`control_budget`):**
- **chunk horizon** (`action_chunk` / `action_horizon`) = actions *predicted* per inference → drives **latency** (the action-phase compute). 
- **execution horizon** (`exec_horizon`, default = full chunk) = actions *executed open-loop* before replanning (receding horizon) → drives the **budget = exec_horizon / control_hz**.

So budget is NOT `1/control_hz` — one inference buys `exec_horizon` control periods of runway (sustained open-loop rate = `exec_horizon / latency`). OpenVLA (horizon 1) collapses to `1/control_hz`. Assumes inference overlaps execution of the previous chunk.

## 5. Analyzer design for VLA

End-to-end per control step becomes three composed phases (vs. today's prefill/decode pair):

1. **Vision encode** (once per observation): patch-embed + ViT layers, full attention over image tokens.
2. **Prefix prefill** over `[vision_tokens + language_tokens]`.
3. **Action generation**, dispatched by head type:
   - AR → iterative decode (like today's `analyze_generate_task`).
   - diffusion/flow → `num_steps ×` forward over chunk tokens.
   - parallel → single forward.

Each phase reuses the existing roofline + footprint machinery. New work = op handlers + phase orchestration. Output: per-op intensity & bound (as today) plus top-line **latency vs control-budget** and **fits-in-memory** verdict per device.

## 6. Phased execution plan

1. **Lock current behavior** — golden snapshot harness over existing OPT/DiT × hardware matrix (`tests/golden_analyzer.py`). *(Safety net.)*  ✅ DONE
2. **Refactor `analyze()` → op-handler dispatch** (`op_handlers.py`: registry of op-type cost handlers; `analyze()` is now a dispatcher). Proven bit-identical vs. golden (7/7 cases PASS). *(Core enabling refactor.)*  ✅ DONE
3. **Add edge hardware** + `memory_capacity`/`INT4` fields. Added Jetson Orin NX 16GB / AGX Orin 64GB / AGX Thor (`hardwares/hardware_params.py`), `memory_capacity` on every entry, `INT4` peak where supported, and INT4 throughput selection in `get_hardware_info`. (FP8 deferred to the dtype-aware quant step — bitwidth alone can't distinguish FP8 from INT8.) Golden: 12/12 PASS (added Orin-NX w4a4kv4 + GQA cases). ✅ DONE
4. **Vision encoder + multi-phase orchestration.** A ViT is a single bidirectional transformer forward, so the encoder reuses the golden-locked `analyze()` via `configs/vit.py` + `model_params/vision_encoders.py` (CLIP-L, DINOv2-L, SigLIP-L presets). Added the `patch_embed` op handler and `roofline_model.evaluate_op` (shared roofline-apply, extracted golden-safe). `vla.py` orchestrates **vision-encode → projector → LLM-prefill**, returns a per-phase latency/OPs breakdown, and a **memory-fit verdict** vs the device's `memory_capacity` plus an optional control-frequency budget check. `analyze_vla_cli.py` + `model_params/vla_models.py` presets (runnable `tinyvla_demo`; `openvla_7b` documents real scale). Golden: 15/15 PASS (added ViT + VLA cases). Action-generation phase deferred to step 5. ✅ DONE
5. **Action-generation phase + new op handlers.** Added the action phase to `vla.py`: **ar** (autoregressive discrete tokens — iterate LLM decode for horizon×tokens/action; OpenVLA), **flow** (diffusion/flow-matching — a separate action expert run num_steps× over the action chunk; π0/Octo), **parallel** (expert once). Added the `linear_attention` op (SARA-RT, O(L)) and a `_transformer_forward` helper that costs an arbitrary transformer at any q/kv length and attention type (enables the expert + future cross-attention). Exposed chunk/steps/Hz on the CLI. Presets: AR (`tinyvla_demo`, `openvla_7b`), flow (`tinyvla_flow_demo`, `pi0_like`), linear-attn (`tinyvla_flow_linear_demo`). Golden: 17/17 PASS. ✅ DONE
   - *Deferred:* backbone-wide linear attention (analyze() still uses quadratic; linear_attention is wired for the action expert only), and flow expert cross-attention to the prefix KV (currently self-attention over the chunk only).
6. **Surface in frontend — as a SEPARATE VLA viewer, not conflated with the LLM viewer.** The existing LLM-viewer-web (cloud LLM serving) and the new VLA-viewer-web (edge robotics) are distinct products with different knobs/hardware/headline outputs. Approach (decided): **vue-router, shared shell** — `/` = existing LLM viewer (behavior untouched), `/vla` = new. Additive backend: `get_vla_graph.py` + `POST /get_vla_graph` + `GET /get_vla_avaliable` (VLA presets + edge hardware); `/get_graph` untouched. Frontend: extract current `App.vue` body into `LlmView.vue` (one mechanical move), `App.vue` becomes `<router-view>`. VLA view is a **purpose-built phase dashboard** (latency-share bars per phase, big FITS / control-budget verdicts, per-phase metrics) rather than reusing the G6 DAG — `VlaView/VlaConfig/VlaReadout/VlaHeader.vue` new; `utils.js` reused; roofline chart available for per-phase drill-down. Phasing: (6a) router + LlmView extract, `/` byte-identical ✅; (6b) backend `/get_vla_graph` + `/get_vla_avaliable` ✅; (6c) VLA phase dashboard (verdict cards, latency-share bar, phase table, memory breakdown) ✅; (6d) polish — hardware split by `category` (cloud-only LLM viewer / edge-only VLA viewer, single source of truth in hardware_params), per-phase roofline drill-down (click a phase → its arithmetic-intensity vs the hardware roofline), dual-encoder caveat, Thor preliminary-spec note ✅. ✅ DONE
   - Per-phase AI/bound live in the backend adapter (`get_vla_graph.py`); `analyze_vla` gained per-phase `memory_access` + `hardware_info` (golden regenerated, 17/17). The drill-down makes the key insight visible: AR action decode is memory-bound (AI≈4) while prefill/projector are compute-bound.
   - **Action decoding (ar/flow/parallel) and expert attention (quadratic/linear SARA-RT) are UI/CLI *config options*, not separate model presets** — grouped under an "Architecture" header in the VLA panel. A base model (vision+LLM backbone) can be analyzed with any action decoder via `dataclasses.replace` on the preset (backend `_STR_OVERRIDES`). The redundant `tinyvla_flow_*` variant presets were removed; the dropdown lists backbones only (`tinyvla_demo`, `openvla_7b`, `pi0_like`).
7. **Seed configs for OpenVLA and π0** — both runnable. ✅ DONE
   - **OpenVLA** (`openvla_7b`): DINOv2-L vision + Llama-2-7B (ungated `NousResearch/Llama-2-7b-hf` mirror) + AR head.
   - **π0** (`pi0_like`): SigLIP-So400m vision (256 tokens @224) + Gemma-2B backbone + ~300M flow expert. PaliGemma is gated and has a composite HF config, so the Gemma-2B backbone is a **local param** (`model_params/llm_backbones.py`) analyzed with the Llama config (Gemma's per-op structure matches: gated MLP, MQA, 2 norms; head_dim = hidden/heads so projections are exact). Golden-locked (18 cases). e.g. AGX Orin 64GB w8a8 → 33ms/step, 30 steps/s, 2.48 GiB, fits.

## 7. Environment notes

- Dedicated conda env **`llm-viewer`** (Python 3.10). Activate: `source ~/miniconda3_2025_02/etc/profile.d/conda.sh && conda activate llm-viewer`.
- **Install with `python -m pip ...`** — bare `pip` resolves to system `/usr/bin/pip` and installs outside the env.
- Deps: transformers, flask, flask_cors, easydict, numpy. **PyTorch not installed and not needed** (pure config math; HF AutoConfig only).
- Removed dead `import torch.nn as nn` from `analyze_cli.py` / `analyze_gen_cli.py` so they run without torch.
- `transformers` 5.10.1 is much newer than the repo targets; opt-125m loads fine, but some pinned configs (chatglm3, older Llama) may differ — surface mismatches when building the test matrix.

## 8. Open decisions / parking lot

- **GQA qk_matmul quirk (found during step 2):** the original analyzer sized the qk_matmul Q-activation load with `num_attention_heads` in decode but `num_key_value_heads` in prefill — physically the Q activation always has attention-heads, so prefill looks like a bug. It only affects GQA models (MHA: nah==nkvh). Currently *preserved* bit-identically via `qk_matmul(..., query_act_heads=...)` (default = attention-heads). **Decision pending:** fix it (drop the legacy override → use attention-heads consistently) and regenerate the GQA golden, or keep as-is. Low urgency; revisit before trusting absolute (vs relative) prefill numbers on GQA VLAs.
- Concrete first VLA model to seed (OpenVLA vs π0): deferred until after the engine refactor.
- How to represent sustained-vs-peak compute under the Jetson power cap (roofline ceiling adjustment).
- Whether vision encoder amortization (encode once, act multiple times) needs explicit modeling for high-Hz control loops.
