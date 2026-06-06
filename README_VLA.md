# VLA-Viewer

Roofline analysis of **Vision-Language-Action (VLA)** models running inference on robotics **edge devices** (NVIDIA Jetson). Part of [**LLM-Viewer**](README.md) — it reuses the LLM roofline engine and op-cost model, and adds the VLA-specific pipeline, edge hardware, and a purpose-built dashboard. The two are kept as **separate products in one app**: `/llm` (cloud LLM serving) and `/vla` (edge robotics), with disjoint model and hardware lists.

<video src="https://github.com/user-attachments/assets/4ff0d590-6b4b-43a5-aa3b-9c27f2d89000" controls width="100%"></video>

_Can't see the player? [Watch the demo video](figs/vla_demo.mp4)._

The VLA Viewer models one observe→act inference step as a pipeline of phases —
**patch embed → vision encode → projector → LLM prefill → action generation** —
and reports, for a chosen edge device:

- **per-phase latency / OPs / arithmetic intensity** (memory- vs compute-bound); click a phase for its roofline plot;
- a **memory-fit verdict** vs the device's unified memory capacity;
- a **control-frequency budget** verdict under action chunking — does the policy keep up with the control loop?

## Architecture & inference choices are config options (not separate model variants)

- **Action decoding**: `autoregressive` (OpenVLA-style discrete tokens), `flow` (diffusion / flow-matching chunk, π0 / Octo), or `parallel` (single-pass action expert).
- **Attention**: `quadratic` or `linear` (SARA-RT) for the action expert.
- **Quantization** (weight / activation / KV-cache bits), image resolution → vision tokens, batch size, flash-attention.
- **Control budget** under chunking: a chunk of *H* predicted actions is executed open-loop, so the latency budget is `exec_horizon / control_hz` — not a single control period. **Chunk/horizon length** drives inference latency; **execution horizon** (steps run open-loop before replanning) drives the budget.

## Models

| Preset | Vision | Backbone | Default head | Runs offline |
|---|---|---|---|---|
| `tinyvla_demo` | CLIP-L/14 | TinyLlama-1.1B | autoregressive | ✅ |
| `openvla_7b` | DINOv2-L/14 | Llama-2-7B (ungated mirror) | autoregressive | ✅ |
| `pi0_like` | SigLIP-So400m | Gemma-2B (PaliGemma) + flow expert | flow | ✅ (local params) |

Any backbone can be analyzed with any action decoder — switch it in the **Architecture** panel.

## Edge hardware

Jetson **Orin NX 16GB**, **AGX Orin 64GB**, **AGX Thor (128GB)**. (The LLM Viewer shows datacenter/cloud GPUs; the two lists do not overlap. New devices self-classify via a `category` field in `hardwares/hardware_params.py`.)

## Command line

```bash
python3 analyze_vla_cli.py tinyvla_demo jetson_orin_nx_16gb --w_bit 4 --a_bit 8 --kv_bit 8 --control_hz 10
python3 analyze_vla_cli.py pi0_like    jetson_agx_orin_64gb --w_bit 8 --a_bit 8 --kv_bit 8 --control_hz 50
python3 analyze_vla_cli.py tinyvla_demo jetson_orin_nx_16gb --control_hz 10 --exec_horizon 8   # receding-horizon replanning
```

## Running the web viewer

One app serves both viewers — see [**Running locally**](README.md#running-locally-both-viewers) in the main README, then open **http://localhost:5173/** and pick the VLA viewer (`/vla`).

> NOTE: roofline times are the theoretical hardware ceiling — use them for **relative** comparison, not absolute timings.

---

← Back to [**LLM-Viewer**](README.md)
