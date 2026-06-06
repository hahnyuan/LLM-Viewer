"""CLI: roofline analysis of one VLA observe->prefill step on edge hardware.

Examples:
    python analyze_vla_cli.py tinyvla_demo jetson_orin_nx_16gb
    python analyze_vla_cli.py tinyvla_demo jetson_orin_nx_16gb --w_bit 4 --a_bit 8 --kv_bit 8
    python analyze_vla_cli.py openvla_7b jetson_agx_orin_64gb   # needs HF auth (gated LLM)
"""

import argparse

from vla import analyze_vla, control_budget
from model_params.vla_models import VLA_MODELS
from utils import str_number, str_number_time


def _gib(n):
    return f"{n / (1024 ** 3):.2f} GiB"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vla", choices=list(VLA_MODELS.keys()), help="VLA preset")
    parser.add_argument("hardware", type=str, help="hardware key, e.g. jetson_orin_nx_16gb")
    parser.add_argument("--num_text_tokens", type=int, default=256, help="language prompt tokens")
    parser.add_argument("--num_images", type=int, default=1, help="number of camera views")
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--w_bit", type=int, default=16)
    parser.add_argument("--a_bit", type=int, default=16)
    parser.add_argument("--kv_bit", type=int, default=None)
    parser.add_argument("--use_flashattention", action="store_true")
    parser.add_argument("--control_hz", type=float, default=None, help="control frequency; checks the latency budget")
    parser.add_argument("--exec_horizon", type=int, default=10,
                        help="actions executed open-loop before replanning (0 = full chunk)")
    args = parser.parse_args()

    r = analyze_vla(
        VLA_MODELS[args.vla], args.hardware,
        num_text_tokens=args.num_text_tokens, num_images=args.num_images, batchsize=args.batchsize,
        w_bit=args.w_bit, a_bit=args.a_bit, kv_bit=args.kv_bit,
        use_flashattention=args.use_flashattention,
    )

    print(f"\nVLA: {r['config']}  |  Hardware: {r['hardware']}")
    print(f"{r['num_images']} cam x {r['tokens_per_image']} = {r['num_image_tokens']} image tokens  |  "
          f"LLM prefix: {r['prefix_len']} tokens (+{args.num_text_tokens} text)  |  "
          f"w{args.w_bit}a{args.a_bit}kv{args.kv_bit or args.a_bit}")
    print("-" * 64)
    print(f"{'phase':<16}{'OPs':>12}{'latency':>14}")
    for name, p in r["phases"].items():
        print(f"{name:<16}{str_number(p['OPs']):>12}{str_number_time(p['time']) + 's':>14}")
    print("-" * 64)
    print(f"{'TOTAL':<16}{'':>12}{str_number_time(r['total_time']) + 's':>14}"
          f"   ({1 / r['total_time']:.1f} steps/s)")

    print("\nMemory footprint:")
    print(f"  weights        {_gib(r['total_weight'])}")
    print(f"  kv cache       {_gib(r['kv_cache'])}")
    print(f"  activations    {_gib(r['act_peak'])} (peak, phases sequential)")
    print(f"  peak total     {_gib(r['peak_memory'])}")
    if r["memory_capacity"]:
        verdict = "FITS" if r["fits_in_memory"] else "DOES NOT FIT"
        print(f"  device cap     {_gib(r['memory_capacity'])}  ->  {verdict}")

    if args.control_hz:
        cb = control_budget(VLA_MODELS[args.vla], r["total_time"], args.control_hz, exec_horizon=args.exec_horizon)
        ok = "within budget" if cb["ok"] else "OVER budget"
        print(f"\nControl @ {args.control_hz} Hz: predicts {cb['chunk_horizon']} actions/inference, "
              f"executes {cb['exec_horizon']} open-loop  ->  "
              f"budget {cb['exec_horizon']}/{args.control_hz}Hz = {str_number_time(cb['budget'])}s  vs  "
              f"latency {str_number_time(r['total_time'])}s ({ok}); sustains ~{cb['achievable_hz']:.1f} Hz")
    print()


if __name__ == "__main__":
    main()
