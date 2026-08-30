"""Unified GPU benchmark runner."""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cases import ROOT, SUPPORTED_CASES, _operator_path, make_case, validate_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=int)
    parser.add_argument("--size", choices=("small", "medium", "large"), default="medium")
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare one untimed Triton result with the reference before timing",
    )
    args = parser.parse_args()
    if args.list:
        print("001-011: complete official tutorials; run the numbered Python file directly")
        print("012-100 unified cases:", " ".join(f"{x:03d}" for x in sorted(SUPPORTED_CASES)))
        return
    if args.op is None: parser.error("--op is required unless --list is used")
    if 1 <= args.op <= 11:
        _, path = _operator_path(args.op)
        print(f"{args.op:03d} is a complete official tutorial; run its file directly.")
        print(f"run: python3 {path.relative_to(ROOT)}")
        return
    if args.op not in SUPPORTED_CASES:
        parser.error("--op must be in [1, 100]")
    import triton
    tri, reference, bytes_moved = make_case(args.op, args.size)
    if args.check:
        skip_reason = validate_case(args.op, tri, reference)
        if skip_reason is None:
            print("correctness=PASS")
        else:
            print(f"correctness=SKIP reason={skip_reason}")
    tri_ms = triton.testing.do_bench(tri)
    ref_ms = triton.testing.do_bench(reference)
    print(f"op={args.op:03d} size={args.size} triton={tri_ms:.6f}ms reference={ref_ms:.6f}ms speedup={ref_ms/tri_ms:.3f}x")
    if bytes_moved is not None:
        print(f"effective_bandwidth={bytes_moved / (tri_ms * 1e-3) / 1e9:.3f} GB/s")


if __name__ == "__main__":
    main()
