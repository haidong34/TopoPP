"""Managed GPU2 continuous BF16 GEMM competitor for the A800 experiment."""

import argparse
import os
import time

from profile_common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--ready-file", required=True)
    args = parser.parse_args()
    if args.gpu < 0 or min(args.m, args.n, args.k) < 1:
        parser.error("Nonnegative GPU and positive GEMM dimensions required")
    import torch

    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    torch.manual_seed(1234)
    with torch.inference_mode():
        left = torch.randn(args.m, args.k, device=device, dtype=torch.bfloat16)
        right = torch.randn(args.k, args.n, device=device, dtype=torch.bfloat16)
        output = torch.empty(args.m, args.n, device=device, dtype=torch.bfloat16)
        for _ in range(5):
            torch.mm(left, right, out=output)
        torch.cuda.synchronize()
        info = {
            "pid": os.getpid(),
            "gpu_id": args.gpu,
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "dtype": "bfloat16",
            "sleep_ms": 0,
            "started_at": time.time(),
            "iterations": 0,
        }
        write_json(args.ready_file, info)
        print("Contention ready:", info, flush=True)
        iterations, last_heartbeat = 0, time.monotonic()
        while True:
            # Bound the host launch queue while keeping the GPU continuously supplied.
            for _ in range(100):
                torch.mm(left, right, out=output)
            iterations += 100
            torch.cuda.synchronize()
            if time.monotonic() - last_heartbeat > 5:
                write_json(
                    args.ready_file,
                    {**info, "iterations": iterations, "heartbeat_at": time.time()},
                )
                last_heartbeat = time.monotonic()


if __name__ == "__main__":
    main()
