#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Дымовой тест форка с packed scatter для Kaggle (и локально).

**Kaggle Notebook**

1. Settings → Internet **On** (для pip из GitHub).

2. Ячейка 1::

    !pip install -q "git+https://github.com/TryDotAtwo/cayleypy.git@feature/bfs-packed-scatter"

3. Ячейка 2::

    !python -m cayleypy.kaggle_fork_smoke

Опционально установка и тест в одной ячейке::

    python -m cayleypy.kaggle_fork_smoke --install

(после --install при необходимости перезапустить kernel и снова запустить без --install).

Переменные окружения (необязательно): ``KAGGLE_FORK_USER``, ``KAGGLE_FORK_BRANCH`` — для ``--install``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _fork_pip_spec() -> str:
    user = os.environ.get("KAGGLE_FORK_USER", "TryDotAtwo")
    branch = os.environ.get("KAGGLE_FORK_BRANCH", "feature/bfs-packed-scatter")
    return f"git+https://github.com/{user}/cayleypy.git@{branch}"


def install_fork(quiet: bool = True) -> None:
    spec = _fork_pip_spec()
    cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd.append("-q")
    cmd.append(spec)
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Install finished. If imports failed before, restart kernel and run without --install.")


def run_tests() -> int:
    import torch

    from cayleypy import CayleyGraph
    from cayleypy.algo.bfs_distributed import BfsDistributed
    from cayleypy.cayley_graph_def import CayleyGraphDef
    from cayleypy.datasets import load_dataset
    from cayleypy.graphs_lib import PermutationGroups

    if not hasattr(BfsDistributed, "_pack_states_for_scatter"):
        print("FAIL: BfsDistributed has no _pack_states_for_scatter — wrong branch or stale install.")
        return 1
    print("OK: _pack_states_for_scatter present")

    expected_lrx8 = load_dataset("lrx_cayley_growth")["8"]

    if torch.cuda.is_available():
        g1 = CayleyGraph(PermutationGroups.lrx(8), device="cuda", num_gpus=1)
        r1 = g1.bfs()
        assert r1.layer_sizes == expected_lrx8, (r1.layer_sizes, expected_lrx8)
        print("OK: CUDA num_gpus=1 LRX(8) layer_sizes match dataset")
    else:
        g0 = CayleyGraph(PermutationGroups.lrx(5), device="cpu", num_gpus=0)
        exp5 = load_dataset("lrx_cayley_growth")["5"][:4]
        r0 = g0.bfs(max_diameter=3)
        assert r0.layer_sizes == exp5, (r0.layer_sizes, exp5)
        print("OK: CPU num_gpus=0 LRX(5) first layers (no CUDA in session)")

    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n >= 2:
        for batch_size in (100, 1000):
            g2 = CayleyGraph(
                PermutationGroups.lrx(8),
                device="cuda",
                num_gpus=2,
                batch_size=batch_size,
            )
            r2 = g2.bfs()
            assert r2.layer_sizes == expected_lrx8, (
                f"batch_size={batch_size}",
                r2.layer_sizes,
                expected_lrx8,
            )
            print(f"OK: CUDA num_gpus=2 batch_size={batch_size} LRX(8)")

        coset_def = PermutationGroups.lrx(10).with_central_state("0110110110")
        expected_coset = [1, 3, 4, 6, 11, 16, 19, 23, 31, 29, 20, 14, 10, 10, 6, 3, 3, 1]
        gc = CayleyGraph(coset_def, device="cuda", num_gpus=2)
        rc = gc.bfs()
        assert rc.layer_sizes == expected_coset, (rc.layer_sizes, expected_coset)
        print("OK: CUDA num_gpus=2 LRX coset")

        tiny = CayleyGraphDef.create([[1, 2, 3, 0]])
        gt = CayleyGraph(tiny, device="cuda", num_gpus=2)
        rt = gt.bfs()
        assert rt.layer_sizes == [1, 1, 1, 1], rt.layer_sizes
        print("OK: CUDA num_gpus=2 non-inverse-closed tiny graph")
    else:
        print("SKIP: multi-GPU checks need 2+ CUDA devices; this session has", n)

    print("\nAll runnable tests passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Kaggle smoke test for CayleyPy fork (BFS packed scatter)")
    parser.add_argument(
        "--install",
        action="store_true",
        help="pip install fork (uses KAGGLE_FORK_USER / KAGGLE_FORK_BRANCH)",
    )
    args = parser.parse_args()
    if args.install:
        install_fork()
    return run_tests()


if __name__ == "__main__":
    raise SystemExit(main())
