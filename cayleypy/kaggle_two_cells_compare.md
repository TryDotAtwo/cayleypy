# Kaggle: два ячейки — upstream `main` и форк

**Результаты BFS (`layer_sizes`)** должны совпадать. **Время и память** сравниваются для отчёта (на одной машине шум возможен; assert только по слоям).

**Settings → Internet: On.** Рабочая папка: `/kaggle/working` или `TMPDIR` / `/tmp`.

---

## Ячейка 1 — `cayleypy/cayleypy`, ветка `main`

```python
# --- Cell 1: upstream main ---
import json
import os
import shutil
import subprocess
import sys
import time
import tracemalloc

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "h5py", "kagglehub", "numba", "scipy"]
)

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("TMPDIR", "/tmp")
MAIN = os.path.join(WORK, "cayley_compare_main")
shutil.rmtree(MAIN, ignore_errors=True)
subprocess.check_call(
    [
        "git",
        "clone",
        "--depth",
        "1",
        "-b",
        "main",
        "https://github.com/cayleypy/cayleypy.git",
        MAIN,
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

sys.path.insert(0, MAIN)

import torch
from cayleypy import CayleyGraph
from cayleypy.cayley_graph_def import CayleyGraphDef
from cayleypy.datasets import load_dataset
from cayleypy.graphs_lib import PermutationGroups


def benchmark_suite() -> dict:
    """Словари: layer_sizes, time_sec, cuda_peak_mib (сумма по GPU), cpu_tracemalloc_peak_mib."""
    layer_sizes: dict = {}
    time_sec: dict = {}
    cuda_peak_mib: dict = {}
    cpu_tracemalloc_peak_mib: dict = {}

    # Эталон из датасета — без замера BFS
    layer_sizes["expected_lrx8"] = list(load_dataset("lrx_cayley_growth")["8"])
    time_sec["expected_lrx8"] = 0.0

    # CPU: пик кучи Python (tracemalloc) за вызов BFS
    tracemalloc.start()
    t0 = time.perf_counter()
    r_cpu = CayleyGraph(PermutationGroups.lrx(5), device="cpu", num_gpus=0).bfs(max_diameter=3)
    time_sec["lrx5_cpu"] = round(time.perf_counter() - t0, 4)
    _, peak_b = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    layer_sizes["lrx5_cpu"] = list(r_cpu.layer_sizes)
    cpu_tracemalloc_peak_mib["lrx5_cpu"] = round(peak_b / (1024 * 1024), 3)

    if not torch.cuda.is_available():
        return {
            "layer_sizes": layer_sizes,
            "time_sec": time_sec,
            "cuda_peak_mib": cuda_peak_mib,
            "cpu_tracemalloc_peak_mib": cpu_tracemalloc_peak_mib,
        }

    def run_cuda(name: str, graph: CayleyGraph):
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(d)
        t0 = time.perf_counter()
        r = graph.bfs()
        torch.cuda.synchronize()
        time_sec[name] = round(time.perf_counter() - t0, 4)
        sm = 0.0
        for d in range(torch.cuda.device_count()):
            sm += torch.cuda.max_memory_allocated(d) / (1024**2)
        cuda_peak_mib[name] = round(sm, 2)
        layer_sizes[name] = list(r.layer_sizes)

    run_cuda("lrx8_g1", CayleyGraph(PermutationGroups.lrx(8), device="cuda", num_gpus=1))

    if torch.cuda.device_count() >= 2:
        for bs in (100, 1000):
            run_cuda(
                f"lrx8_g2_bs{bs}",
                CayleyGraph(
                    PermutationGroups.lrx(8),
                    device="cuda",
                    num_gpus=2,
                    batch_size=bs,
                ),
            )
        coset_def = PermutationGroups.lrx(10).with_central_state("0110110110")
        run_cuda("coset_g2", CayleyGraph(coset_def, device="cuda", num_gpus=2))
        tiny = CayleyGraphDef.create([[1, 2, 3, 0]])
        run_cuda("tiny_g2", CayleyGraph(tiny, device="cuda", num_gpus=2))

    return {
        "layer_sizes": layer_sizes,
        "time_sec": time_sec,
        "cuda_peak_mib": cuda_peak_mib,
        "cpu_tracemalloc_peak_mib": cpu_tracemalloc_peak_mib,
    }


upstream = benchmark_suite()
out_path = os.path.join(WORK, "cayley_compare_upstream.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(upstream, f, indent=2, ensure_ascii=False)

print("Cell 1 OK — upstream (main). Saved:", out_path)
print("layer_sizes:", json.dumps(upstream["layer_sizes"], indent=2, ensure_ascii=False))
print("time_sec:", json.dumps(upstream["time_sec"], indent=2, ensure_ascii=False))
print("cuda_peak_mib:", json.dumps(upstream["cuda_peak_mib"], indent=2, ensure_ascii=False))
print("cpu_tracemalloc_peak_mib:", json.dumps(upstream["cpu_tracemalloc_peak_mib"], indent=2, ensure_ascii=False))
```

---

## Ячейка 2 — форк (по умолчанию `TryDotAtwo/cayleypy`, `feature/bfs-packed-scatter`)

Переопределение: `CAYLEY_FORK_URL`, `CAYLEY_FORK_BRANCH`.

```python
# --- Cell 2: fork + сравнение с ячейкой 1 ---
import json
import os
import shutil
import subprocess
import sys
import time
import tracemalloc

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("TMPDIR", "/tmp")
MAIN = os.path.join(WORK, "cayley_compare_main")
FORK_URL = os.environ.get("CAYLEY_FORK_URL", "https://github.com/TryDotAtwo/cayleypy.git")
FORK_BRANCH = os.environ.get("CAYLEY_FORK_BRANCH", "feature/bfs-packed-scatter")
FORK = os.path.join(WORK, "cayley_compare_fork")

for name in list(sys.modules.keys()):
    if name == "cayleypy" or name.startswith("cayleypy."):
        del sys.modules[name]
if MAIN in sys.path:
    sys.path.remove(MAIN)

shutil.rmtree(FORK, ignore_errors=True)
subprocess.check_call(
    ["git", "clone", "--depth", "1", "-b", FORK_BRANCH, FORK_URL, FORK],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

sys.path.insert(0, FORK)

import torch
from cayleypy import CayleyGraph
from cayleypy.cayley_graph_def import CayleyGraphDef
from cayleypy.datasets import load_dataset
from cayleypy.graphs_lib import PermutationGroups


def benchmark_suite() -> dict:
    layer_sizes: dict = {}
    time_sec: dict = {}
    cuda_peak_mib: dict = {}
    cpu_tracemalloc_peak_mib: dict = {}

    layer_sizes["expected_lrx8"] = list(load_dataset("lrx_cayley_growth")["8"])
    time_sec["expected_lrx8"] = 0.0

    tracemalloc.start()
    t0 = time.perf_counter()
    r_cpu = CayleyGraph(PermutationGroups.lrx(5), device="cpu", num_gpus=0).bfs(max_diameter=3)
    time_sec["lrx5_cpu"] = round(time.perf_counter() - t0, 4)
    _, peak_b = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    layer_sizes["lrx5_cpu"] = list(r_cpu.layer_sizes)
    cpu_tracemalloc_peak_mib["lrx5_cpu"] = round(peak_b / (1024 * 1024), 3)

    if not torch.cuda.is_available():
        return {
            "layer_sizes": layer_sizes,
            "time_sec": time_sec,
            "cuda_peak_mib": cuda_peak_mib,
            "cpu_tracemalloc_peak_mib": cpu_tracemalloc_peak_mib,
        }

    def run_cuda(name: str, graph: CayleyGraph):
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(d)
        t0 = time.perf_counter()
        r = graph.bfs()
        torch.cuda.synchronize()
        time_sec[name] = round(time.perf_counter() - t0, 4)
        sm = 0.0
        for d in range(torch.cuda.device_count()):
            sm += torch.cuda.max_memory_allocated(d) / (1024**2)
        cuda_peak_mib[name] = round(sm, 2)
        layer_sizes[name] = list(r.layer_sizes)

    run_cuda("lrx8_g1", CayleyGraph(PermutationGroups.lrx(8), device="cuda", num_gpus=1))

    if torch.cuda.device_count() >= 2:
        for bs in (100, 1000):
            run_cuda(
                f"lrx8_g2_bs{bs}",
                CayleyGraph(
                    PermutationGroups.lrx(8),
                    device="cuda",
                    num_gpus=2,
                    batch_size=bs,
                ),
            )
        coset_def = PermutationGroups.lrx(10).with_central_state("0110110110")
        run_cuda("coset_g2", CayleyGraph(coset_def, device="cuda", num_gpus=2))
        tiny = CayleyGraphDef.create([[1, 2, 3, 0]])
        run_cuda("tiny_g2", CayleyGraph(tiny, device="cuda", num_gpus=2))

    return {
        "layer_sizes": layer_sizes,
        "time_sec": time_sec,
        "cuda_peak_mib": cuda_peak_mib,
        "cpu_tracemalloc_peak_mib": cpu_tracemalloc_peak_mib,
    }


fork_report = benchmark_suite()
json_path = os.path.join(WORK, "cayley_compare_upstream.json")
with open(json_path, encoding="utf-8") as f:
    upstream_report = json.load(f)


def _assert_layers_match():
    uls, fls = upstream_report["layer_sizes"], fork_report["layer_sizes"]
    if set(fls.keys()) != set(uls.keys()):
        raise AssertionError(
            "разные ключи layer_sizes\n"
            f"fork only: {set(fls.keys()) - set(uls.keys())}\n"
            f"upstream only: {set(uls.keys()) - set(fls.keys())}"
        )
    for k in sorted(uls.keys()):
        if fls[k] != uls[k]:
            raise AssertionError(f"mismatch layer_sizes[{k!r}]:\n  fork:     {fls[k]}\n  upstream: {uls[k]}")


_assert_layers_match()
print("OK: итоговые layer_sizes совпадают с upstream.\n")

# Сравнение времени и памяти (информативно; не падаем из-за шума)
ut, ft = upstream_report["time_sec"], fork_report["time_sec"]
uc, fc = upstream_report["cuda_peak_mib"], fork_report["cuda_peak_mib"]
upc, fpc = upstream_report["cpu_tracemalloc_peak_mib"], fork_report["cpu_tracemalloc_peak_mib"]

print("--- Время (сек): upstream vs fork, относительная разница ---")
for k in sorted(set(ut.keys()) | set(ft.keys())):
    a, b = ut.get(k), ft.get(k)
    if a is None or b is None or k == "expected_lrx8":
        continue
    if a == 0:
        continue
    pct = 100.0 * (b - a) / a
    print(f"  {k}: upstream={a} fork={b}  ({pct:+.1f}% от upstream)")

print("\n--- CUDA пик (MiB, сумма по GPU после теста): upstream vs fork ---")
for k in sorted(set(uc.keys()) | set(fc.keys())):
    a, b = uc.get(k), fc.get(k)
    if a is None or b is None:
        continue
    pct = 100.0 * (b - a) / a if a else 0.0
    print(f"  {k}: upstream={a} fork={b}  ({pct:+.1f}% от upstream)")

print("\n--- CPU tracemalloc пик (MiB) для lrx5_cpu: upstream vs fork ---")
for k in sorted(set(upc.keys()) | set(fpc.keys())):
    print(f"  {k}: upstream={upc.get(k)} fork={fpc.get(k)}")

print("\nПолный отчёт fork (как в JSON-структуре):", json.dumps(fork_report, indent=2, ensure_ascii=False))
```

---

Сначала выполните **ячейку 1**, затем **ячейку 2** в том же сеансе (одинаковые CUDA/CPU).

**Замечания:**

- **VRAM:** после каждого CUDA-теста — сумма `torch.cuda.max_memory_allocated` по всем устройствам; перед тестом — `reset_peak_memory_stats`.
- **CPU:** для `lrx5_cpu` — пик по **tracemalloc** за вызов (не полный RSS процесса).
- **Время:** `time.perf_counter()` вокруг одного вызова `bfs()` (построение графа + BFS внутри конструктора там, где так устроено).
