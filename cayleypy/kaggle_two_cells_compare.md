# Kaggle: два ячейки — upstream `main` и форк, сравнение результатов

**Settings → Internet: On.** На Kaggle путь рабочей папки: `/kaggle/working` (если нет — используется `TMPDIR` / `/tmp`).

Внизу — **ячейка 1** и **ячейка 2**: скопируйте каждую в отдельную ячейку ноутбука по порядку.

---

## Ячейка 1 — репозиторий `cayleypy/cayleypy`, ветка `main`

```python
# --- Cell 1: upstream main ---
import json
import os
import shutil
import subprocess
import sys

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


def bfs_suite() -> dict:
    """Одинаковый набор прогонов для ячейки 1 и ячейки 2."""
    out: dict = {}
    out["expected_lrx8"] = list(load_dataset("lrx_cayley_growth")["8"])
    out["lrx5_cpu"] = list(
        CayleyGraph(PermutationGroups.lrx(5), device="cpu", num_gpus=0).bfs(max_diameter=3).layer_sizes
    )
    if torch.cuda.is_available():
        out["lrx8_g1"] = list(
            CayleyGraph(PermutationGroups.lrx(8), device="cuda", num_gpus=1).bfs().layer_sizes
        )
        if torch.cuda.device_count() >= 2:
            for bs in (100, 1000):
                out[f"lrx8_g2_bs{bs}"] = list(
                    CayleyGraph(
                        PermutationGroups.lrx(8),
                        device="cuda",
                        num_gpus=2,
                        batch_size=bs,
                    )
                    .bfs()
                    .layer_sizes
                )
            coset_def = PermutationGroups.lrx(10).with_central_state("0110110110")
            out["coset_g2"] = list(
                CayleyGraph(coset_def, device="cuda", num_gpus=2).bfs().layer_sizes
            )
            tiny = CayleyGraphDef.create([[1, 2, 3, 0]])
            out["tiny_g2"] = list(
                CayleyGraph(tiny, device="cuda", num_gpus=2).bfs().layer_sizes
            )
    return out


upstream = bfs_suite()
out_path = os.path.join(WORK, "cayley_compare_upstream.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(upstream, f, indent=2, ensure_ascii=False)

print("Cell 1 OK — upstream (main). Saved:", out_path)
for k in sorted(upstream.keys()):
    print(k, "->", upstream[k])
```

---

## Ячейка 2 — форк (по умолчанию `TryDotAtwo/cayleypy`, ветка `feature/bfs-packed-scatter`)

Перед запуском можно задать: `os.environ["CAYLEY_FORK_URL"]` (репозиторий git) и `os.environ["CAYLEY_FORK_BRANCH"]`.

```python
# --- Cell 2: fork + сравнение с сохранённым JSON из ячейки 1 ---
import json
import os
import shutil
import subprocess
import sys

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


def bfs_suite() -> dict:
    out: dict = {}
    out["expected_lrx8"] = list(load_dataset("lrx_cayley_growth")["8"])
    out["lrx5_cpu"] = list(
        CayleyGraph(PermutationGroups.lrx(5), device="cpu", num_gpus=0).bfs(max_diameter=3).layer_sizes
    )
    if torch.cuda.is_available():
        out["lrx8_g1"] = list(
            CayleyGraph(PermutationGroups.lrx(8), device="cuda", num_gpus=1).bfs().layer_sizes
        )
        if torch.cuda.device_count() >= 2:
            for bs in (100, 1000):
                out[f"lrx8_g2_bs{bs}"] = list(
                    CayleyGraph(
                        PermutationGroups.lrx(8),
                        device="cuda",
                        num_gpus=2,
                        batch_size=bs,
                    )
                    .bfs()
                    .layer_sizes
                )
            coset_def = PermutationGroups.lrx(10).with_central_state("0110110110")
            out["coset_g2"] = list(
                CayleyGraph(coset_def, device="cuda", num_gpus=2).bfs().layer_sizes
            )
            tiny = CayleyGraphDef.create([[1, 2, 3, 0]])
            out["tiny_g2"] = list(
                CayleyGraph(tiny, device="cuda", num_gpus=2).bfs().layer_sizes
            )
    return out


fork = bfs_suite()
json_path = os.path.join(WORK, "cayley_compare_upstream.json")
with open(json_path, encoding="utf-8") as f:
    upstream = json.load(f)

if set(fork.keys()) != set(upstream.keys()):
    raise AssertionError(
        "разные наборы ключей (запустите ячейку 1 в том же рантайме / на той же машине)\n"
        f"fork only: {set(fork.keys()) - set(upstream.keys())}\n"
        f"upstream only: {set(upstream.keys()) - set(fork.keys())}"
    )

for k in sorted(upstream.keys()):
    if fork[k] != upstream[k]:
        raise AssertionError(f"mismatch {k!r}:\n  fork:     {fork[k]}\n  upstream: {upstream[k]}")

print("Cell 2 OK — fork совпадает с upstream по всем ключам.")
for k in sorted(fork.keys()):
    print(k, "->", fork[k])
```

---

Если ячейка 1 не создала JSON (не запускали или другой `WORK`), ячейка 2 упадёт при открытии файла — сначала выполните ячейку 1.
