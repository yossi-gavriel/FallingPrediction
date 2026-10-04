"""Workstation resources: logical CPUs, RAM, an XGBoost GPU probe, thread limits and the GPU -> CPU fallback.

Defaults never take the whole machine: ``--jobs`` defaults to ~60% of the logical cores (configurable, at least 1, capped); every worker runs
single-threaded BLAS / OpenMP (OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS = 1 and threadpoolctl) so parallel workers never
oversubscribe the CPU; XGBoost gets an explicit ``nthread``. ``--device auto`` uses a GPU only when the installed XGBoost build has CUDA AND a
small synthetic training on the GPU succeeds without falling back; otherwise CPU (recorded). A GPU error during the run switches the rest of the
run to CPU (logged) instead of stopping it.
"""

from __future__ import annotations

import ctypes
import math
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def logical_cpus() -> int:
    return max(1, int(os.cpu_count() or 1))


def memory_bytes() -> dict[str, int | None]:
    """Total / available physical memory (best effort; None when unknown)."""
    try:
        if sys.platform == "win32":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
            return {"total": int(st.ullTotalPhys), "available": int(st.ullAvailPhys)}
        page = os.sysconf("SC_PAGE_SIZE")
        total = page * os.sysconf("SC_PHYS_PAGES")
        avail = page * os.sysconf("SC_AVPHYS_PAGES") if hasattr(os, "sysconf") and "SC_AVPHYS_PAGES" in os.sysconf_names else None
        return {"total": int(total), "available": int(avail) if avail is not None else None}
    except Exception:  # noqa: BLE001 - informative only
        return {"total": None, "available": None}


def process_rss() -> int | None:
    try:
        if sys.platform == "win32":
            class PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            h = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
            ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb)  # type: ignore[attr-defined]
            return int(c.WorkingSetSize)
        status = Path("/proc/self/status")                      # Linux; elsewhere (macOS) the value is simply not reported
        if status.is_file():
            for line in status.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        return None
    except Exception:  # noqa: BLE001
        return None


def default_jobs(cfg: Any) -> int:
    rc = cfg["resources"]
    return max(1, min(int(rc["max_default_jobs"]), int(math.floor(float(rc["default_core_share"]) * logical_cpus()))))


def limit_threads() -> None:
    """Single-threaded BLAS / OpenMP in this process and every worker it starts (set before workers spawn)."""
    for v in THREAD_VARS:
        os.environ[v] = "1"
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except Exception:  # noqa: BLE001
        pass


def nvidia_smi() -> list[dict[str, str]]:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=10, check=False)
        if r.returncode != 0:
            return []
        out = []
        for line in r.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2:
                out.append({"name": parts[0], "memory": parts[1], "driver": parts[2] if len(parts) > 2 else ""})
        return out
    except Exception:  # noqa: BLE001
        return []


def xgb_gpu_probe() -> dict[str, Any]:
    """A small synthetic GPU training; any error or a silent device fallback = no usable GPU."""
    import numpy as np

    try:
        import xgboost as xgb
    except Exception as exc:  # noqa: BLE001
        return {"usable": False, "reason": f"xgboost import failed: {type(exc).__name__}"}
    info = {}
    try:
        info = dict(xgb.build_info())
    except Exception:  # noqa: BLE001
        pass
    if not info.get("USE_CUDA"):
        return {"usable": False, "reason": "the installed XGBoost build has no CUDA support (e.g. xgboost-cpu)", "xgboost": xgb.__version__}
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 10)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=2000) > 0).astype(np.float32)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            b = xgb.train({"device": "cuda", "tree_method": "hist", "objective": "binary:logistic", "verbosity": 1}, xgb.DMatrix(X, label=y), num_boost_round=5)
            b.predict(xgb.DMatrix(X))
        msgs = [str(x.message) for x in w]
        if any("changed from GPU to CPU" in m or "no visible gpu" in m.lower() or ("cuda" in m.lower() and "fail" in m.lower()) for m in msgs):
            return {"usable": False, "reason": "XGBoost fell back from GPU to CPU in the probe", "xgboost": xgb.__version__}
        return {"usable": True, "reason": "GPU probe trained and predicted on CUDA", "xgboost": xgb.__version__}
    except Exception as exc:  # noqa: BLE001
        return {"usable": False, "reason": f"GPU probe failed: {type(exc).__name__}: {str(exc)[:200]}", "xgboost": xgb.__version__}


def resolve_device(requested: str) -> dict[str, Any]:
    if requested not in ("auto", "cpu", "gpu"):
        raise ValueError("--device must be auto, cpu or gpu")
    if requested == "cpu":
        return {"requested": "cpu", "device": "cpu", "reason": "CPU requested", "gpus": nvidia_smi()}
    probe = xgb_gpu_probe()
    gpus = nvidia_smi()
    if probe["usable"]:
        return {"requested": requested, "device": "cuda", "reason": probe["reason"], "gpus": gpus}
    return {"requested": requested, "device": "cpu", "reason": f"GPU not used: {probe['reason']} -> CPU" + (" (gpu was requested)" if requested == "gpu" else ""),
            "gpus": gpus}


def environment(cfg: Any, jobs: int, device: dict[str, Any]) -> dict[str, Any]:
    import numpy
    import pandas
    import sklearn

    try:
        import xgboost

        xv = xgboost.__version__
    except Exception:  # noqa: BLE001
        xv = None
    try:
        import optuna

        ov = optuna.__version__
    except Exception:  # noqa: BLE001
        ov = None
    mem = memory_bytes()
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(), "logical_cpus": logical_cpus(),
            "jobs": int(jobs), "default_jobs": default_jobs(cfg), "ram_total_gb": round(mem["total"] / 2**30, 1) if mem["total"] else None,
            "ram_available_gb": round(mem["available"] / 2**30, 1) if mem["available"] else None, "device": device,
            "thread_env": {v: os.environ.get(v) for v in THREAD_VARS},
            "versions": {"numpy": numpy.__version__, "pandas": pandas.__version__, "scikit-learn": sklearn.__version__, "xgboost": xv, "optuna": ov,
                         "falls_ml": __import__("falls_ml").__version__}}
