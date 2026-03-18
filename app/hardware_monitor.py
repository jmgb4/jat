"""Best-effort hardware stats for SSE (CPU/RAM/GPU)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from typing import Any


async def _run(cmd: list[str], timeout: float = 2.0) -> tuple[int, str]:
    def _blocking() -> tuple[int, str]:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return p.returncode, (p.stdout or "").strip()
        except Exception:
            return 1, ""

    return await asyncio.to_thread(_blocking)


async def get_hardware_stats() -> dict[str, Any]:
    stats: dict[str, Any] = {}

    # CPU/RAM via psutil if available.
    try:
        import psutil  # type: ignore

        stats["cpu_percent"] = float(psutil.cpu_percent(interval=0.0))
        vm = psutil.virtual_memory()
        stats["ram_used"] = int(vm.used)
        stats["ram_total"] = int(vm.total)
        stats["ram_percent"] = float(vm.percent)
        # Process RSS (helps detect Python-side memory growth separately from Ollama spill).
        try:
            p = psutil.Process(os.getpid())
            rss = int(p.memory_info().rss)
            stats["process_rss"] = rss
            stats["process_rss_mb"] = float(rss) / (1024 * 1024)
        except Exception:
            pass
    except Exception:
        pass

    # GPU via nvidia-smi if available.
    try:
        rc, out = await _run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=2.0,
        )
        if rc == 0 and out:
            # First GPU only.
            line = out.splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                stats["gpu_name"] = parts[0]
                stats["gpu_util"] = float(parts[1])
                stats["gpu_mem_used"] = int(float(parts[2])) * 1024 * 1024
                stats["gpu_mem_total"] = int(float(parts[3])) * 1024 * 1024
                stats["gpu_temp"] = float(parts[4])
    except Exception:
        pass

    return stats

