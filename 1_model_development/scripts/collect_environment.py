#!/usr/bin/env python3
"""Collect environment information for benchmarking context."""

import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def get_git_commit():
    """Get git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()[:8]
    except Exception:
        return None


def get_package_version(package_name: str) -> str | None:
    """Get package version."""
    try:
        mod = __import__(package_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def get_cpu_info():
    """Get CPU information."""
    try:
        import psutil
        return {
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_percent": psutil.cpu_percent(interval=1),
        }
    except Exception:
        return {}


def get_ram_info():
    """Get RAM information."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024 ** 3), 2),
            "available_gb": round(mem.available / (1024 ** 3), 2),
            "percent": mem.percent,
        }
    except Exception:
        return {}


def main():
    output_dir = Path("benchmarks/environment")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    env_info = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "platform": {
            "system": platform.system(),
            "processor": platform.processor(),
            "architecture": platform.architecture()[0],
            "python_version": platform.python_version(),
        },
        "packages": {
            "numpy": get_package_version("numpy"),
            "opencv": get_package_version("cv2"),
            "onnx": get_package_version("onnx"),
            "onnxruntime": get_package_version("onnxruntime"),
            "torch": get_package_version("torch"),
            "torchvision": get_package_version("torchvision"),
            "ultralytics": get_package_version("ultralytics"),
            "psutil": get_package_version("psutil"),
            "pyyaml": get_package_version("yaml"),
        },
        "system": {
            "cpu": get_cpu_info(),
            "ram": get_ram_info(),
        },
        "git": {
            "commit": get_git_commit(),
        },
    }
    
    # Write JSON
    output_path = output_dir / "host.json"
    with open(output_path, "w") as f:
        json.dump(env_info, f, indent=2)
    
    print(f"✓ Environment info saved to {output_path}")
    
    # Print summary
    print(f"\nEnvironment Summary:")
    print(f"  Platform: {env_info['platform']['system']} ({env_info['platform']['architecture']})")
    print(f"  Python: {env_info['platform']['python_version']}")
    print(f"  CPU: {env_info['system']['cpu'].get('cpu_count_logical', 'N/A')} logical cores")
    print(f"  RAM: {env_info['system']['ram'].get('total_gb', 'N/A')} GB")
    print(f"  Git: {env_info['git']['commit']}")


if __name__ == "__main__":
    main()
