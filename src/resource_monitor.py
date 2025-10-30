"""Resource monitoring for tracking CPU, memory, and GPU usage during training."""

import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

# Try to import GPU monitoring libraries
try:
    import GPUtil

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: GPUtil not installed. GPU monitoring will be disabled.")
    print("Install with: pip install gputil")


class ResourceMonitor:
    """Monitor and log CPU, memory, and GPU usage periodically."""

    def __init__(self, log_interval: int = 60, output_dir: str = "eval_results"):
        """
        Initialize the resource monitor.

        :param log_interval: Time in seconds between resource usage logs
        :param output_dir: Directory to save resource usage logs
        """
        self.log_interval = log_interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.output_file = self.output_dir / "resource_usage.json"
        self.running = False
        self.thread = None
        self.process = psutil.Process()
        self.resource_history = []

        # Load existing history if available
        if self.output_file.exists():
            with open(self.output_file, "r") as f:
                self.resource_history = json.load(f)

    def _get_cpu_usage(self):
        """Get CPU usage percentage."""
        return self.process.cpu_percent(interval=1.0)

    def _get_memory_usage(self):
        """Get memory usage in MB and percentage."""
        mem_info = self.process.memory_info()
        mem_percent = self.process.memory_percent()
        return {
            "rss_mb": mem_info.rss / (1024 * 1024),  # Resident Set Size
            "vms_mb": mem_info.vms / (1024 * 1024),  # Virtual Memory Size
            "percent": mem_percent,
        }

    def _get_gpu_usage(self):
        """Get GPU usage if available."""
        if not GPU_AVAILABLE:
            return None

        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return None

            gpu_data = []
            for gpu in gpus:
                gpu_data.append(
                    {
                        "id": gpu.id,
                        "name": gpu.name,
                        "load": gpu.load * 100,  # GPU utilization %
                        "memory_used_mb": gpu.memoryUsed,
                        "memory_total_mb": gpu.memoryTotal,
                        "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        "temperature": gpu.temperature,
                    }
                )
            return gpu_data
        except Exception as e:
            print(f"Warning: Could not get GPU info: {e}")
            return None

    def _collect_resources(self):
        """Collect current resource usage."""
        timestamp = datetime.now().isoformat()

        entry = {
            "timestamp": timestamp,
            "cpu_percent": self._get_cpu_usage(),
            "memory": self._get_memory_usage(),
            "gpu": self._get_gpu_usage(),
        }

        return entry

    def _monitor_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        print(f"Resource monitor started (logging every {self.log_interval}s)")
        while self.running:
            try:
                entry = self._collect_resources()
                self.resource_history.append(entry)

                # Save to file
                with open(self.output_file, "w") as f:
                    json.dump(self.resource_history, f, indent=2)

                # Print summary
                cpu = entry["cpu_percent"]
                mem = entry["memory"]["percent"]
                gpu_str = ""
                if entry["gpu"]:
                    gpu_loads = [g["load"] for g in entry["gpu"]]
                    gpu_str = (
                        f", GPU: {', '.join([f'{load:.1f}%' for load in gpu_loads])}"
                    )

                print(f"[Resources] CPU: {cpu:.1f}%, Memory: {mem:.1f}%{gpu_str}")

            except Exception as e:
                print(f"Error in resource monitoring: {e}")

            # Sleep for the specified interval
            time.sleep(self.log_interval)

    def start(self):
        """Start monitoring resources in a background thread."""
        if self.running:
            print("Resource monitor is already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring resources."""
        if not self.running:
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

        # Final save
        with open(self.output_file, "w") as f:
            json.dump(self.resource_history, f, indent=2)

        print(f"Resource monitor stopped. Data saved to {self.output_file}")

    def get_summary(self):
        """Get a summary of resource usage."""
        if not self.resource_history:
            return "No resource data collected"

        cpu_values = [e["cpu_percent"] for e in self.resource_history]
        mem_values = [e["memory"]["percent"] for e in self.resource_history]

        summary = {
            "cpu": {
                "mean": np.mean(cpu_values),
                "max": np.max(cpu_values),
                "min": np.min(cpu_values),
            },
            "memory": {
                "mean": np.mean(mem_values),
                "max": np.max(mem_values),
                "min": np.min(mem_values),
            },
        }

        # GPU summary if available
        gpu_entries = [e["gpu"] for e in self.resource_history if e["gpu"]]
        if gpu_entries:
            gpu_loads = [g[0]["load"] for g in gpu_entries if g]
            summary["gpu"] = {
                "mean": np.mean(gpu_loads),
                "max": np.max(gpu_loads),
                "min": np.min(gpu_loads),
            }

        return summary
