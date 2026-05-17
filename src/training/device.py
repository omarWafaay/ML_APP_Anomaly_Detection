from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available. Install a CUDA-enabled PyTorch build "
                "or use --device cpu."
            )
        return torch.device(requested)
    raise ValueError("device must be one of: auto, cpu, cuda, cuda:<index>")


def device_report(requested: str = "auto") -> dict[str, object]:
    selected = resolve_device(requested)
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "selected_device": str(selected),
    }
