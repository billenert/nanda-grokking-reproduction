"""Device selection utilities."""

import torch


def get_device(device_str: str = "cuda") -> torch.device:
    """Get the appropriate torch device based on config and availability.

    Args:
        device_str: Device string from config ("cuda" or "cpu").

    Returns:
        torch.device object for the selected device.

    Raises:
        ValueError: If device_str is not "cuda" or "cpu".
    """
    if device_str not in ("cuda", "cpu"):
        raise ValueError(f"Invalid device: {device_str}. Must be 'cuda' or 'cpu'.")

    if device_str == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(device_str)


def get_dtype(dtype_str: str = "float32", device: torch.device | None = None) -> torch.dtype:
    """Get the appropriate torch dtype based on config.

    Args:
        dtype_str: Dtype string from config ("float32" or "float16").
        device: The device being used (for validation).

    Returns:
        torch.dtype object.

    Raises:
        ValueError: If dtype_str is invalid or float16 requested on CPU.
    """
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float16":
        if device is not None and device.type == "cpu":
            raise ValueError("float16 is only supported on CUDA devices.")
        return torch.float16
    else:
        raise ValueError(f"Invalid dtype: {dtype_str}. Must be 'float32' or 'float16'.")
