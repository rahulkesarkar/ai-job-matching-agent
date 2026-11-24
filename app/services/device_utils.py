def select_torch_device(torch_module=None) -> str:
    """Select the best available PyTorch device across platforms."""
    if torch_module is None:
        import torch
        torch_module = torch

    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        return "cuda"

    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)

    if mps is not None and mps.is_available():
        return "mps"

    return "cpu"
