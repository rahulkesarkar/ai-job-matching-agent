from types import SimpleNamespace

from app.services.device_utils import select_torch_device


def test_prefers_cuda_when_available():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: True)
        ),
    )
    assert select_torch_device(fake_torch) == "cuda"


def test_selects_mps_when_cuda_is_unavailable():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: True)
        ),
    )
    assert select_torch_device(fake_torch) == "mps"


def test_falls_back_to_cpu_when_no_accelerator_is_available():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False)
        ),
    )
    assert select_torch_device(fake_torch) == "cpu"


def test_falls_back_to_cpu_when_mps_backend_is_missing():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(),
    )
    assert select_torch_device(fake_torch) == "cpu"
