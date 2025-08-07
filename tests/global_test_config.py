from pathlib import Path
import pytest
import genesis as gs
import torch


@pytest.fixture(scope="session")
def init_gs():
    cuda_available = torch.cuda.is_available()
    if not gs._initialized:
        gs.init(backend=gs.gpu if cuda_available else gs.cpu, logging_level="error")
    yield


@pytest.fixture(scope="session")
def base_data_dir():
    cuda_available = torch.cuda.is_available()
    base_dir = Path(
        "/workspace/data/" if cuda_available else "/home/maksym/Work/repairs-data/"
    )
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir
