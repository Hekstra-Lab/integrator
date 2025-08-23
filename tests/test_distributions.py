from torch import nn

from integrator.model.distributions import BaseDistribution, DirichletDistribution
from integrator.registry import REGISTRY
from integrator.utils import create_data_loader, create_integrator, load_config
from utils import CONFIGS

REGISTRY

config = load_config(CONFIGS["config3d"])
dataloader = create_data_loader(config.model_dump())
integrator = create_integrator(config.model_dump())


args = config.model_dump()["components"]["qp"]["args"]


args_3d = {"dirichlet": {"dmodel": 64, "input_shape": [3, 21, 21]}}


def test_dirichletDistribution():
    # create instance of distribution
    qp = DirichletDistribution(**args)

    assert isinstance(qp, BaseDistribution)
    assert isinstance(qp, nn.Module)
