from integrator.data_loaders import (
    ShoeboxDataModule,
    ShoeboxDataModule2D,
    SimulatedShoeboxLoader,
)
from integrator.model.distributions import (
    DirichletDistribution,
    FoldedNormalA,
    FoldedNormalDistribution,
    GammaDistribution,
    GammaDistributionRepamA,
    GammaDistributionRepamB,
    GammaDistributionRepamC,
    GammaDistributionRepamD,
    HalfNormalDistribution,
    LogNormalDistribution,
)
from integrator.model.encoders import (
    IntensityEncoder,
    MLPMetadataEncoder,
    ShoeboxEncoder,
)
from integrator.model.integrators import IntegratorModelA, IntegratorModelB
from integrator.model.loss import Loss

REGISTRY = {
    "encoders": {
        "shoebox_encoder": ShoeboxEncoder,
        "intensity_encoder": IntensityEncoder,
        "mlp_metadata_encoder": MLPMetadataEncoder,
    },
    "loss": {
        "default": Loss,
    },
    "integrator": {
        "modela": IntegratorModelA,
        "modelb": IntegratorModelB,
    },
    "surrogates": {
        "gamma": GammaDistribution,
        "gammaA": GammaDistributionRepamA,
        "gammaB": GammaDistributionRepamB,
        "gammaC": GammaDistributionRepamC,
        "gammaD": GammaDistributionRepamD,
        "log_normal": LogNormalDistribution,
        "folded_normal": FoldedNormalDistribution,
        "half_normal": HalfNormalDistribution,
        "dirichlet": DirichletDistribution,
        "folded_normal_A": FoldedNormalA,
    },
    "data_loader": {
        "default": ShoeboxDataModule,
        "shoebox_data_module_2d": ShoeboxDataModule2D,
        "simulated_data": SimulatedShoeboxLoader,
    },
}
