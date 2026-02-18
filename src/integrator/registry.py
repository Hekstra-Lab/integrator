from integrator.data_loaders import (
    ShoeboxDataModule,
    ShoeboxDataModule2D,
    SimulatedShoeboxLoader,
)
from integrator.model.distributions import (
    BivariateLogNormalSurrogate,
    DirichletDistribution,
    DirichletDistributionB,
    FoldedNormalA,
    FoldedNormalDistribution,
    GammaDistribution,
    GammaDistributionRepamA,
    GammaDistributionRepamB,
    GammaDistributionRepamC,
    GammaDistributionRepamD,
    HalfNormalDistribution,
    LogNormalA,
    LogNormalDistribution,
)
from integrator.model.encoders import (
    IntensityEncoder,
    MLPMetadataEncoder,
    ShoeboxEncoder,
)
from integrator.model.integrators import (
    IntegratorModelA,
    IntegratorModelB,
    IntegratorModelC,
)
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
        "modelc": IntegratorModelC,
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
        "dirichletB": DirichletDistributionB,
        "folded_normal_A": FoldedNormalA,
        "log_normal_A": LogNormalA,
        "bivariate_log_normal": BivariateLogNormalSurrogate,
    },
    "data_loader": {
        "default": ShoeboxDataModule,
        "shoebox_data_module_2d": ShoeboxDataModule2D,
        "simulated_data": SimulatedShoeboxLoader,
    },
}
