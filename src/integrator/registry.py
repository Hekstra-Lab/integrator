from integrator.data_loaders import (
    ShoeboxDataModule,
    ShoeboxDataModule2D,
    SimulatedShoeboxLoader,
)
from integrator.model.distributions import (
    DirichletDistribution,
    FixedBasisProfileSurrogate,
    FoldedNormalDistribution,
    GammaDistributionRepamA,
    GammaDistributionRepamB,
    GammaDistributionRepamC,
    GammaDistributionRepamD,
    GammaDistributionRepamE,
    LearnedBasisProfileSurrogate,
    LogNormalDistribution,
)
from integrator.model.encoders import (
    IntensityEncoder,
    MLPMetadataEncoder,
    ShoeboxEncoder,
)
from integrator.model.integrators import (
    HierarchicalIntegratorA,
    HierarchicalIntegratorB,
    HierarchicalIntegratorC,
    IntegratorModelA,
    IntegratorModelB,
    IntegratorModelC,
)
from integrator.model.loss import (
    Loss,
    PerBinLoss,
    WilsonLoss,
)

REGISTRY = {
    "encoders": {
        "shoebox_encoder": ShoeboxEncoder,
        "intensity_encoder": IntensityEncoder,
        "mlp_metadata_encoder": MLPMetadataEncoder,
    },
    "loss": {
        "default": Loss,
        "per_bin": PerBinLoss,
        "wilson": WilsonLoss,
    },
    "integrator": {
        "modela": IntegratorModelA,
        "modelb": IntegratorModelB,
        "modelc": IntegratorModelC,
        "hierarchicalA": HierarchicalIntegratorA,
        "hierarchicalB": HierarchicalIntegratorB,
        "hierarchicalC": HierarchicalIntegratorC,
    },
    "surrogates": {
        "gammaA": GammaDistributionRepamA,
        "gammaB": GammaDistributionRepamB,
        "gammaC": GammaDistributionRepamC,
        "gammaD": GammaDistributionRepamD,
        "gammaE": GammaDistributionRepamE,
        "log_normal": LogNormalDistribution,
        "folded_normal": FoldedNormalDistribution,
        "dirichlet": DirichletDistribution,
        "learned_basis_profile": LearnedBasisProfileSurrogate,
        "fixed_basis_profile": FixedBasisProfileSurrogate,
    },
    "data_loader": {
        "default": ShoeboxDataModule,
        "shoebox_data_module_2d": ShoeboxDataModule2D,
        "simulated_data": SimulatedShoeboxLoader,
    },
}
