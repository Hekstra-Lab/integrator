from integrator.data_loaders import (
    ShoeboxDataModule,
    ShoeboxDataModule2D,
    SimulatedShoeboxLoader,
)
from integrator.model.distributions import (
    DirichletDistribution,
    EmpiricalProfileSurrogate,
    FixedBasisProfileSurrogate,
    FoldedNormalDistribution,
    GammaDistributionRepamA,
    GammaDistributionRepamB,
    GammaDistributionRepamC,
    GammaDistributionRepamD,
    LearnedBasisProfileSurrogate,
    LogNormalDistribution,
    PerBinProfileSurrogate,
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
    WilsonPerBinLoss,
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
        "wilson_per_bin": WilsonPerBinLoss,
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
        "log_normal": LogNormalDistribution,
        "folded_normal": FoldedNormalDistribution,
        "dirichlet": DirichletDistribution,
        "learned_basis_profile": LearnedBasisProfileSurrogate,
        "fixed_basis_profile": FixedBasisProfileSurrogate,
        "per_bin_profile": PerBinProfileSurrogate,

        # Legacy aliases
        "linear_profile_surrogate": LearnedBasisProfileSurrogate,
        "logistic_normal_surrogate": FixedBasisProfileSurrogate,
        "per_bin_logistic_normal": PerBinProfileSurrogate,
        "empirical_profile_surrogate": EmpiricalProfileSurrogate,
    },
    "data_loader": {
        "default": ShoeboxDataModule,
        "shoebox_data_module_2d": ShoeboxDataModule2D,
        "simulated_data": SimulatedShoeboxLoader,
    },
}
