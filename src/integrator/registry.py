from integrator.data_loaders import (
    PolyShoeboxDataModule,
    ShoeboxDataModule,
    SimulatedShoeboxLoader,
)
from integrator.model.distributions import (
    DirichletDistribution,
    FixedBasisProfileSurrogate,
    FoldedNormalDistribution,
    GammaDistributionRepamA,
    GammaDistributionRepamB,
    GammaDistributionRepamE,
    LearnedBasisProfileSurrogate,
    LogNormalDistribution,
    PositionAwareProfileSurrogate,
)
from integrator.model.encoders import (
    IntensityEncoder,
    ShoeboxEncoder,
)
from integrator.model.integrators import (
    HierarchicalIntegrator,
    HierarchicalIntegrator3Enc,
    Integrator,
)
from integrator.model.loss import (
    Loss,
    SpectralWilsonLoss,
    WilsonLoss,
)

REGISTRY = {
    "encoders": {
        "shoebox_encoder": ShoeboxEncoder,
        "intensity_encoder": IntensityEncoder,
    },
    "loss": {
        "default": Loss,
        "wilson": WilsonLoss,
        "spectral_wilson": SpectralWilsonLoss,
    },
    "integrator": {
        "integrator": Integrator,
        "hierarchical": HierarchicalIntegrator,
        "hierarchical_3enc": HierarchicalIntegrator3Enc,
    },
    "surrogates": {
        "gammaA": GammaDistributionRepamA,
        "gammaB": GammaDistributionRepamB,
        "gammaE": GammaDistributionRepamE,
        "log_normal": LogNormalDistribution,
        "folded_normal": FoldedNormalDistribution,
        "dirichlet": DirichletDistribution,
        "learned_basis_profile": LearnedBasisProfileSurrogate,
        "position_aware_profile": PositionAwareProfileSurrogate,
        "fixed_basis_profile": FixedBasisProfileSurrogate,
    },
    "data_loader": {
        "default": ShoeboxDataModule,
        "simulated_data": SimulatedShoeboxLoader,
        "poly_data": PolyShoeboxDataModule,
    },
}
