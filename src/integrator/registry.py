from integrator.data_loaders import (
    PolyShoeboxDataModule,
    RaggedShoeboxDataModule,
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
    GammaDistributionRepamE,
    LearnedBasisProfileSurrogate,
    LogNormalDistribution,
    RaggedLogisticNormalSurrogate,
)
from integrator.model.encoders import (
    IntensityEncoder,
    RaggedIntensityEncoder,
    RaggedShoeboxEncoder,
    ShoeboxEncoder,
)
from integrator.model.integrators import (
    HierarchicalIntegrator,
    Integrator,
    RaggedHierarchicalIntegrator,
)
from integrator.model.loss import (
    Loss,
    PolyWilsonLoss,
    WilsonLoss,
)

REGISTRY = {
    "encoders": {
        "shoebox_encoder": ShoeboxEncoder,
        "intensity_encoder": IntensityEncoder,
        "ragged_shoebox_encoder": RaggedShoeboxEncoder,
        "ragged_intensity_encoder": RaggedIntensityEncoder,
    },
    "loss": {
        "default": Loss,
        "wilson": WilsonLoss,
        "poly_wilson": PolyWilsonLoss,
    },
    "integrator": {
        "integrator": Integrator,
        "hierarchical": HierarchicalIntegrator,
        "hierarchical_ragged": RaggedHierarchicalIntegrator,
    },
    "surrogates": {
        "gammaA": GammaDistributionRepamA,
        "gammaB": GammaDistributionRepamB,
        "gammaE": GammaDistributionRepamE,
        "log_normal": LogNormalDistribution,
        "folded_normal": FoldedNormalDistribution,
        "dirichlet": DirichletDistribution,
        "learned_basis_profile": LearnedBasisProfileSurrogate,
        "fixed_basis_profile": FixedBasisProfileSurrogate,
        "ragged_learned_basis_profile": RaggedLogisticNormalSurrogate,
    },
    "data_loader": {
        "default": ShoeboxDataModule,
        "shoebox_data_module_2d": ShoeboxDataModule2D,
        "simulated_data": SimulatedShoeboxLoader,
        "ragged_data": RaggedShoeboxDataModule,
        "poly_data": PolyShoeboxDataModule,
    },
}
