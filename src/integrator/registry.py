from integrator.data_loaders import (
    RaggedShoeboxDataModule,
    ShoeboxDataModule,
    ShoeboxDataModule2D,
    SimulatedShoeboxLoader,
)
from integrator.model.distributions import (
    ConvProfileSurrogate,
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
    RaggedLogisticNormalSurrogate,
)
from integrator.model.encoders import (
    IntensityEncoder,
    MLPMetadataEncoder,
    RaggedIntensityEncoder,
    RaggedShoeboxEncoder,
    ShoeboxEncoder,
)
from integrator.model.integrators import (
    HierarchicalIntegratorA,
    HierarchicalIntegratorB,
    HierarchicalIntegratorC,
    IntegratorModelA,
    IntegratorModelB,
    IntegratorModelC,
    RaggedHierarchicalIntegratorB,
    RaggedHierarchicalIntegratorC,
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
        # Ragged (variable-shoebox-size) versions:
        "ragged_shoebox_encoder": RaggedShoeboxEncoder,
        "ragged_intensity_encoder": RaggedIntensityEncoder,
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
        # Ragged-input variants:
        "hierarchicalB_ragged": RaggedHierarchicalIntegratorB,
        "hierarchicalC_ragged": RaggedHierarchicalIntegratorC,
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
        "conv_profile": ConvProfileSurrogate,
        # Ragged profile surrogate with on-the-fly learned basis:
        "ragged_learned_basis_profile": RaggedLogisticNormalSurrogate,
    },
    "data_loader": {
        "default": ShoeboxDataModule,
        "shoebox_data_module_2d": ShoeboxDataModule2D,
        "simulated_data": SimulatedShoeboxLoader,
        # Ragged data module that consumes mksbox-dials chunks:
        "ragged_data": RaggedShoeboxDataModule,
    },
}
