from integrator.data_loaders import (
    PolychromaticDataModule,
    RotationDataModule,
)
from integrator.model.distributions import (
    DirichletDistribution,
    FixedBasisProfileSurrogate,
    GammaDistributionRepamA,
    GammaDistributionRepamB,
    GammaDistributionRepamE,
    LearnedBasisProfileSurrogate,
    PositionAwareProfileSurrogate,
)
from integrator.model.encoders import (
    IntensityEncoder,
    ProfileEncoder,
)
from integrator.model.integrators import (
    HierarchicalIntegrator,
    HierarchicalIntegrator3Enc,
)
from integrator.model.loss import (
    MonochromaticWilsonLoss,
    PolychromaticWilsonLoss,
)

REGISTRY = {
    "encoders": {
        "profile_encoder": ProfileEncoder,
        "intensity_encoder": IntensityEncoder,
    },
    "loss": {
        "monochromatic_wilson": MonochromaticWilsonLoss,
        "polychromatic_wilson": PolychromaticWilsonLoss,
    },
    "integrator": {
        "hierarchical": HierarchicalIntegrator,
        "hierarchical_3enc": HierarchicalIntegrator3Enc,
    },
    "surrogates": {
        "gammaA": GammaDistributionRepamA,
        "gammaB": GammaDistributionRepamB,
        "gammaE": GammaDistributionRepamE,
        "dirichlet": DirichletDistribution,
        "learned_basis_profile": LearnedBasisProfileSurrogate,
        "position_aware_profile": PositionAwareProfileSurrogate,
        "fixed_basis_profile": FixedBasisProfileSurrogate,
    },
    "data_loader": {
        "rotation_data": RotationDataModule,
        "polychromatic_data": PolychromaticDataModule,
    },
}
