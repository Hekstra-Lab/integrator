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
from integrator.model.scaling import RefinementIntegrator, ScalingIntegrator
from integrator.model.loss import (
    MonochromaticWilsonLoss,
    PolychromaticWilsonLoss,
    RefinementLoss,
)

REGISTRY = {
    "encoders": {
        "profile_encoder": ProfileEncoder,
        "intensity_encoder": IntensityEncoder,
    },
    "loss": {
        "monochromatic_wilson": MonochromaticWilsonLoss,
        "polychromatic_wilson": PolychromaticWilsonLoss,
        "refinement": RefinementLoss,
    },
    "integrator": {
        "hierarchical": HierarchicalIntegrator,
        "hierarchical_3enc": HierarchicalIntegrator3Enc,
        "scaling": ScalingIntegrator,
        "refinement": RefinementIntegrator,
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
