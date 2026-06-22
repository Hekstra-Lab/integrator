from integrator.configs import DirichletParams, GammaParams
from integrator.data_loaders import (
    PolychromaticDataModule,
    RotationDataModule,
)
from integrator.model.distributions import (
    DirichletDistribution,
    ProfileSurrogate,
    build_gamma,
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
from integrator.model.scaling import (
    AmortizedMergingIntegrator,
    DifferenceMergingIntegrator,
    MergingWilsonLoss,
    SVAEMergingIntegrator,
)

REGISTRY = {
    "encoders": {
        "profile_encoder": ProfileEncoder,
        "intensity_encoder": IntensityEncoder,
    },
    "loss": {
        "monochromatic_wilson": MonochromaticWilsonLoss,
        "polychromatic_wilson": PolychromaticWilsonLoss,
        "merging_wilson": MergingWilsonLoss,
    },
    "integrator": {
        "hierarchical": HierarchicalIntegrator,
        "hierarchical_3enc": HierarchicalIntegrator3Enc,
        "amortized_merging": AmortizedMergingIntegrator,
        "svae_merging": SVAEMergingIntegrator,
        "difference_merging": DifferenceMergingIntegrator,
    },
    "surrogates": {
        "gamma": build_gamma,
        "dirichlet": DirichletDistribution,
        "learned_basis_profile": ProfileSurrogate,
    },
    "data_loader": {
        "rotation_data": RotationDataModule,
        "polychromatic_data": PolychromaticDataModule,
    },
    "priors": {
        "gamma": (GammaParams, ()),
        "dirichlet": (DirichletParams, ("concentration",)),
    },
}
