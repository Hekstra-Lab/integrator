from integrator.configs import DirichletParams, GammaParams
from integrator.data_loaders import (
    PolychromaticDataModule,
    RotationDataModule,
)
from integrator.model.distributions import (
    BackgroundPlaneSurrogate,
    DirichletDistribution,
    GeometricProfileSurrogate,
    NanoBraggProfileSurrogate,
    ProfileSurrogate,
    build_gamma,
)
from integrator.model.encoders import (
    IntensityEncoder,
    ProfileEncoder,
)
from integrator.model.integrators import (
    HierarchicalIntegrator,
    HierarchicalIntegrator2Enc,
    HierarchicalIntegrator3Enc,
    HierarchicalIntegrator3EncIB,
    SVAEHybridIntegrator,
    SVAEIntegrator,
)
from integrator.model.loss import (
    GlobalPriorLoss,
    MonochromaticWilsonLoss,
    PolychromaticWilsonLoss,
)

REGISTRY = {
    "encoders": {
        "profile_encoder": ProfileEncoder,
        "intensity_encoder": IntensityEncoder,
    },
    "loss": {
        "global_prior": GlobalPriorLoss,
        "monochromatic_wilson": MonochromaticWilsonLoss,
        "polychromatic_wilson": PolychromaticWilsonLoss,
    },
    "integrator": {
        "hierarchical": HierarchicalIntegrator,
        "hierarchical_2enc": HierarchicalIntegrator2Enc,
        "hierarchical_3enc": HierarchicalIntegrator3Enc,
        "hierarchical_3enc_ib": HierarchicalIntegrator3EncIB,
        "svae": SVAEIntegrator,
        "svae_hybrid": SVAEHybridIntegrator,
    },
    "surrogates": {
        "gamma": build_gamma,
        "dirichlet": DirichletDistribution,
        "learned_basis_profile": ProfileSurrogate,
        "geometric_basis_profile": GeometricProfileSurrogate,
        "nanobragg_profile": NanoBraggProfileSurrogate,
        "background_plane": BackgroundPlaneSurrogate,
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
