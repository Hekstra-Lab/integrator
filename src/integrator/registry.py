from integrator.configs import DirichletParams, GammaParams
from integrator.data_loaders import (
    PolychromaticDataModule,
    RotationDataModule,
    ChunkedDataModule,
)
from integrator.model.distributions import (
    DirichletDistribution,
    ProfileSurrogate,
    build_gamma,
)

from integrator.model.distributions.lognormal import build_lognormal
from integrator.model.distributions.folded_normal import build_folded_normal

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
    # Normal and StudentT observation likelihoods added for MFX data
    "surrogates": {
        "gamma": build_gamma,
        "dirichlet": DirichletDistribution,
        "learned_basis_profile": ProfileSurrogate,
        "lognormal": build_lognormal,
        "foldednormal": build_folded_normal,
    },
    "data_loader": {
        "rotation_data": RotationDataModule,
        "polychromatic_data": PolychromaticDataModule,
        "chunked_rotation_data": ChunkedDataModule,
    },
    "priors": {
        "gamma": (GammaParams, ()),
        "dirichlet": (DirichletParams, ("concentration",)),
    },
}
