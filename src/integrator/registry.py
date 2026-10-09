from integrator.configs import DirichletParams, GammaParams
from integrator.data_loaders import (
    PolychromaticDataModule,
    RaggedShoeboxDataModule,
    RotationDataModule,
)
from integrator.model.distributions import (
    GammaDistributionRepamE,
    GammaDistributionRepamB,
    GammaDistributionRepamA,
    BackgroundPlaneSurrogate,
    DirichletDistribution,
    FixedBasisProfileSurrogate,
    GeometricProfileSurrogate,
    NanoBraggProfileSurrogate,
    PhysicalProfileSurrogate,
    ProfileSurrogate,
    RaggedLogisticNormalSurrogate,
    ShapeletProfileSurrogate,
    build_gamma,
)
from integrator.model.encoders import (
    IntensityEncoder,
    ProfileEncoder,
    RaggedIntensityEncoder,
    RaggedShoeboxEncoder,
)
from integrator.model.integrators import (
    HierarchicalIntegrator,
    HierarchicalIntegrator2Enc,
    HierarchicalIntegrator3Enc,
    HierarchicalIntegrator3EncIB,
    RaggedHierarchicalIntegrator,
    SVAEHybridIntegrator,
    SVAEIntegrator,
)
from integrator.model.scaling import (
    AmortizedMergingIntegrator,
    ConjugateIntegrator,
    ConjugateMergingIntegrator,
    DeepSetsMergingIntegrator,
    HierarchicalSVAEIntegrator,
    HierarchicalScalingIntegrator,
    MergingIntegrator,
    RefinementIntegrator,
    SVAEIntegrator as ScalingSVAEIntegrator,
    ScalingIntegrator,
    VariationalRefinementIntegrator,
)
from integrator.model.loss import (
    GlobalPriorLoss,
    AmplitudeWilsonLoss,
    MonochromaticWilsonLoss,
    PolychromaticWilsonLoss,
    RefinementLoss,
)

REGISTRY = {
    "encoders": {
        "profile_encoder": ProfileEncoder,
        "intensity_encoder": IntensityEncoder,
        "ragged_shoebox_encoder": RaggedShoeboxEncoder,
        "ragged_intensity_encoder": RaggedIntensityEncoder,
    },
    "loss": {
        "global_prior": GlobalPriorLoss,
        "monochromatic_wilson": MonochromaticWilsonLoss,
        "amplitude_wilson": AmplitudeWilsonLoss,
        "polychromatic_wilson": PolychromaticWilsonLoss,
        "refinement": RefinementLoss,
    },
    "integrator": {
        "hierarchical": HierarchicalIntegrator,
        "hierarchical_2enc": HierarchicalIntegrator2Enc,
        "hierarchical_3enc": HierarchicalIntegrator3Enc,
        "hierarchical_3enc_ib": HierarchicalIntegrator3EncIB,
        "ragged_hierarchical": RaggedHierarchicalIntegrator,
        "svae": SVAEIntegrator,
        "svae_hybrid": SVAEHybridIntegrator,
        "scaling": ScalingIntegrator,
        "merging": MergingIntegrator,
        "deepsets_merging": DeepSetsMergingIntegrator,
        "amortized_merging": AmortizedMergingIntegrator,
        "conjugate_merging": ConjugateMergingIntegrator,
        "hierarchical_scaling": HierarchicalScalingIntegrator,
        "conjugate": ConjugateIntegrator,
        # the merging branch's variant, a different model that also
        # called itself SVAEIntegrator
        "svae_scaling": ScalingSVAEIntegrator,
        "hierarchical_svae": HierarchicalSVAEIntegrator,
        "refinement": RefinementIntegrator,
        "variational_refinement": VariationalRefinementIntegrator,
    },
    "surrogates": {
        # legacy per-variant names from the merging branch; the
        # preferred spelling is `gamma` with `reparameterization`
        "gammaA": GammaDistributionRepamA,
        "gammaB": GammaDistributionRepamB,
        "gammaE": GammaDistributionRepamE,
        "gamma": build_gamma,
        "dirichlet": DirichletDistribution,
        "learned_basis_profile": ProfileSurrogate,
        "geometric_basis_profile": GeometricProfileSurrogate,
        "fixed_basis_profile": FixedBasisProfileSurrogate,
        "shapelet_profile": ShapeletProfileSurrogate,
        "physical_profile": PhysicalProfileSurrogate,
        "nanobragg_profile": NanoBraggProfileSurrogate,
        "background_plane": BackgroundPlaneSurrogate,
        "ragged_learned_basis_profile": RaggedLogisticNormalSurrogate,
    },
    "data_loader": {
        "rotation_data": RotationDataModule,
        "polychromatic_data": PolychromaticDataModule,
        "ragged_data": RaggedShoeboxDataModule,
    },
    "priors": {
        "gamma": (GammaParams, ()),
        "dirichlet": (DirichletParams, ("concentration",)),
    },
}
