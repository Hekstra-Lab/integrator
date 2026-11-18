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
    HellingerBasisProfileSurrogate,
    NanoBraggProfileSurrogate,
    PhysicalProfileSurrogate,
    ProfileSurrogate,
    RaggedLogisticNormalSurrogate,
    ShapeletProfileSurrogate,
    ControlMeshProfileSurrogate,
    MixtureGaussianProfileSurrogate,
    ShiftedBasisProfileSurrogate,
    SteerableProfileSurrogate,
    TuckerBasisProfileSurrogate,
    build_gamma,
)
from integrator.model.distributions.quasiconformal_profile import (
    QuasiconformalProfileSurrogate,
)
from integrator.model.encoders import (
    IntensityEncoder,
    SummaryStatsEncoder,
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
    SVAEMergingIntegrator,
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
        "summary_stats_encoder": SummaryStatsEncoder,
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
        "svae_merging": SVAEMergingIntegrator,
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
        "hellinger_basis_profile": HellingerBasisProfileSurrogate,
        "tucker_basis_profile": TuckerBasisProfileSurrogate,
        "control_mesh_profile": ControlMeshProfileSurrogate,
        "mixture_gaussian_profile": MixtureGaussianProfileSurrogate,
        "steerable_profile": SteerableProfileSurrogate,
        "shifted_basis_profile": ShiftedBasisProfileSurrogate,
        "geometric_basis_profile": GeometricProfileSurrogate,
        "quasiconformal_profile": QuasiconformalProfileSurrogate,
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
