from integrator.configs import DirichletParams, GammaParams
from integrator.data_loaders import (
    PolychromaticDataModule,
    RaggedShoeboxDataModule,
    RotationDataModule,
)
from integrator.model.distributions import (
    BackgroundPlaneSurrogate,
    ControlMeshProfileSurrogate,
    DirichletDistribution,
    FixedBasisProfileSurrogate,
    GammaDistributionRepamA,
    GammaDistributionRepamB,
    GammaDistributionRepamE,
    GeometricProfileSurrogate,
    StreakBaseProfileSurrogate,
    StreakMixtureProfileSurrogate,
    StreakKMixProfileSurrogate,
    StreakKMixV2GeomPriorSurrogate,
    StreakKMixV2ProfileSurrogate,
    HellingerBasisProfileSurrogate,
    MixtureGaussianProfileSurrogate,
    NanoBraggProfileSurrogate,
    PhysicalProfileSurrogate,
    ProfileSurrogate,
    RaggedLogisticNormalSurrogate,
    ShapeletProfileSurrogate,
    ShiftedBasisProfileSurrogate,
    SteerableProfileSurrogate,
    TuckerBasisProfileSurrogate,
    build_gamma,
)
from integrator.model.distributions.learned_geom_prior import GMMGeomPriorSurrogate, LearnedBasisGeomPriorSurrogate
from integrator.model.distributions.quasiconformal_profile import (
    QuasiconformalProfileSurrogate,
)
from integrator.model.encoders import (
    IntensityEncoder,
    ProfileEncoder,
    RaggedIntensityEncoder,
    RaggedShoeboxEncoder,
    SummaryStatsEncoder,
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
from integrator.model.integrators.svae_gate_integrators import (
    SVAEBackgroundIntegrator,
    SVAECalibIntegrator,
    SVAEConjugateIntegrator,
)
from integrator.model.integrators.svae_collapsed_integrator import SVAECollapsedIntegrator
from integrator.model.integrators.svae_per_sample_integrator import SVAEPerSampleIntegrator
from integrator.model.loss import (
    AmplitudeWilsonLoss,
    GlobalPriorLoss,
    MonochromaticWilsonLoss,
    PolychromaticWilsonBgMapLoss,
    PolychromaticWilsonEBLoss,
    PolychromaticWilsonLoss,
    RefinementLoss,
)
from integrator.model.scaling import (
    AmortizedMergingIntegrator,
    ConjugateIntegrator,
    ConjugateMergingIntegrator,
    DeepSetsMergingIntegrator,
    HierarchicalScalingIntegrator,
    HierarchicalSVAEIntegrator,
    MergingIntegrator,
    RefinementIntegrator,
    ScalingIntegrator,
    SVAEMergingIntegrator,
    VariationalRefinementIntegrator,
)
from integrator.model.scaling import (
    SVAEIntegrator as ScalingSVAEIntegrator,
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
        "polychromatic_wilson_bgmap": PolychromaticWilsonBgMapLoss,
        "polychromatic_wilson_eb": PolychromaticWilsonEBLoss,
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
        # gate-construction variants; `svae` itself is untouched so every
        # run already on disk keeps the architecture it trained with
        "svae_conjugate": SVAEConjugateIntegrator,
        "svae_bg": SVAEBackgroundIntegrator,
        "svae_calib": SVAECalibIntegrator,
        "svae_per_sample": SVAEPerSampleIntegrator,
        "svae_collapsed": SVAECollapsedIntegrator,
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
        "streak_base_profile": StreakBaseProfileSurrogate,
        "streak_mixture_profile": StreakMixtureProfileSurrogate,
        "streak_kmix_profile": StreakKMixProfileSurrogate,
        "streak_kmix_v2_profile": StreakKMixV2ProfileSurrogate,
        "streak_kmix_v2_geom": StreakKMixV2GeomPriorSurrogate,
        "learned_basis_geom_prior_profile": LearnedBasisGeomPriorSurrogate,
        "gmm_geom_prior_profile": GMMGeomPriorSurrogate,
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
