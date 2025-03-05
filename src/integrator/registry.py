from .model.encoders import (
    CNNResNet,
    MLPMetadataEncoder,
    CNNResNet2,
    DevEncoder,
    CNN_3d,
    MLPImageEncoder,
    UNetDirichletConcentration,
)
from .model.decoders import Decoder, MVNDecoder, BernoulliDecoder
from .model.profiles import (
    SignalAwareProfile,
    SignalAwareMVNProfile,
    DirichletProfile,
    BetaProfile,
    MVNProfile,
)
from .model.loss import Loss, MVNLoss, BernoulliLoss
from .model.integrators import (
    DefaultIntegrator,
    BernoulliIntegrator,
    DevIntegrator,
    MVNIntegrator,
    UNetIntegrator,
)
from .model.distribution import (
    GammaDistribution,
    LogNormalDistribution,
    RelaxedBernoulliDistribution,
)
from .data_loaders import ShoeboxDataModule
import torch

REGISTRY = {
    "metadata_encoder": {
        "mlp_metadata_encoder": MLPMetadataEncoder,  # for metadata
    },
    "image_encoder": {
        # "cnn_3d": CNNResNet,  # done
        "mlp_image_encoder": MLPImageEncoder,  # done
        "3d_cnn": CNN_3d,  # shoebox encoder
        "unet": UNetDirichletConcentration,
        "dev_encoder": DevEncoder,
    },
    "decoder": {
        "default_decoder": Decoder,
        "mvn_decoder": MVNDecoder,
        "bernoulli_decoder": BernoulliDecoder,
    },
    "profile": {
        "dirichlet": DirichletProfile,
        "beta": BetaProfile,
        "mvn": MVNProfile,
        "signal_aware_dirichlet": SignalAwareProfile,
        "signal_aware_mvn": SignalAwareMVNProfile,
    },
    "loss": {
        "elbo": Loss,
        "mvn_loss": MVNLoss,
        "bernoulli_loss": BernoulliLoss,
    },
    "integrator": {
        "default_integrator": DefaultIntegrator,
        "bernoulli_integrator": BernoulliIntegrator,
        "test_integrator": DevIntegrator,
        "mvn_integrator": MVNIntegrator,
        "unet_integrator": UNetIntegrator,
    },
    "q_I": {
        "gamma": GammaDistribution,
        "log_normal": LogNormalDistribution,
    },
    "q_bg": {
        "gamma": GammaDistribution,
    },
    "q_z": {
        "bernoulli": RelaxedBernoulliDistribution,
    },
    "data_loader": {
        "default": ShoeboxDataModule,
    },
}

ARGUMENT_RESOLVER = {
    "trainer": {
        "accelerator": {
            "auto": lambda: "gpu" if torch.cuda.is_available() else "cpu",
            "gpu": "gpu",
            "cpu": "cpu",
        },
    },
    "loss": {
        "p_bg": {
            "gamma": torch.distributions.gamma.Gamma,
        },
        "p_I": {
            "gamma": torch.distributions.gamma.Gamma,
        },
        "p_p": {
            "beta": torch.distributions.beta.Beta,
            "laplace": torch.distributions.laplace.Laplace,
        },
    },
}
