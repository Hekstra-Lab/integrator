from .model.encoders import *
from .model.decoders import *
from .model.profiles import *
from .model.loss import *
from .model.integrators import *
from .model.distribution import *
from .data_loaders import *
import torch

REGISTRY = {
    "metadata_encoder": {
        "mlp_metadata_encoder": MLPMetadataEncoder,
    },
    "encoder": {
        "cnn_3d": CNNResNet2,
        "mlp_image_encoder": MLPImageEncoder,
        "shoebox_encoder": ShoeboxEncoder,
        "normfree_mlp": NormFreeNet,
        "normfree_3d": NormFreeConv3D,
    },
    "image_encoder": {
        "cnn_3d": CNNResNet2,
        "mlp_image_encoder": MLPImageEncoder,
        "shoebox_encoder": ShoeboxEncoder,
        "normfree_mlp": NormFreeNet,
        "normfree_3d": NormFreeConv3D,
        "mlp_metadata_encoder": MLPMetadataEncoder,
    },
    "profile_encoder": {
        "shoebox_encoder": ShoeboxEncoder,
    },
    "intensity_encoder": {
        "shoebox_encoder": ShoeboxEncoder,
    },
    "decoder": {
        "default_decoder": Decoder,
        "mvn_decoder": MVNDecoder,
    },
    "profile": {
        "dirichlet": DirichletProfile,
        "mvn": MVNProfile,
    },
    "loss": {
        "elbo": Loss,
        "mvn_loss": MVNLoss,
        "lrmvn_loss": LRMVNLoss,
        "loss2": Loss2,
    },
    "integrator": {
        "default_integrator": DefaultIntegrator,
        "mvn_integrator": MVNIntegrator,
        "mlp_integrator": MLPIntegrator,
        "lrmvn_integrator": LRMVNIntegrator,
        "integrator": Integrator,
        "integrator2": IntegratorFourierFeatures,
        "integrator3": IntegratorLog1p,
        "integrator4": IntegratorLog1p2,
        "integrator5": IntegratorFFLog1p,
        "integrator6": IntegratorMLP,
    },
    "q_I": {
        "gamma": GammaDistribution,
        "log_normal": LogNormalDistribution,
        "normal": NormalDistribution,
    },
    "q_bg": {
        "gamma": GammaDistribution,
        "half_normal": HalfNormalDistribution,
        "log_normal": LogNormalDistribution,
    },
    "data_loader": {
        "default": ShoeboxDataModule,
        "shoebox_data_module": ShoeboxDataModule2,
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
            "half_normal": torch.distributions.half_normal.HalfNormal,
        },
        "p_I": {
            "gamma": torch.distributions.gamma.Gamma,
            "log_normal": torch.distributions.log_normal.LogNormal,
            "normal": torch.distributions.normal.Normal,
        },
        "p_p": {
            "laplace": torch.distributions.laplace.Laplace,
        },
    },
}
# %%
