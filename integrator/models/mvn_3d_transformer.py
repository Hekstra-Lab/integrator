import torch
import torch.nn as nn

import math

import rs_distributions.distributions as rsd

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from integrator.models.integrator_mvn_transformer import BackgroundDistribution
from integrator.models.integrator_mvn_transformer import Integrator
from integrator.models.integrator_mvn_transformer import BackgroundIndicator
from integrator.models.integrator_mvn_transformer import Builder
from integrator.models.integrator_mvn_transformer import IntensityDistribution
from integrator.models.integrator_mvn_transformer import Encoder
from integrator.models.integrator_mvn_transformer import Decoder
from integrator.models.integrator_mvn_transformer import Loss
from integrator.models.integrator_mvn_transformer import Profile

from integrator.io import ShoeboxDataModule

from integrator.layers import Standardize
from integrator.layers import Linear


class IntegratorTransformer:
    def __init__(
        self,
        depth=10,
        dmodel=64,
        feature_dim=7,
        dropout=0.5,
        beta=1.0,
        mc_samples=100,
        max_size=1024,
        eps=1e-5,
        batch_size=50,
        learning_rate=0.001,
        epochs=100,
        intensity_dist=torch.distributions.gamma.Gamma,
        background_dist=torch.distributions.gamma.Gamma,
        prior_I=torch.distributions.exponential.Exponential(rate=torch.tensor(0.05)),
        prior_bg=rsd.FoldedNormal(0, 0.1),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        shoebox_file="./samples.pt",
        metadata_file="./metadata.pt",
        dead_pixel_mask_file="./masks.pt",
        subset_size=10,
        p_I_scale=0.0001,
        p_bg_scale=0.0001,
        num_components=5,
        bg_indicator=False,
        img_size=21,
        patch_size=7,
        num_hiddens=24,
        mlp_num_hiddens=48,
        num_blks=2,
        num_heads=8,
        emb_dropout=0.5,
        blk_dropout=0.1,
        lr=0.1,
    ):
        super().__init__()
        self.depth = depth
        self.dmodel = dmodel
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.beta = beta
        self.mc_samples = mc_samples
        self.max_size = max_size
        self.eps = eps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.intensity_dist = intensity_dist
        self.background_dist = background_dist
        self.prior_I = prior_I
        self.prior_bg = prior_bg
        self.device = device
        self.shoebox_file = shoebox_file
        self.metadata_file = metadata_file
        self.dead_pixel_mask_file = dead_pixel_mask_file
        self.subset_size = subset_size
        self.p_I_scale = p_I_scale
        self.p_bg_scale = p_bg_scale
        self.num_components = num_components
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_blks = (num_blks,)
        self.num_heads = (num_heads,)
        self.num_hiddens = (num_hiddens,)
        self.blk_dropout = (blk_dropout,)
        self.emb_dropout = (emb_dropout,)
        self.mlp_num_hiddens = (mlp_num_hiddens,)
        if bg_indicator:
            self.bg_indicator = BackgroundIndicator()
        else:
            self.bg_indicator = None

    def LoadData(self, val_split=0.3, test_split=0.1):
        # Initialize the DataModule
        data_module = ShoeboxDataModule(
            shoebox_data=self.shoebox_file,
            metadata=self.metadata_file,
            dead_pixel_mask=self.dead_pixel_mask_file,
            batch_size=self.batch_size,
            val_split=val_split,
            test_split=test_split,
            include_test=False,
            subset_size=self.subset_size,
            single_sample_index=None,
        )

        # Setup data module
        data_module.setup()

        # Length (number of samples) of train loader
        self.train_loader_len = len(data_module.train_dataloader())
        # self.dataset = data_module

        return data_module

    def BuildModel(
        self,
        img_size=21,
        patch_size=7,
        num_hiddens=24,
        mlp_num_hiddens=48,
        num_heads=2,
        num_blks=2,
        emb_dropout=0.1,
        blk_dropout=0.1,
        lr=0.1,
        precision="32",
    ):
        # Intensity prior distribution
        standardization = Standardize(max_counts=self.train_loader_len)

        bg_distribution_model = BackgroundDistribution(
            self.dmodel, self.background_dist
        )

        spot_profile = Profile(self.dmodel, num_components=self.num_components)

        intensity_distribution_model = IntensityDistribution(
            self.dmodel, self.intensity_dist
        )

        encoder = Encoder(
            img_size=img_size,
            patch_size=patch_size,
            num_hiddens=num_hiddens,
            mlp_num_hiddens=mlp_num_hiddens,
            num_heads=num_heads,
            num_blks=num_blks,
            emb_dropout=emb_dropout,
            blk_dropout=blk_dropout,
            lr=lr,
        )
        # Variational distribution and profile builder

        builder = Builder(
            intensity_distribution_model,
            bg_distribution_model,
            spot_profile,
            bg_indicator=self.bg_indicator,
        )

        # Decoder
        decoder = Decoder()

        # loss calculation

        loss_model = Loss(
            prior_I=self.prior_I,
            prior_bg=self.prior_bg,
            p_I_scale=self.p_I_scale,
            p_bg_scale=self.p_bg_scale,
        )

        # Number of steps to train for
        steps = 1000 * self.train_loader_len

        # Integration model
        integration_model = Integrator(
            encoder,
            builder,
            standardization,
            decoder,
            loss_model,
            total_steps=steps,
            n_cycle=4,
            lr=self.learning_rate,
            max_epochs=self.epochs,
            penalty_scale=0.0,
            use_bg_profile=self.use_bg_profile,
        )

        logger = TensorBoardLogger(
            save_dir="./integrator_logs", name="integrator_model"
        )

        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor="train_loss",
            dirpath="./integrator_logs/checkpoints/",
            filename="integrator-{epoch:02d}-{train_loss:.2f}",
            save_top_k=3,
            mode="min",
        )

        # Progress bar
        progress_bar = TQDMProgressBar(refresh_rate=1)

        # Training module
        trainer = Trainer(
            max_epochs=self.epochs,
            accelerator="gpu",  # Use "cpu" for CPU training
            devices="auto",
            num_nodes=1,
            precision=precision,  # Use 32-bit precision for CPU
            accumulate_grad_batches=1,
            check_val_every_n_epoch=1,
            callbacks=[checkpoint_callback, progress_bar],
            logger=logger,
            log_every_n_steps=10,
        )
        return trainer, integration_model