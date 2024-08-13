import torch
from integrator.io import ShoeboxDataModule
from rs_distributions import distributions as rsd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar,
    DeviceStatsMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger

from integrator.layers import Standardize
from integrator.io import ShoeboxDataModule

from integrator.models.integrator_mvn_3d import Encoder
from integrator.models.integrator_mvn_3d import Profile
from integrator.models.integrator_mvn_3d import Builder
from integrator.models.integrator_mvn_3d import BackgroundDistribution
from integrator.models.integrator_mvn_3d import BackgroundIndicator
from integrator.models.integrator_mvn_3d import Decoder
from integrator.models.integrator_mvn_3d import Loss
from integrator.models.integrator_mvn_3d import IntensityDistribution
from integrator.models.integrator_mvn_3d import Integrator


class MixtureModel3DMVN:
    def __init__(
        self,
        depth=10,
        dmodel=32,
        feature_dim=7,
        dropout=None,
        beta=1.0,
        mc_samples=100,
        max_size=1024,
        eps=1e-5,
        batch_size=50,
        learning_rate=0.001,
        epochs=500,
        intensity_dist=torch.distributions.gamma.Gamma,
        background_dist=torch.distributions.gamma.Gamma,
        prior_I=torch.distributions.exponential.Exponential(rate=torch.tensor(1.0)),
        prior_bg=torch.distributions.exponential.Exponential(rate=torch.tensor(1.0)),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        shoebox_file="./samples.pt",
        metadata_file="./metadata.pt",
        dead_pixel_mask="./masks.pt",
        subset_size=10,
        p_I_scale=0.0001,
        p_bg_scale=0.0001,
        num_components=1,
        bg_indicator=False,
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
        self.dead_pixel_mask = dead_pixel_mask
        self.subset_size = subset_size
        self.p_I_scale = p_I_scale
        self.p_bg_scale = p_bg_scale
        self.num_components = num_components
        if bg_indicator:
            self.bg_indicator = BackgroundIndicator()
        else:
            self.bg_indicator = None

    def LoadData(
        self,
        subset_size=10,
        val_split=0.3,
        test_split=0.1,
        include_test=False,
        single_sample_index=None,
    ):
        # Initialize the DataModule
        data_module = ShoeboxDataModule(
            shoebox_data=self.shoebox_file,
            metadata=self.metadata_file,
            dead_pixel_mask=self.dead_pixel_mask,
            batch_size=self.batch_size,
            val_split=val_split,
            test_split=test_split,
            include_test=include_test,
            subset_size=subset_size,
            single_sample_index=single_sample_index,
        )

        # Setup data module
        data_module.setup()

        # Length (number of samples) of train loader
        self.train_loader_len = len(data_module.train_dataloader())
        # self.dataset = data_module

        return data_module

    def BuildModel(self, precision=32):
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
            self.depth, self.dmodel, self.feature_dim, dropout=self.dropout
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
            lr=self.learning_rate,
            max_epochs=self.epochs,
            penalty_scale=0.0,
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
        device_stats = DeviceStatsMonitor()

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
            callbacks=[checkpoint_callback, progress_bar, device_stats],
            logger=logger,
            log_every_n_steps=10,
        )

        return trainer, integration_model
