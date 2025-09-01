import pytest

from integrator.config.schema import Cfg
from integrator.utils import create_data_loader, create_integrator, load_config
from utils import get_configs, get_data


def test_get_configs():
    assert isinstance(get_configs(), dict)


def test_get_data():
    assert isinstance(get_data(), dict)


# Check that all configs in CONFIGS load properly
def test_load_config():
    for _, v in get_configs().items():
        config = load_config(v)
        assert isinstance(config, Cfg)


@pytest.mark.parametrize("name,val", list(get_configs().items()))
def test_each_config_validates(name, val):
    cfg = load_config(val)
    assert isinstance(cfg, Cfg), f"{name} did not validate"


@pytest.mark.parametrize("name,val", list(get_configs().items()))
def test_each_config_dataloader(name, val):
    import torch

    config = load_config(val)

    data_loader = create_data_loader(config.model_dump())
    assert hasattr(data_loader, "setup")
    assert hasattr(data_loader, "train_dataloader")

    counts, shoebox, masks, reference = next(
        iter(data_loader.train_dataloader())
    )

    assert isinstance(counts, torch.Tensor)
    assert isinstance(masks, torch.Tensor)
    assert isinstance(shoebox, torch.Tensor)
    assert isinstance(reference, torch.Tensor)


@pytest.mark.parametrize("name,val", list(get_configs().items()))
def test_each_integrator(name, val):
    import pytorch_lightning as pl

    config = load_config(val)
    integrator = create_integrator(config.dict())

    assert isinstance(integrator, pl.LightningModule)

    data_loader = create_data_loader(config.model_dump())

    counts, shoebox, masks, reference = next(
        iter(data_loader.train_dataloader())
    )
    out = integrator(counts, shoebox, masks, reference)

    assert isinstance(out, dict), (
        "Output of `Integrator.forward()` is not a dictionary"
    )


# create prediction writer
# Try to fit each configuration
# Each configuration represents a different model architecture
@pytest.mark.parametrize("name,val", list(get_configs().items()))
def test_prediction_writer(name, val):
    from pytorch_lightning.callbacks import RichProgressBar

    from integrator.callbacks import PredWriter
    from integrator.utils import (
        create_data_loader,
        create_integrator,
        create_trainer,
        load_config,
    )

    config = load_config(val)
    pred_writer = PredWriter(
        output_dir=None,
        write_interval=config.dict()["trainer"]["args"]["callbacks"][
            "pred_writer"
        ]["write_interval"],
    )

    data = create_data_loader(config.model_dump())

    integrator = create_integrator(config.dict())

    trainer = create_trainer(
        config.dict(),
        callbacks=[
            pred_writer,
            RichProgressBar(),
        ],
    )

    # Fit the model
    trainer.fit(
        integrator,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )
