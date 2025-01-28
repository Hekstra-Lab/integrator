from integrator.callbacks import PredWriter
from integrator.utils import (
    load_config,
    create_integrator,
    create_integrator_from_checkpoint,
    create_data_loader,
    create_trainer,
    parse_args,
    override_config,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

if __name__ == "__main__":
    args = parse_args()

    # Load configuration file
    config = load_config(args.config)

    # override config options from command line
    override_config(args, config)

    # Create data loader
    data = create_data_loader(config)

    # Create integrator model
    integrator = create_integrator(config)

    # Create callbacks
    pred_writer = PredWriter(
        output_dir=config["trainer"]["params"]["callbacks"]["pred_writer"][
            "output_dir"
        ],
        write_interval=config["trainer"]["params"]["callbacks"]["pred_writer"][
            "write_interval"
        ],
    )

    ## create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{val_loss:.2f}",
        every_n_epochs=2,
        save_top_k=-1,
        save_last="link",
    )

    # Create trainer
    trainer = create_trainer(config, data, callbacks=[pred_writer, checkpoint_callback])

    # Fit the model
    trainer.fit(
        integrator,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )

    # create prediction integrator from last checkpoint
    pred_integrator = create_integrator_from_checkpoint(
        config,
        trainer.logger.log_dir + "/checkpoints/last.ckpt",
    )

    # Predict
    trainer.predict(
        pred_integrator,
        return_predictions=False,
        dataloaders=data.predict_dataloader(),
    )
