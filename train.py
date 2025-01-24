import argparse
from integrator.callbacks import PredWriter
from integrator.utils import (
    load_config,
    create_integrator,
    create_data_loader,
    create_trainer,
    parse_args,
)


if __name__ == "__main__":
    args = parse_args()

    # Load configuration file
    config = load_config(args.config)

    # Override config options from command line
    if args.batch_size:
        config["data_loader"]["params"]["batch_size"] = args.batch_size
    if args.epochs:
        config["trainer"]["params"]["max_epochs"] = args.epochs

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

    # Create trainer
    trainer = create_trainer(config, data, callbacks=[pred_writer])

    # Fit the model
    trainer.fit(
        integrator,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )

    trainer.predict(
        integrator,
        return_predictions=False,
        dataloaders=data.predict_dataloader(),
    )
