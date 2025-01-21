from integrator.utils import (
    load_config,
    create_integrator,
    create_data_loader,
    create_trainer,
)

# Load configuration file
config = load_config(
    "/Users/luis/integratorv3/integrator/src/integrator/configs/config.yaml"
)

# Create data loader
data = create_data_loader(config)

# Create integrator model
integrator = create_integrator(config)

# Create trainer
trainer = create_trainer(config, data)

# Fit the model
trainer.fit(
    integrator,
    train_dataloaders=data.train_dataloader(),
    val_dataloaders=data.val_dataloader(),
)
