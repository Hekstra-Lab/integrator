from integrator.callbacks import PredWriter
import tracemalloc
import os
import re
import glob
from integrator.utils import (
    load_config,
    create_integrator,
    create_integrator_from_checkpoint,
    create_data_loader,
    create_trainer,
    parse_args,
    override_config,
    clean_from_memory,
    predict_from_checkpoints,
    reflection_file_writer,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
import psutil
import torch

if __name__ == "__main__":
    # def predict_from_checkpoints(config, data, version_dir, path):
    # for ckpt in glob.glob(path):
    # epoch = re.search(r"epoch=(\d+)", ckpt).group(0)
    # epoch = epoch.replace("=", "_")
    # ckpt_dir = version_dir + "/predictions/" + epoch
    # Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # # prediction writer for current checkpoint
    # pred_writer = PredWriter(
    # output_dir=ckpt_dir,
    # write_interval=config["trainer"]["params"]["callbacks"]["pred_writer"][
    # "write_interval"
    # ],
    # )
    # print('after predwriter')
    # log_memory()

    # trainer = create_trainer(
    # config,
    # data,
    # callbacks=[
    # pred_writer,
    # ],
    # )
    # print('created_new_trainer')
    # print(f'checkpoint:{ckpt}')
    # log_memory()

    # pred_integrator = create_integrator_from_checkpoint(
    # config,
    # ckpt,
    # )
    # print('created integrator from checkpoint')
    # log_memory()

    # print('running trainer.predict')
    # trainer.predict(
    # pred_integrator,
    # return_predictions=False,
    # dataloaders=data.predict_dataloader(),
    # )

    # clean_from_memory(trainer, pred_writer, pred_writer)

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
        output_dir=None,
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
    trainer = create_trainer(
        config,
        data,
        callbacks=[
            pred_writer,
            checkpoint_callback,
        ],
    )

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

    version_dir = trainer.logger.log_dir
    path = os.path.join(version_dir, "checkpoints", "epoch*.ckpt")

    # override to stop new version dirs from being created
    config["trainer"]["params"]["logger"] = False

    # clean from memory
    clean_from_memory(pred_writer, pred_writer, pred_writer, checkpoint_callback)

    # predict from checkpoints
    predict_from_checkpoints(config, trainer, pred_integrator, data, version_dir, path)

    # write refl files
    prediction_path = version_dir + "/predictions/"
    prediction_directories = glob.glob(prediction_path + "epoch*")
    prediction_files = glob.glob(prediction_path + "epoch*/*.pt")

    reflection_file_writer(
        prediction_directories,
        prediction_files,
        config["output"]["refl_file"],
    )

    # reflection_file_writer(prediction_directories, prediction_files)
    # for ckpt in glob.glob(path):
    # epoch = re.search(r"epoch=(\d+)", ckpt).group(0)
    # epoch = epoch.replace("=", "_")
    # ckpt_dir = version_dir + "/predictions/" + epoch
    # Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # # prediction writer for current checkpoint
    # pred_writer = PredWriter(
    # output_dir=ckpt_dir,
    # write_interval=config["trainer"]["params"]["callbacks"]["pred_writer"][
    # "write_interval"
    # ],
    # )
    # print('after predwriter')
    # log_memory()
    # trainer.callbacks = [pred_writer]

# #        trainer = create_trainer(
# #            config,
# #            data,
# #            callbacks=[
# #                pred_writer,
# #            ],
# #        )
# #        print('created_new_trainer')
# #        print(f'checkpoint:{ckpt}')
# #        log_memory()


# checkpoint = torch.load(ckpt,map_location='cpu')
# pred_integrator.load_state_dict(checkpoint['state_dict'])
# pred_integrator.to(torch.device('cuda'))
# pred_integrator.eval()

# print('created integrator from checkpoint')
# log_memory()

# print('running trainer.predict')
# trainer.predict(
# pred_integrator,
# return_predictions=False,
# dataloaders=data.predict_dataloader(),
# )

# clean_from_memory(pred_writer, pred_writer, pred_writer)


# predict from checkpoints

# predict_from_checkpoints(config, data, version_dir, path)

# prediction_path = version_dir + "/predictions/"
# prediction_directories = glob.glob(prediction_path + "epoch*")
# prediction_files = glob.glob(prediction_path + "epoch*/*.pt")

# reflection_file_writer(prediction_directories, prediction_files)
