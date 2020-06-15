from dsrepr.utils.helpers import _log_level_string_to_int, _LOG_LEVEL_STRINGS
from dsrepr.models.conv_beta_vae import ConvolutionalBetaVAE
import os
import numpy as np
import yaml
import logging
from configargparse import ArgumentParser, YAMLConfigFileParser
import pytorch_lightning as pl
from pytorch_lightning import loggers, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import MNIST

from dotenv import load_dotenv
load_dotenv()


DATAHOME = os.getenv("DATAHOME")
TRIALDATA = os.getenv("TRIALDATA")


def train_mnist():
    parser = ArgumentParser(config_file_parser_class=YAMLConfigFileParser, default_config_files=[
                            'mnist_config.yml', "/home/gvisona/Projects/interpreting-representations/dsrepr/experiments/config/mnist_config.yml"])
    parser.add('-c', '--config', required=True,
               is_config_file=True, help='config file path')

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="betaVAE_MNIST")
    parser.add_argument("--neptune_project", type=str,
                        default="gvisona/idr0017")

    parser.add_argument("--model_dir", type=str,
                        default=DATAHOME)
    parser.add_argument("--log_dir", type=str,
                        default=DATAHOME)
    parser.add_argument("--log_level", default="INFO", type=_log_level_string_to_int, nargs='?',
                        help="Set the logging output level. {0}".format(_LOG_LEVEL_STRINGS))
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=10)

    parser = ConvolutionalBetaVAE.add_model_specific_args(parser)

    hparams = parser.parse_args()

    # set seeds
    torch.manual_seed(hparams.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(hparams.seed)

    tb_logger = loggers.TensorBoardLogger(hparams.log_dir)
    neptune_logger = loggers.NeptuneLogger(
        project_name=hparams.neptune_project,
        params=vars(hparams),
        experiment_name=hparams.experiment_name)
    logging.basicConfig(level=hparams.log_level)
    with open(os.path.join(hparams.log_dir, 'config.yml'), 'w') as outfile:
        yaml.dump(hparams.__dict__, outfile, default_flow_style=False)
    train_ds = MNIST(TRIALDATA,                                          train=True,
                     download=True, transform=torchvision.transforms.ToTensor())
    val_ds = MNIST(TRIALDATA,                                        train=False,
                   download=True, transform=torchvision.transforms.ToTensor())

    model = ConvolutionalBetaVAE(hparams, train_ds, val_ds)

    early_stopping_cb = EarlyStopping('val_loss', patience=5)
    trainer = Trainer(gpus=hparams.gpus,
                      max_epochs=hparams.max_epochs,
                      default_save_path=hparams.model_dir,
                      logger=[tb_logger, neptune_logger],
                      callbacks=[early_stopping_cb])
    trainer.fit(model)


if __name__ == "__main__":
    train_mnist()
