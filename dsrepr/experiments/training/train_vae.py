import os
import numpy as np
import yaml
import logging
from configargparse import ArgumentParser
# from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import loggers, Trainer
import torch
import torch.nn as nn
from torchvision.transforms import RandomCrop

from dsrepr.datasets.idr0017 import IDR0017_FullImgs_ZarrDataset
from dsrepr.models.vae import ConvolutionalBetaVAE

def RandomCropTransform(np_arr, size):
    half_size = size//2
    cx = np.random.randint(half_size+1, np_arr.shape[-1]-half_size-1)
    cy = np.random.randint(half_size+1, np_arr.shape[-2]-half_size-1)
    return np_arr[...,cy-half_size:cy+half_size, cx-half_size:cx+half_size]

def train_idr0017():
    parser = ArgumentParser()
    parser.add('-c', '--config', is_config_file=True, help='config file path')

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="betaVAE_MNIST")

    parser.add_argument("--model_dir", type=str,
                        default="/home/gvisona/Desktop/runs")
    parser.add_argument("--log_dir", type=str,
                        default="/home/gvisona/Desktop/runs")
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=2)

    parser = ConvolutionalBetaVAE.add_model_specific_args(parser)

    

    hparams = parser.parse_args()

    # set seeds
    torch.manual_seed(hparams.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(hparams.seed)

    tb_logger = loggers.TensorBoardLogger(hparams.log_dir)

    with open(os.path.join(hparams.log_dir, 'vae_config.yml'), 'w') as outfile:
        yaml.dump(hparams.__dict__, outfile, default_flow_style=False)

    transform = lambda im: RandomCropTransform(im, 128)

    train_ds = IDR0017_FullImgs_ZarrDataset("/media/gvisona/GioData/idr0017/data/idr0017.zarr", training=True, transform=transform)
    val_ds = IDR0017_FullImgs_ZarrDataset("/media/gvisona/GioData/idr0017/data/idr0017.zarr", training=True, transform=transform)

    model = ConvolutionalBetaVAE(hparams, train_ds, val_ds)

    print("HELLO")
    # trainer = Trainer(gpus=hparams.gpus,
    #                   max_epochs=hparams.max_epochs,
    #                   default_save_path=hparams.model_dir,
    #                   logger=[tb_logger])
    # trainer.fit(model)

if __name__=="__main__":
    train_idr0017()