import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import sys
from itertools import cycle
from dsrepr.models.architectures import ConvolutionalEncoder, TransposeConvolutionalDecoder
from dsrepr.utils.functions import gaussian_reparameterization, calc_output_size_convnet, calc_output_size_transpose_convnet
from dsrepr.utils.helpers import ArgRange
from dsrepr.utils.viz import random_compare

class ConvolutionalBetaVAE(LightningModule):
    def __init__(self, hparams, train_dataset, val_dataset, test_dataset=None):
        super().__init__()
        self.hparams = hparams
        self.beta = hparams.beta
        self.crop_output = hparams.crop_output
        self.warmup_epochs = hparams.warmup_epochs
        self.warmup_factor = 0 if self.warmup_epochs>0 else 1

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.conv_encoder = ConvolutionalEncoder(hparams)

        split_input_dim = self.conv_encoder.output_dim

        self.fc_mu = nn.Linear(split_input_dim, hparams.latent_dim)
        self.fc_logvar = nn.Linear(split_input_dim, hparams.latent_dim)

        self.tconv_decoder = TransposeConvolutionalDecoder(hparams)


        if hparams.reconstruction_loss == "MSE":
            self.reconstruction_loss = F.mse_loss
        elif hparams.reconstruction_loss == "L1":
            self.reconstruction_loss = F.l1_loss
        elif hparams.reconstruction_loss == "Huber":
            self.reconstruction_loss = nn.SmoothL1Loss()

    def encode(self, x):
        # x = self.encoder_conv(x)
        # x = torch.flatten(x, start_dim=1)
        # if self.encoder_fc is not None:
        #     x = self.encoder_fc(x)
        x = self.conv_encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        out = self.tconv_decoder(z)
        return out

    def loss_function(self, reconstr_x, x, mu, log_var, kld_weight=1.0, beta=1.0):
        recon_loss = self.reconstruction_loss(reconstr_x, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 +
                                               log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recon_loss + beta * kld_weight * kld_loss
        return loss, {'total_loss': loss, 'recon_loss': recon_loss, 'kld_loss': beta*kld_weight*kld_loss}

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = gaussian_reparameterization(mu, logvar)
        out = self.decode(z)
        if self.crop_output and out.shape!=x.shape:
            x_h, x_w = x.shape[2:]
            out_h, out_w = out.shape[2:]
            if out_h<x_h or out_w<x_w:
                raise Exception
            off_h = (out_h-x_h)//2
            off_w = (out_w-x_w)//2
            out = out[..., off_h:off_h+x_h, off_w:off_w+x_w]
        return [out, mu, logvar]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate,
                                     weight_decay=self.hparams.weight_decay)
        if self.hparams.lr_scheduler is not None and self.hparams.lr_scheduler!= "None":
            if self.hparams.lr_scheduler == "Exponential":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                                   gamma=self.hparams.scheduler_gamma)
            elif self.hparams.lr_scheduler == "Step":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.scheduler_step_size,
                                                            gamma=self.hparams.scheduler_gamma)
            return optimizer, scheduler
        return optimizer

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            shuffle=True,
                            drop_last=True, pin_memory=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset,
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            shuffle=False,
                            drop_last=True, pin_memory=True)
        return loader

    def training_step(self, batch, batch_idx):
        input_tensor = batch[0]
        output_tensor, mu, logvar = self(input_tensor)
        train_loss, logs = self.loss_function(output_tensor, input_tensor, mu, logvar,
                                                    kld_weight=self.hparams.batch_size /
                                                    len(self.train_dataset),
                                                    beta=self.beta*self.warmup_factor)

        return {"loss": train_loss, "log": logs}

    def training_epoch_end(self, training_outputs):
        if self.warmup_epochs>0:
            self.warmup_factor = min(self.current_epoch/self.warmup_epochs, 1)
        return {}


    def validation_step(self, batch, batch_idx):
        input_tensor = batch[0]
        output_tensor, mu, logvar = self(input_tensor)
        val_loss, logs = self.loss_function(output_tensor, input_tensor, mu, logvar,
                                                  kld_weight=self.hparams.batch_size/len(self.val_dataset),
                                                  beta=self.beta)
        outputs = {"loss": val_loss}
        outputs.update(logs)

        # # log images once per epoch
        neptune_logger = None
        if isinstance(self.logger, pl.loggers.neptune.NeptuneLogger):
            neptune_logger = self.logger
        if isinstance(self.logger, pl.loggers.base.LoggerCollection):
            for logger in self.logger:
                if isinstance(logger, pl.loggers.neptune.NeptuneLogger):
                    neptune_logger = logger
        if batch_idx == 0 and neptune_logger is not None:
            figures = random_compare(
                input_tensor.cpu(), output_tensor.detach().cpu(), comparisons_per_img=3, n_images=2)
            for f in figures:
                neptune_logger.experiment.log_image(
                    "validation_reconstruction", f)
                plt.close(f)
        return outputs

    def validation_epoch_end(self, outputs):
        if outputs:
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
            tensorboard_logs = {'val_loss': avg_loss}
            return {'val_loss': avg_loss, 'log': tensorboard_logs}
        return {}

    @staticmethod
    def add_model_specific_args(parser):
        # parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--reconstruction_loss', type=str, choices=['MSE', 'L1', 'Huber'],  default="MSE")
        parser.add_argument('--beta', type=float,  default=1.0)
        parser.add_argument('--warmup_epochs', type=int, default=0)

        parser.add_argument('--input_size', type=int)
        parser.add_argument('--input_channels', type=int, default=3)
        parser.add_argument('--crop_output', default=True,
                            type=lambda x: (str(x).lower() == 'true'))

        parser.add_argument('--latent_dim', type=int)

        parser.add_argument('--conv_batch_norm', default=False,
                            type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument('--fc_batch_norm', default=False,
                            type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument('--fc_dropout_p', type=float,
                            default=0.0, choices=[ArgRange(0.0, 1.0)])

        parser.add_argument('--conv_encoder_feature_maps',
                            nargs='*', type=int, default=[8, 16, 32, 64])
        parser.add_argument('--conv_encoder_kernel_sizes',
                            nargs='*', type=int, default=[4, 4, 4, 4])
        parser.add_argument('--conv_encoder_paddings',
                            nargs='*', type=int, default=[1, 1, 1, 1])
        parser.add_argument('--conv_encoder_strides',
                            nargs='*', type=int, default=[2, 2, 2, 2])
        parser.add_argument('--conv_encoder_dilations',
                            nargs='*', type=int, default=[1, 1, 1, 1])
        parser.add_argument('--conv_encoder_activations',
                            nargs='*', type=str, default="ReLU")

        parser.add_argument('--fc_encoder_hidden_layers',
                            nargs='*', type=int, default=[64])
        parser.add_argument('--fc_encoder_activations',
                            nargs='*', type=str, default="ReLU")

        parser.add_argument('--conv_decoder_feature_maps',
                            nargs='*', type=int, default=[64, 32, 8])
        parser.add_argument('--conv_decoder_kernel_sizes',
                            nargs='*', type=int, default=[4, 4, 4])
        parser.add_argument('--conv_decoder_paddings',
                            nargs='*', type=int, default=[1, 1, 1])
        parser.add_argument('--conv_decoder_strides',
                            nargs='*', type=int, default=[2, 2, 2])
        parser.add_argument('--conv_decoder_dilations',
                            nargs='*', type=int, default=[1, 1, 1])
        parser.add_argument('--conv_decoder_activations',
                            nargs='*', type=str, default="ReLU")
        parser.add_argument('--conv_decoder_final_activation',
                            default=False, type=lambda x: (str(x).lower() == 'true'))

        parser.add_argument('--fc_decoder_hidden_layers',
                            nargs='*', type=int, default=[32, 64, 128, 255])
        parser.add_argument('--fc_decoder_activations',
                            nargs='*', type=str, default="ReLU")

        parser.add_argument('--output_activation', default=False,
                            type=lambda x: (str(x).lower() == 'true'))

        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--learning_rate', type=float,
                            default=0.02, choices=[ArgRange(0.0, None)])
        parser.add_argument('--weight_decay', type=float,
                            default=0.002, choices=[ArgRange(0.0, None)])

        parser.add_argument('--lr_scheduler', default=None,
                            choices=[None, "None", "Step", "Exponential"])
        parser.add_argument('--scheduler_gamma', type=float,
                            choices=[ArgRange(0.0, 1.0)])
        parser.add_argument('--scheduler_step_size', type=int)

        return parser

