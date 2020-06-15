import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule
from argparse import ArgumentParser
import numpy as np
import sys
from itertools import cycle
from dsrepr.models.architectures import FullyConnectedModule, ConvolutionalModule, TransposeConvolutionalModule
from dsrepr.models.losses import VAE_loss, BetaVAE_Loss
from dsrepr.utils.functions import gaussian_reparameterization, calc_output_size_convnet, calc_output_size_transpose_convnet


class ArgRange(object):
    def __init__(self, start=None, end=None):
        self.start = start
        self.end = end

    def __eq__(self, other):
        if self.start is None:
            return other <= self.end
        if self.end is None:
            return self.start <= other
        return self.start <= other <= self.end


class BetaVAE(LightningModule):
    def __init__(self, hparams, train_dataset, val_dataset, test_dataset=None):
        super().__init__()
        self.hparams = hparams
        self.beta = hparams.beta

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        if isinstance(hparams.encoder_activations, str):
            encoder_activations = [
                getattr(nn, hparams.encoder_activations)]*len(hparams.encoder_hidden_layers)
        elif isinstance(hparams.encoder_activations, (list, tuple)):
            encoder_activations = [getattr(nn, act)
                                   for act in hparams.encoder_activations]

        self.encoder = FullyConnectedModule(input_size=hparams.input_size,
                                            hidden_layers=hparams.encoder_hidden_layers,
                                            dropout_p=hparams.dropout_p,
                                            activations=encoder_activations,
                                            batch_norm=hparams.batch_norm,
                                            final_activation=True)

        self.fc_mu = nn.Linear(
            hparams.encoder_hidden_layers[-1], hparams.latent_dim)
        self.fc_logvar = nn.Linear(
            hparams.encoder_hidden_layers[-1], hparams.latent_dim)

        if isinstance(hparams.decoder_activations, str):
            decoder_activations = [
                getattr(nn, hparams.decoder_activations)]*len(hparams.decoder_hidden_layers)
        elif isinstance(hparams.decoder_activations, (list, tuple)):
            decoder_activations = [getattr(nn, act)
                                   for act in hparams.decoder_activations]
        self.decoder = FullyConnectedModule(input_size=hparams.latent_dim,
                                            hidden_layers=hparams.decoder_hidden_layers,
                                            dropout_p=hparams.dropout_p,
                                            activations=decoder_activations,
                                            batch_norm=hparams.batch_norm,
                                            final_activation=hparams.output_activation)
        if hparams.reconstruction_loss == "MSE":
            self.reconstruction_loss = F.mse_loss
        elif hparams.reconstruction_loss == "L1":
            self.reconstruction_loss = F.l1_loss
        elif hparams.reconstruction_loss == "Huber":
            self.reconstruction_loss = nn.SmoothL1Loss()

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def loss_function(self, reconstr_x, x, mu, log_var, kld_weight=1.0, beta=1.0):
        recon_loss = self.reconstruction_loss(reconstr_x, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 +
                                               log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recon_loss + beta * kld_weight * kld_loss
        return loss, {'total_loss': loss, 'recon_loss': recon_loss, 'kld_loss': beta*kld_weight*kld_loss}

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = gaussian_reparameterization(mu, logvar)
        return [self.decoder(z), mu, logvar]

    # def train_dataloader(self):
    #     loader = DataLoader(self.train_ds, 
    #                         batch_size=self.hparams.batch_size, 
    #                         num_workers=self.hparams.num_workers, 
    #                         shuffle=True,
    #                         drop_last=True, pin_memory=True)
    #     return loader

    # def val_dataloader(self):
    #     loader = DataLoader(self.val_ds, 
    #                         batch_size=self.hparams.batch_size, 
    #                         num_workers=self.hparams.num_workers, 
    #                         shuffle=False,
    #                         drop_last=True, pin_memory=True)
    #     return loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate,
                                     weight_decay=self.hparams.weight_decay)
        if self.hparams.lr_scheduler is not None:
            if self.hparams.lr_scheduler == "Exponential":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                                   gamma=self.hparams.scheduler_gamma)
            elif self.hparams.lr_scheduler == "Step":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.scheduler_step_size,
                                                            gamma=self.hparams.scheduler_gamma)
            return optimizer, scheduler
        return optimizer

    def training_step(self, batch, batch_idx):
        input_tensor = batch[0]
        output_tensor, mu, logvar = self(input_tensor)
        train_loss, logs = self.model.loss_function(output_tensor, input_tensor, mu, logvar,
                                                    kld_weight=self.hparams.batch_size /
                                                    len(self.train_dataset),
                                                    beta=self.beta)

        return {"loss": train_loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        input_tensor = batch[0]
        output_tensor, mu, logvar = self(input_tensor)
        val_loss, logs = self.model.loss_function(output_tensor, input_tensor, mu, logvar,
                                                  kld_weight=self.hparams.batch_size/len(self.val_dataset))
        outputs = {"loss": val_loss}
        outputs.update(logs)

        # # log images once per epoch
        # if batch_idx == 0:
        #     figures = random_compare(
        #         real_img.cpu(), results[0].detach().cpu(), comparisons_per_img=3, n_images=2)
        #     for f in figures:
        #         self.logger[1].experiment.log_image(
        #             "validation_reconstruction", f)
        #         plt.close(f)
        return outputs

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--reconstruction_loss', type=str,
                            required=True, choices=['MSE', 'L1', 'Huber'],  default="MSE")
        parser.add_argument('--beta', type=float, required=True,  default=1.0)

        parser.add_argument('--input_size', type=int, required=True)
        parser.add_argument('--latent_dim', type=int, required=True)
        parser.add_argument('--batch_norm', default=False,
                            type=lambda x: (str(x).lower() == 'true'))
        parser.add_argument('--dropout_p', type=float,
                            default=0.0, choices=[ArgRange(0.0, 1.0)])

        parser.add_argument('--encoder_hidden_layers',
                            nargs='*', type=int, default=[128, 64, 32])
        parser.add_argument('--encoder_activations',
                            nargs='*', type=str, default="ReLU")

        parser.add_argument('--decoder_hidden_layers',
                            nargs='*', type=int, default=[32, 64, 128, 256])
        parser.add_argument('--decoder_activations',
                            nargs='*', type=str, default="ReLU")

        parser.add_argument('--output_activation', default=False,
                            type=lambda x: (str(x).lower() == 'true'))

        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--learning_rate', type=float,
                            default=0.002, choices=[ArgRange(0.0, None)])
        parser.add_argument('--weight_decay', type=float,
                            default=0.002, choices=[ArgRange(0.0, None)])

        parser.add_argument('--lr_scheduler', default=None,
                            choices=[None, "Step", "Exponential"])
        parser.add_argument('--scheduler_gamma', type=float,
                            choices=[ArgRange(0.0, 1.0)])
        parser.add_argument('--scheduler_step_size', type=int)

        return parser



