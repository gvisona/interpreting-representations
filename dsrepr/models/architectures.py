from itertools import cycle
import numpy as np
import torch
import torch.nn as nn
from dsrepr.utils.functions import gaussian_reparameterization, calc_output_size_convnet, calc_output_size_transpose_convnet


class ConvolutionalModule(nn.Module):
    def __init__(self,
                 input_channels=1,
                 feature_maps=[32, 32, 64, 64],
                 kernel_sizes=[4, 4, 4, 4],
                 paddings=1,
                 strides=2,
                 dilations=1,
                 batch_norm=False,
                 final_activation=False,
                 activations=nn.ReLU):
        super(ConvolutionalModule, self).__init__()

        assert isinstance(feature_maps, (list, tuple))
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]*len(feature_maps)
        if isinstance(paddings, int):
            paddings = [paddings]*len(feature_maps)
        if isinstance(strides, int):
            strides = [strides]*len(feature_maps)
        if isinstance(dilations, int):
            dilations = [dilations]*len(feature_maps)
        if not isinstance(activations, (list, tuple)):
            activations = [activations]*len(feature_maps)
        assert len(feature_maps) == len(
            kernel_sizes) == len(paddings) == len(strides)
        assert len(feature_maps) == len(dilations) == len(activations)

        self.feature_maps = feature_maps
        self.kernel_sizes = kernel_sizes
        self.paddings = paddings
        self.strides = strides
        self.dilations = dilations
        self.activations = activations
        self.batch_norm = batch_norm

        map_dim = [input_channels] + feature_maps
        self.convolutional_module = nn.Sequential()
        for k in range(len(map_dim) - 1):
            block = nn.Sequential()
            block.add_module("Conv", nn.Conv2d(in_channels=map_dim[k],
                                               out_channels=map_dim[k+1],
                                               kernel_size=kernel_sizes[k],
                                               padding=paddings[k],
                                               dilation=dilations[k],
                                               stride=strides[k]))
            if batch_norm:
                block.add_module("BatchNorm", nn.BatchNorm2d(num_features=map_dim[k+1],
                                                             eps=1e-05,
                                                             momentum=0.1,
                                                             affine=True,
                                                             track_running_stats=True))
            if k < len(map_dim)-2 or final_activation:
                block.add_module("Act", activations[k]())
            self.convolutional_module.add_module(
                "Block_{}".format(k+1), block)

        # Access layers with self.convolutional_module.__getattr__('Layer_1')

    def forward(self, x):
        # convolutional blocks
        return self.convolutional_module(x)

    def calc_output_size(self, img_height, img_width=None):
        # TODO fix for dilation
        if img_width is None:
            img_width = img_height
        height = img_height
        width = img_width
        for i in range(len(self.feature_maps)):
            height = (height - self.kernel_sizes[i] +
                    2*self.paddings[i])/self.strides[i] + 1
            width = (width - self.kernel_sizes[i] +
                    2*self.paddings[i])/self.strides[i] + 1
        return height, width

class TransposeConvolutionalModule(nn.Module):
    def __init__(self,
                 input_channels=1,
                 feature_maps=[64, 64, 32, 32, 1],
                 kernel_sizes=[4, 4, 4, 4, 4],
                 paddings=1,
                 strides=2,
                 dilations=1,
                 batch_norm=False,
                 final_activation=False,
                 activations=nn.ReLU):
        super(TransposeConvolutionalModule, self).__init__()

        assert isinstance(feature_maps, (list, tuple))
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]*len(feature_maps)
        if isinstance(paddings, int):
            paddings = [paddings]*len(feature_maps)
        if isinstance(strides, int):
            strides = [strides]*len(feature_maps)
        if isinstance(dilations, int):
            dilations = [dilations]*len(feature_maps)
        if not isinstance(activations, (list, tuple)):
            activations = [activations]*len(feature_maps)
        assert len(feature_maps) == len(
            kernel_sizes) == len(paddings) == len(strides)
        assert len(feature_maps) == len(dilations) == len(activations)

        self.feature_maps = feature_maps
        self.kernel_sizes = kernel_sizes
        self.paddings = paddings
        self.strides = strides
        self.dilations = dilations
        self.activations = activations
        self.batch_norm = batch_norm

        map_dim = [input_channels] + feature_maps
        self.transpose_convolutional_module = nn.Sequential()
        for k in range(len(map_dim) - 1):
            block = nn.Sequential()
            block.add_module("TranspConv", nn.ConvTranspose2d(in_channels=map_dim[k],
                                               out_channels=map_dim[k+1],
                                               kernel_size=kernel_sizes[k],
                                               padding=paddings[k],
                                               dilation=dilations[k],
                                               stride=strides[k]))
            if batch_norm:
                block.add_module("BatchNorm", nn.BatchNorm2d(num_features=map_dim[k+1],
                                                             eps=1e-05,
                                                             momentum=0.1,
                                                             affine=True,
                                                             track_running_stats=True))
            if k < len(map_dim)-2 or final_activation:
                block.add_module("Act", activations[k]())
            self.transpose_convolutional_module.add_module(
                "Block_{}".format(k+1), block)

        # Access layers with self.convolutional_block.__getattr__('Layer_1')

    def forward(self, x):
        # convolutional blocks
        return self.transpose_convolutional_module(x)

    def calc_output_size(self, img_size):
        size = img_size
        for i in range(len(self.feature_maps)):
            size = (size - self.kernel_sizes[i] +
                    2*self.paddings[i])/self.strides[i] + 1
        return size


class FullyConnectedModule(nn.Module):
    def __init__(self,
                 input_size=32,
                 hidden_layers=[256, 128, 32],
                 dropout_p=0,
                 activations=nn.ReLU,
                 batch_norm=False,
                 final_activation=False
                 ):
        super(FullyConnectedModule, self).__init__()
        assert isinstance(hidden_layers, (list, tuple))
        if not isinstance(activations, (list, tuple)):
            activations = [activations]*len(hidden_layers)
        assert len(hidden_layers) == len(activations)

        layers = [input_size] + hidden_layers
        self.fc_block = nn.Sequential()
        for k in range(len(layers) - 1):
            block = nn.Sequential()
            block.add_module("Linear", nn.Linear(layers[k], layers[k+1]))
            if batch_norm:
                block.add_module("BatchNorm", nn.BatchNorm1d(layers[k+1],
                                                             eps=1e-05,
                                                             momentum=0.1,
                                                             affine=True,
                                                             track_running_stats=True))
            if k < len(layers)-2 or final_activation:
                block.add_module("Act", activations[k]())
            if dropout_p > 0:
                block.add_module("Drop", nn.Dropout(p=dropout_p))

            self.fc_block.add_module(
                "Block_{}".format(k+1), block)

    def forward(self, x):
        # convolutional blocks
        return self.fc_block(x)



class ConvolutionalEncoder(nn.Module):
    def __init__(self, hparams):
        super(ConvolutionalEncoder, self).__init__()
        if isinstance(hparams.conv_encoder_activations, str):
            conv_encoder_activations = [getattr(
                nn, hparams.conv_encoder_activations)]*len(hparams.conv_encoder_feature_maps)
        elif isinstance(hparams.conv_encoder_activations, (list, tuple)):
            conv_encoder_activations = [
                getattr(nn, act) for act in hparams.conv_encoder_activations]
            if len(conv_encoder_activations)<len(hparams.conv_encoder_feature_maps):
                available_activations = cycle(conv_encoder_activations)
                conv_encoder_activations = [next(available_activations) for _ in range(len(hparams.conv_encoder_feature_maps))]

        input_channels = hparams.input_channels
        self.encoder_conv = ConvolutionalModule(input_channels=input_channels,
                                                feature_maps=hparams.conv_encoder_feature_maps,
                                                kernel_sizes=hparams.conv_encoder_kernel_sizes,
                                                paddings=hparams.conv_encoder_paddings,
                                                strides=hparams.conv_encoder_strides,
                                                dilations=hparams.conv_encoder_dilations,
                                                batch_norm=hparams.conv_batch_norm,
                                                activations=conv_encoder_activations,
                                                final_activation=True)

        final_shape = calc_output_size_convnet(hparams.input_size, hparams.conv_encoder_kernel_sizes, hparams.conv_encoder_paddings,
                                               hparams.conv_encoder_strides, hparams.conv_encoder_dilations)

        self.encoder_fc = None
        output_dim = final_shape[0]*final_shape[1]*hparams.conv_encoder_feature_maps[-1]
        if hparams.fc_encoder_hidden_layers:
            if isinstance(hparams.fc_encoder_activations, str):
                fc_encoder_activations = [getattr(
                    nn, hparams.fc_encoder_activations)]*len(hparams.fc_encoder_hidden_layers)
            elif isinstance(hparams.fc_encoder_activations, (list, tuple)):
                fc_encoder_activations = [
                    getattr(nn, act) for act in hparams.fc_encoder_activations]
                if len(fc_encoder_activations)<len(hparams.fc_encoder_hidden_layers):
                    available_activations = cycle(fc_encoder_activations)
                    fc_encoder_activations = [next(available_activations) for _ in range(len(hparams.fc_encoder_hidden_layers))]

            self.encoder_fc = FullyConnectedModule(input_size=output_dim,
                                                   hidden_layers=hparams.fc_encoder_hidden_layers,
                                                   dropout_p=hparams.fc_dropout_p,
                                                   activations=fc_encoder_activations,
                                                   batch_norm=hparams.fc_batch_norm,
                                                   final_activation=True)
            output_dim=hparams.fc_encoder_hidden_layers[-1]
        self.output_dim = output_dim

    def forward(self, x):
        x = self.encoder_conv(x)
        x = torch.flatten(x, start_dim=1)
        if self.encoder_fc is not None:
            x = self.encoder_fc(x)
        return x


class TransposeConvolutionalDecoder(nn.Module):
    def __init__(self, hparams):
        super(TransposeConvolutionalDecoder, self).__init__()
        self.decoder_fc = None
        if hparams.fc_decoder_hidden_layers:
            if isinstance(hparams.fc_decoder_activations, str):
                fc_decoder_activations = [
                    getattr(nn, hparams.fc_decoder_activations)]*len(hparams.fc_decoder_hidden_layers)
            elif isinstance(hparams.fc_decoder_activations, (list, tuple)):
                fc_decoder_activations = [
                    getattr(nn, act) for act in hparams.fc_decoder_activations]
                if len(fc_decoder_activations)<len(hparams.fc_decoder_hidden_layers):
                    available_activations = cycle(fc_decoder_activations)
                    fc_decoder_activations = [next(available_activations) for _ in range(len(hparams.fc_decoder_hidden_layers))]
            self.decoder_fc = FullyConnectedModule(input_size=hparams.latent_dim,
                                                   hidden_layers=hparams.fc_decoder_hidden_layers,
                                                   dropout_p=hparams.fc_dropout_p,
                                                   activations=fc_decoder_activations,
                                                   batch_norm=hparams.fc_batch_norm,
                                                   final_activation=True)

        conv_decoder_feature_maps = hparams.conv_decoder_feature_maps + [hparams.input_channels]
        conv_decoder_input_channels, conv_decoder_feature_maps = conv_decoder_feature_maps[0], conv_decoder_feature_maps[1:]
        self.conv_decoder_input_channels = conv_decoder_input_channels
        if isinstance(hparams.conv_decoder_activations, str):
            conv_decoder_activations = [getattr(
                nn, hparams.conv_decoder_activations)]*len(hparams.conv_decoder_feature_maps)
        elif isinstance(hparams.conv_decoder_activations, (list, tuple)):
            conv_decoder_activations = [
                getattr(nn, act) for act in hparams.conv_decoder_activations]
            if len(conv_decoder_activations)<len(hparams.conv_decoder_feature_maps):
                available_activations = cycle(conv_decoder_activations)
                conv_decoder_activations = [next(available_activations) for _ in range(len(hparams.conv_decoder_feature_maps))]
        self.decoder_conv = TransposeConvolutionalModule(input_channels=conv_decoder_input_channels,
                                                         feature_maps=conv_decoder_feature_maps,
                                                         kernel_sizes=hparams.conv_decoder_kernel_sizes,
                                                         paddings=hparams.conv_decoder_paddings,
                                                         strides=hparams.conv_decoder_strides,
                                                         dilations=hparams.conv_decoder_dilations,
                                                         batch_norm=hparams.conv_batch_norm,
                                                         final_activation=hparams.conv_decoder_final_activation,
                                                         activations=conv_decoder_activations)

    def forward(self, z):
        if self.decoder_fc is not None:
            z = self.decoder_fc(z)
        z_shape = z.shape
        h = int(np.sqrt(z_shape[-1]/self.conv_decoder_input_channels))
        out = self.decoder_conv(z.view(z_shape[0],self.conv_decoder_input_channels, h, -1))
        return out
