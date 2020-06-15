import torch
import torch.nn as nn


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




# class ConvolutionalEncoder(nn.Module):

#     def __init__(self,
#                  latent_dim=32,
#                  input_channels=1,
#                  feature_maps=[32, 32, 64, 64],
#                  kernel_sizes=[4, 4, 4, 4],
#                  paddings=1,
#                  strides=2,
#                  final_size=[4, 4],
#                  fc_layers=[256],
#                  act_class=nn.ReLU):
#         super(ConvolutionalEncoder, self).__init__()

#         assert isinstance(feature_maps, (list, tuple))
#         if isinstance(kernel_sizes, int):
#             kernel_sizes = [kernel_sizes]*len(feature_maps)
#         if isinstance(paddings, int):
#             paddings = [paddings]*len(feature_maps)
#         if isinstance(strides, int):
#             strides = [strides]*len(feature_maps)
#         assert len(feature_maps) == len(
#             kernel_sizes) == len(paddings) == len(strides)
#         # convolutional blocks
#         conv_blocks = []
#         map_dim = [in_channels] + feature_maps
#         for k in range(len(map_dim) - 1):
#             conv_blocks.append(
#                 nn.Sequential(nn.Conv2d(in_channels=map_dim[k],
#                                         out_channels=map_dim[k+1],
#                                         kernel_size=kernel_sizes[k],
#                                         padding=1,
#                                         stride=2),
#                               act_class()))
#         self.conv_net = nn.Sequential(*conv_blocks)

#         # flattening
#         flattened_tensor_length = final_size[0]*final_size[1]*feature_maps[-1]

#         # fc blocks
#         fc_blocks = []
#         fc_layers.insert(0, flattened_tensor_length)
#         for k in range(len(fc_layers) - 1):
#             fc_blocks.append(
#                 nn.Sequential(
#                     nn.Linear(fc_layers[k], fc_layers[k+1]),
#                     act_class()))
#         self.fc_net = nn.Sequential(*fc_blocks)

#         self.fc_mu = nn.Linear(fc_layers[-1], latent_dim)
#         self.fc_var = nn.Linear(fc_layers[-1], latent_dim)

#     def forward(self, x):

#         # convolutional blocks
#         x = self.conv_net(x)
#         # flattening
#         x = torch.flatten(x, start_dim=1)
#         # fc blocks
#         x = self.fc_net(x)
#         # split the result into mu and var components
#         # of the latent Gaussian distribution
#         mu = self.fc_mu(x)
#         logvar = self.fc_var(x)

#         return mu, logvar


# class TransposeConvolutionalDecoder(torch.nn.Module):

#     def __init__(self,
#                  latent_dim=32,
#                  out_channels=1,
#                  feature_maps=[64, 64, 32, 32],
#                  kernel_size=[4, 4, 4, 4],
#                  start_size=[4, 4],
#                  act_class=nn.ReLU,
#                  output_act_class=None):

#         super(TransposeConvolutionalDecoder, self).__init__()

#         # copy over
#         self.feature_maps = feature_maps
#         self.pool_size = start_size

#         flattened_tensor_length = start_size[0]*start_size[1]*feature_maps[0]
#         self.input_layer = nn.Linear(latent_dim, flattened_tensor_length)

#         # deconvolutional blocks
#         deconv_blocks = []
#         map_dim = feature_maps + [out_channels]
#         for k in range(len(map_dim) - 1):
#             module_list = [nn.ConvTranspose2d(
#                 in_channels=map_dim[k],
#                 out_channels=map_dim[k+1],
#                 kernel_size=kernel_size[k],
#                 padding=1,
#                 stride=2)]
#             # add activations
#             if k < len(feature_maps) - 2:
#                 module_list.append(act_class())
#             else:
#                 if output_act_class:
#                     module_list.append(output_act_class())
#             deconv_blocks.append(nn.Sequential(*module_list))

#         self.deconv_net = nn.Sequential(*deconv_blocks)

#     def forward(self, x):
#         # reshape
#         x = self.input_layer(x)
#         x = x.view(-1, self.feature_maps[0],
#                    self.pool_size[0], self.pool_size[1])

#         # deconvolutional blocks
#         x = self.deconv_net(x)
#         return x
