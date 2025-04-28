import torch.nn.init as init
import torch
import torch.nn as nn

from .multiScaleConv import MultiscaleConv1DBlock as MultiscaleConvBlock
from .subPixel1D import SubPixel1D
from .superPixel1D import SuperPixel1D

# -------------------
#  Generator
# -------------------
class Generator(nn.Module):

    def __init__(self, layers=4, n_filters=(64, 128, 256, 512, 512)):

        super(Generator, self).__init__()

        self.downsampling_layers = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()
        self.layers = layers
        self.n_filters = n_filters

        n_in = 1
        n_out_arr = []

        # -----------------
        # Downsampling layers
        # -----------------
        for l in range(self.layers):
            n_out = self.n_filters[l] // 4

            conv_layer = MultiscaleConvBlock(in_channels=n_in,
                                             out_channels=n_out)
            # conv_layer.apply(self.initialize_weights)

            x = nn.Sequential(
                conv_layer,
                nn.ReLU(),
                # nn.BatchNorm1d(4*n_in),
                # nn.LeakyReLU(0.2),
                SuperPixel1D(r=2)
            )

            self.downsampling_layers.append(x)

            # n_out comes from 4 stacked layers and SuperPixel

            # n_out = 8*n_in
            n_out_arr.append(8 * n_out)
            n_in = 8 * n_out

        # -----------------
        # Bottleneck layer
        # -----------------
        conv_layer_1 = MultiscaleConvBlock(in_channels=n_in,
                                           out_channels=n_in // 8)

        conv_layer_2 = MultiscaleConvBlock(in_channels=n_in,
                                           out_channels=n_in // 2)

        # conv_layer_1.apply(self.initialize_weights)
        # conv_layer_2.apply(self.initialize_weights)

        x = nn.Sequential(
            conv_layer_1,
            nn.ReLU(),
            # nn.BatchNorm1d(4 * n_in),
            # nn.LeakyReLU(0.2),
            SuperPixel1D(r=2),
            conv_layer_2,
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            SubPixel1D(r=2))

        self.bottleneck_layer = x
        # we add here also stack
        n_out = n_in

        # -----------------
        # Upsampling layer
        # -----------------
        for l in range(self.layers):
            n_in = n_out + n_out_arr[len(n_out_arr) - l - 1]

            n_out_conv = self.n_filters[len(n_out_arr) - l - 1] // 4

            conv_layer = MultiscaleConvBlock(in_channels=n_in,
                                             out_channels=n_out_conv)

            # conv_layer.apply(self.initialize_weights)

            x = nn.Sequential(
                conv_layer,
                nn.Dropout(0.5),
                nn.LeakyReLU(0.2),
                SubPixel1D(r=2))

            self.upsampling_layers.append(x)

            # n_out comes from Skip Connection, 4 stacked layers, SubPixel
            n_out = 2 * n_out_conv

        ####################
        # define final layer
        ####################
        x = nn.Conv1d(n_out, 1, kernel_size=27, padding=13)
        self.final_layer = x

    def initialize_weights(self, module):
        if isinstance(module, nn.Conv1d):
            init.orthogonal_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)

    def forward(self, x):

        x = x.transpose(1, 2)
        x_start = x
        downsampling_l = []

        for layer in self.downsampling_layers:
            x = layer(x)
            downsampling_l.append(x)
        x = self.bottleneck_layer(x)

        for l, l_in in list(zip(self.upsampling_layers, reversed(downsampling_l))):
            x = torch.concat((x, l_in), axis=1)
            x = l(x)

        x = self.final_layer(x)
        x = x + x_start
        x = x.transpose(2, 1)

        return x

    def create_objective(self, P, Y):
        # Compute L2 loss

        l2_loss = torch.mean((P - Y) ** 2, dim=[1, 2]) + 1e-6

        norm = torch.mean(Y ** 2, dim=[1, 2])

        avg_l2_loss = torch.mean(l2_loss, dim=0)
        avg_norm = torch.mean(norm, dim=0)

        sqrt_l2_loss = torch.sqrt(torch.mean((P - Y) ** 2, dim=[1, 2]) + 1e-6)
        avg_sqrt_l2_loss = torch.mean(sqrt_l2_loss, dim=0)

        # avg_sqrt_l2_loss = torch.mean(sqrt_l2_loss, dim=0)

        sqrn_l2_norm = torch.sqrt(torch.mean(Y ** 2, dim=[1, 2]))

        snr = 20 * torch.log10(sqrn_l2_norm / (sqrt_l2_loss + 1e-8))
        avg_snr = torch.mean(snr, dim=0)

        return avg_sqrt_l2_loss, avg_l2_loss, avg_norm, avg_snr


#-------------------
#  Discriminator
#-------------------
class Discriminator(nn.Module):

    def __init__(self, layers, time_dim, n_filters = (64, 128, 256, 256, 512)):

        super(Discriminator, self).__init__()

        self.layers = layers
        self.downsampling_layers = nn.ModuleList()

        n_in  = 1
        n_out = 128

        conv_layer = MultiscaleConvBlock(in_channels = n_in,
                                         out_channels = n_out//4)

        #conv_layer.apply(self.initialize_weights)

        n_in = n_out

        x = nn.Sequential(
            conv_layer,
            nn.LeakyReLU(0.2))

        #------------
        # 32 channels
        #------------
        self.downsampling_layers.append(x)

        self.n_filters = n_filters

        for l in range(self.layers):

            n_out = self.n_filters[l]//4

            conv_layer = MultiscaleConvBlock(in_channels = n_in,
                                        out_channels = n_out)
            #conv_layer.apply(self.initialize_weights)
            batch_norm = nn.BatchNorm1d(4*n_out)
            #batch_norm.apply(self.initialize_weights)

            x = nn.Sequential(
                conv_layer,
                batch_norm,
                nn.Dropout(0.5),
                nn.LeakyReLU(0.2),
                SuperPixel1D(r=2))

            #x.apply(self.initialize_weights)

            n_in = 8*n_out
            # ------------
            # 32 - 64 - 128 - 256 - 256 - 128 - 64 - 32 channels, 2 x n_filters
            # ------------

            self.downsampling_layers.append(x)

        self.input_features = n_in*time_dim // (2 ** self.layers)

        fc_outdim = 1024//32

        self.fc_1 = nn.Linear(self.input_features, fc_outdim)
        self.fc_2 = nn.Linear(fc_outdim, 1)

        self.final_layer = nn.Sequential(
            self.fc_1,
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            self.fc_2)

    def initialize_weights(self, module):
        if isinstance(module, nn.Conv1d):
            init.orthogonal_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)

    def forward(self, x):

        fmap_r = []

        x = x.transpose(1,2)

        for l in self.downsampling_layers:

            x = l(x)
            fmap_r.append(x)

        x = x.view(x.size(0), -1)

        x = self.final_layer(x)

        return x, fmap_r


def BCEWithSquareLoss(discriminator_output, targets):
    """
    Computes the adversarial loss using BCEWithLogitsLoss.

    This function applies the standard binary cross-entropy loss with logits,
    commonly used in GANs for adversarial training. In some cases, MSE loss
    can also be used as an alternative adversarial loss.

    Args:
        discriminator_output (torch.Tensor): The output logits from the discriminator.
        targets (torch.Tensor): The target labels (real or fake).

    Returns:
        torch.Tensor: The computed adversarial loss.
    """

    bce_loss = nn.BCEWithLogitsLoss()
    loss = bce_loss(discriminator_output, targets)

    # -----------------
    # Sometimes MSE loss is used as adversarial loss
    # -----------------

    # mse_loss = nn.MSELoss()
    # loss = mse_loss(discriminator_output, targets)

    return loss

