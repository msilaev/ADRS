import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
import torch.nn.init as init

DRATE = 2

class SubPixel1D(nn.Module):
    def __init__(self, r):
        super(SubPixel1D, self).__init__()
        self.r = r

    def forward(self, x):
        b, c, w = x.size()

        x = x.view(b, c // self.r, self.r, w)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(b, c // self.r, w * self.r)

        return x

class AudioTfilm(nn.Module):

    def __init__(self, layers = 4, pool_size = 2, strides = 2 ):

        super(AudioTfilm, self).__init__()

        self.pool_size = pool_size
        self.strides = strides

        self.n_filters = [128, 384, 512, 512]
        self.n_filtersizes = [65, 33, 17, 9]
        self.n_blocks = [128, 64, 32, 16, 8]

        self.downsampling_layers = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()
        self.layers = layers

        self.lstm_downsample = nn.ModuleList()

        self.lstm_upsample = nn.ModuleList()

        self.lstm_bottleneck = nn.ModuleList()

        # Downsampling layers
        for l, nf, fs in zip(list(range(self.layers)),
                             self.n_filters,
                             self.n_filtersizes):

            conv_layer = nn.Conv1d(in_channels=1
            if len(self.downsampling_layers) == 0
                    else self.n_filters[len(self.downsampling_layers) - 1],
                              out_channels = nf,
                              kernel_size = fs,
                              dilation = DRATE,
                              stride = 1,
                              #padding = 'same')
                              padding=fs-1 )

            init.orthogonal_(conv_layer.weight)

            x = nn.Sequential(
                conv_layer,
                nn.MaxPool1d(kernel_size = self.pool_size,
                             stride = self.strides),
                             #padding = self.pool_size//2),
                nn.LeakyReLU(0.2))

            self.downsampling_layers.append(x)

            nb = 128 // (2 ** l)
            nf = self.n_filters[l]

            lstm = nn.LSTM(input_size = nf, hidden_size = nf,
                                batch_first=True, bidirectional=False)

            ###########################3333
            # Initialize weights with orthogonal initialization

            initialize_lstm_weights(lstm)

            self.lstm_downsample.append(lstm)

        # Bottleneck layer
        nf = self.n_filters[-1]
        fs = self.n_filtersizes[-1]
        self.bottleneck_layer = nn.Sequential(
            nn.Conv1d(in_channels = nf,
                      out_channels = nf,
                      kernel_size = fs,
                      dilation = DRATE,
                      stride = 1,
                      padding = fs -1),
                      #padding = "same"),
            nn.MaxPool1d(kernel_size = self.pool_size,
                         stride = self.strides),
                         #padding = self.pool_size // 2),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2)
        )

        lstm = nn.LSTM(input_size=nf, hidden_size=nf,
                       batch_first=True, bidirectional=False)

        initialize_lstm_weights(lstm)
        self.lstm_bottleneck.append(lstm)

        # Upsampling layers
        len_filters = len(self.n_filters)
        rev_n_filters = self.n_filters[::-1]
        rev_n_filtersizes = self.n_filtersizes[::-1]

        for ind in range(len_filters):

            fs = rev_n_filtersizes[ind]

            if (ind == 0) :
                in_channels = self.n_filters[-1]
            elif (ind == 1) :
                in_channels = rev_n_filters[ind-1] + self.n_filters[-1]
            else :
                in_channels = rev_n_filters[ind-1] + rev_n_filters[ind-1]

            conv_layer = nn.Conv1d(in_channels = in_channels ,
                              out_channels = 2*rev_n_filters[ind],
                              kernel_size = fs,
                              dilation = DRATE,
                              #padding = 'same')
                              #stride = 1,
                              padding = fs -1)

            init.orthogonal_(conv_layer.weight)

            x = nn.Sequential(
                    conv_layer,
                    #nn.BatchNorm1d(2 * rev_n_filters[ind]),
                    nn.Dropout(0.5),
                    nn.ReLU(),
                    SubPixel1D(r=2)
                )

            self.upsampling_layers.append(x)

            nf = rev_n_filters[ind]

            lstm = nn.LSTM(input_size = nf, hidden_size = nf,
                           batch_first = True, bidirectional = False)

            initialize_lstm_weights(lstm)

            self.lstm_upsample.append(lstm)

        # Final layer
        conv_layer = nn.Conv1d(in_channels=2 * rev_n_filters[-1],
                  out_channels=2, kernel_size=9, padding=4)

        init.orthogonal_(conv_layer.weight)

        self.final_layer = nn.Sequential (conv_layer, SubPixel1D(r=2))

    #######################################
    def _make_normalizer(self, x, n_filters, n_block, type, ind):
        # x: Input tensor of shape (batch_size, channels, width)

        # Define the MaxPool1D layer
        max_pool = nn.MaxPool1d(kernel_size=int(n_block))

        # Apply MaxPool1D to the input tensor
        x_down = max_pool(x)  # Resulting shape (batch_size, channels, width // n_block)

        # Permute the dimensions to prepare for LSTM
        x_down = x_down.permute(0, 2, 1).contiguous()  # Shape: (batch_size, width, channels)

        if type == "downsample":
            lstm = self.lstm_downsample[ind]
        elif type == "bottleneck":
            lstm = self.lstm_bottleneck[ind]
        elif type == "upsample":
            lstm = self.lstm_upsample[ind]
        else:
            raise ValueError(f"Invalid type: {type}. Expected one of ['downsample', 'bottleneck', 'upsample']")

        # Apply LSTM to the pooled and permuted tensor
        x_rnn, _ = lstm(x_down)

        return x_rnn

    def _apply_normalizer(self, x_in, x_norm, n_filters, n_block):

        batch_size, channels, width = x_in.size()
        n_steps = width//n_block

        x_in = x_in.permute(0, 2, 1)

        x_in = x_in.reshape( batch_size, n_steps, n_block, channels )
        x_norm = x_norm.reshape( batch_size, n_steps, 1, channels )

        x_out = x_norm * x_in

        x_out = \
            x_out.permute(0, 3, 1, 2).\
                contiguous().reshape(batch_size, channels, n_steps * n_block)

        return x_out


    def forward(self, x):

        x = x.transpose(1,2)

        x_start = x

        downsampling_l = []

        for i, layer in enumerate(self.downsampling_layers):

            x = layer(x)

            nb = 128 // (2 ** i)
            nf = self.n_filters[i]

            x_norm = self._make_normalizer(x, nf, nb, type = "downsample", ind = i)
            x = self._apply_normalizer(x, x_norm, nf, nb)

            downsampling_l.append(x)

        x = self.bottleneck_layer(x)
        nb = 128 // (2 ** self.layers)
        nf = self.n_filters[-1]

        x_norm = self._make_normalizer(x, nf, nb, type="bottleneck", ind=0)
        x = self._apply_normalizer(x, x_norm, nf, nb)

        ind = 0
        for l, l_in in list(zip( self.upsampling_layers, reversed(downsampling_l) )):

            x = l(x)
            nf = x.shape[1]
            nb = 128 // (2 ** self.layers)

            x_norm = self._make_normalizer(x, nf, nb, type="upsample", ind=ind)
            x = self._apply_normalizer(x, x_norm, nf, nb)

            x = torch.concat((x, l_in), axis=1)

            ind +=1

        x = self.final_layer(x)

        # -----------------------------
        # Additive residual connection
        # -----------------------------
        x = x + x_start

        x = x.transpose(1,2)

        return x

    def create_objective(self, P, Y):
        # Compute L2 loss

        l2_loss = torch.mean((P - Y) ** 2, dim=[1, 2]) + 1e-6
        norm = torch.mean(Y ** 2, dim=[1, 2])

        avg_l2_loss = torch.mean(l2_loss, dim =0)
        avg_norm = torch.mean(norm, dim =0)

        sqrt_l2_loss = torch.sqrt(torch.mean((P - Y) ** 2, dim=[1, 2]) + 1e-6)
        avg_sqrt_l2_loss = torch.mean(sqrt_l2_loss, dim=0)

        sqrn_l2_norm = torch.sqrt(torch.mean(Y ** 2, dim=[1, 2]))

        snr = 20 * torch.log10(sqrn_l2_norm / (sqrt_l2_loss + 1e-8))
        avg_snr = torch.mean(snr, dim=0)

        return avg_sqrt_l2_loss, avg_l2_loss, avg_norm, avg_snr


# Define the weights initialization functions
def weights_init(m):

    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def initialize_lstm_weights(lstm):

    for name, param in lstm.named_parameters():
        if 'weight_hh' in name:
            nn.init.orthogonal_(param)
        elif 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)

# Example usage
if __name__ == "__main__":
    ## Initialize the model

    model = SubPixel1D(r=2)

    np_array_total = np.arange(0, 10)  # np.array([1, 2, 3, 4, 5, 6 ,7, 8])
    input_tensor_total = torch.tensor(np_array_total.flatten(),
                                      dtype=torch.float32).unsqueeze(0).unsqueeze(2)

    y = model(input_tensor_total)

    #model = AudioTfilm()






