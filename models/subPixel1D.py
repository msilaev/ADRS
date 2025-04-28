import torch.nn as nn

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