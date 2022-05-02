import torch
import torch.nn.functional as F
from torch import nn


class Binarizer(torch.autograd.Function):
    """
    An elementwise function that bins values
    to 0 or 1 depending on a threshold of
    0.5

    Input: a tensor with values in range(0,1)

    Returns: a tensor with binary values: 0 or 1
    based on a threshold of 0.5

    Equation(1) in paper
    """
    @staticmethod
    def forward(ctx, i):
        return (i > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def bin_values(x):
    return Binarizer.apply(x)


def conv(ni, nf, ks=3, stride=1, padding=1, **kwargs):
    _conv = nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding, **kwargs)
    nn.init.kaiming_normal_(_conv.weight)
    nn.init.zeros_(_conv.bias)
    return _conv


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)


class ResBlock(nn.Module):
    def __init__(self, ni, nh=128):
        super().__init__()

        self.conv1 = conv(ni, nh)
        self.conv2 = conv(nh, ni)
        # initilize 2nd conv with zeros to preserve variance
        # known as Fixup initialization
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, return_imp_map=False):
        super(Encoder, self).__init__()
        self.return_imp_map = return_imp_map
        self.stem = nn.Sequential(conv(3, 128, 8, 4, 2), nn.ReLU(),
                                  ResBlock(128), nn.ReLU(),
                                  conv(128, 256, 4, 2, 1), nn.ReLU(),
                                  ResBlock(256), nn.ReLU(),
                                  ResBlock(256), nn.ReLU())

        self.head = nn.Sequential(conv(256, 64, 3, 1, 1),
                                  nn.Sigmoid(),
                                  Lambda(bin_values))

        self.imp_map_extractor = nn.Sequential(conv(256, 128), nn.ReLU(),
                                               conv(128, 128), nn.ReLU(),
                                               conv(128, 1), nn.Sigmoid())

        # initiating layers before Sigmoid with Xavier
        nn.init.xavier_normal_(self.head[0].weight)
        nn.init.xavier_normal_(self.imp_map_extractor[4].weight)

    def extra_repr(self):
        params = sum(p.numel() for p in self.parameters())
        return f'Total Params: {params}'

    def forward(self, x):
        stem = self.stem(x)
        if self.return_imp_map:
            return self.head(stem), self.imp_map_extractor(stem)
        else:
            return self.head(stem)


class DepthToSpace(torch.nn.Module):

    """
    A class used by the decoder while regenerating the image that moves values
    from the depth dimension to the height and width dimensions (spatial)

    Input: A tensor of size [N,C,H,W]
    Returns: A tensor of size [N,C/(block_size*block_size),H*block_size,W*block_size]

    Parameters
    ----------
    block_size: An int that is greater than 2. It decide

    Extra
    -----
    https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space
    """

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(conv(64, 512, 1, 1, 0), nn.ReLU(),
                                     ResBlock(512), nn.ReLU(),
                                     ResBlock(512), nn.ReLU(),
                                     DepthToSpace(2),
                                     conv(128, 256), nn.ReLU(),
                                     ResBlock(256), nn.ReLU(),
                                     DepthToSpace(4),
                                     conv(16, 32), nn.ReLU(),
                                     conv(32, 3))

    def extra_repr(self):
        params = sum(p.numel() for p in self.parameters())
        return f'Total Params: {params}'

    def forward(self, x):
        return self.decoder(x)


class Quantizer(torch.autograd.Function):
    """
    Equation (4) in paper implemented with L = 16 quantized values
    Elementwise function

    Input: An importance map of shape (N,1,h,w) with each element
            having a value between (0-1) representing importance probability

    Output: A tensor of the same shape with each element quantized to
            L different integer values from 0 to (L-1)
    """
    @staticmethod
    def forward(ctx, i):
        p = i.clone()
        L = 16
        for l in range(L):
            p[(p >= l/L)*(p < (l+1)/L)] = l
        return p

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def quantize_values(x):
    return Quantizer.apply(x)


class Mask(torch.autograd.Function):
    """
    Equation (6) and (7) in paper. This mask will be element-wise
    multiplied with the binary feature map generated by the encoder.

    Input:  A quantized importance map of shape (N,1,h,w)
            with L different integer values from 0 to (L-1)

    Output: A 3-D mask of dimensions (N,64,h,w) filled with
            sequential 1s and 0s

    """
    @staticmethod
    def forward(ctx, i):
        device = i.device
        N, _, H, W = i.shape
        n = 64
        L = 16
        qimp = i
        mask = torch.zeros(n, N*H*W).to(device)
        qimp_flat = qimp.view(1, N*H*W)
        for indx in range(n):
            mask[indx, :] = torch.where(indx < (n/L)*qimp_flat,
                                        torch.Tensor([1]).to(device), torch.Tensor([0]).to(device))
        mask = mask.view(n, N, H, W).permute((1, 0, 2, 3))
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        N, _, H, W = grad_output.shape
        return torch.ones(N, 1, H, W).to(grad_output.device)


def generate_mask(x):
    return Mask.apply(x)
