import torch
from torch import nn
import torch.nn.functional as F
from torch import tensor

class Binarizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        return (i>0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def bin_values(x):
    return Binarizer.apply(x)

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)

relu = nn.ReLU()

def conv(ni, nf, ks=3, stride=1, padding=1, **kwargs):
    _conv = nn.Conv2d(ni, nf, kernel_size=ks,stride=stride,padding=padding, **kwargs)
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
        #initilize 2nd conv with zeros to preserve variance
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        return x  + self.conv2(F.relu(self.conv1(x)))

class Encoder(nn.Module):
    def __init__(self,return_imp_map=False):
        super(Encoder, self).__init__()
        self.return_imp_map = return_imp_map
        self.stem = nn.Sequential(conv(3, 128, 8, 4, 2), relu,
                                   ResBlock(128), relu,
                                   conv(128, 256, 4, 2, 1), relu,
                                   ResBlock(256), relu,
                                   ResBlock(256), relu)

        self.head = nn.Sequential(conv(256, 64, 3, 1, 1),
                                   nn.Sigmoid(),
                                   Lambda(bin_values))


        self.imp_map_extractor = nn.Sequential(conv(256,128), relu,
                                                conv(128,128), relu,
                                                conv(128,1), nn.Sigmoid())

        #initiating layers before Sigmoid with Xavier
        nn.init.xavier_normal_(self.head[0].weight)
        nn.init.xavier_normal_(self.imp_map_extractor[4].weight)

    def extra_repr(self):
        params = sum(p.numel() for p in self.parameters())
        return f'Total Params: {params}'

    def forward(self,x):
        stem = self.stem(x)
        if self.return_imp_map:return self.head(stem), self.imp_map_extractor(stem)
        else: return self.head(stem)


class DepthToSpace(torch.nn.Module):
    def __init__(self,block_size):
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
        self.decoder = nn.Sequential(conv(64,512,1,1,0), relu,
                                    ResBlock(512), relu,
                                    ResBlock(512), relu,
                                    DepthToSpace(2),
                                    conv(128,256), relu,
                                    ResBlock(256), relu,
                                    DepthToSpace(4),
                                    conv(16,32), relu,
                                    conv(32,3))

    def extra_repr(self):
        params = sum(p.numel() for p in self.parameters())
        return f'Total Params: {params}'

    def forward(self,x):
        return self.decoder(x)

class Quantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        p = i.clone()
        L = 16
        for l in range(L):
            p[(p>=l/L)*(p<(l+1)/L)] = l
        return p
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def quantize_values(x):
    return Quantizer.apply(x)

class Mask(torch.autograd.Function):
    @staticmethod
    def forward(ctx,i):
        if i.is_cuda: 
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        N,_,H,W = i.shape
        n = 64
        L = 16
        mask = torch.zeros(n, N*H*W).to(device)
        qimp_flat = qimp.view(1, N*H*W)
        for indx in range(n):
            mask[indx,:] = torch.where(indx < (n/L)*qimp_flat,torch.Tensor([1]).to(device),torch.Tensor([0]).to(device))
        mask = mask.view(n,N,H,W).permute((1,0,2,3))
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        N,C,H,W = grad_output.shape
        if grad_output.is_cuda: return torch.ones(N,1,H,W).cuda()
        else: return torch.ones(N,1,H,W)

def generate_mask(x):
    return Mask.apply(x)