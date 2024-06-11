import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .bcq_shift import find_B_torch
from .bcq import quantize as bcq_quantize

def quantize(x, alpha, groupsize = -1, use_bst=True):

    alpha.to(x.device)
    N, K = x.shape
    wbits = alpha.shape[-1]
    if groupsize == -1:
        groupsize = K
    x = x.reshape([N, K // groupsize, groupsize])
    w = x
    B = torch.zeros(N, K // groupsize, groupsize, wbits, device=x.device)

    # B[:, :, :, 0] = torch.sign(w)
    # for i in range(1, wbits):
    #     w = w - B[:, :, :, i - 1] * alpha[:, :, i - 1].unsqueeze(-1).expand_as(w)
    #     B[:, :, :, i] = torch.sign(w)
    
    if use_bst:
        B = find_B_torch(x.reshape(-1, groupsize), alpha.reshape(-1, wbits))
        B = B.reshape([N, K // groupsize, groupsize, wbits])
    else:
        B[:, :, :, 0] = torch.sign(w)
        for i in range(1, wbits):
            w = w - B[:, :, :, i - 1] * alpha[:, :, i - 1].unsqueeze(-1).expand_as(w)
            B[:, :, :, i] = torch.sign(w)

    ret = torch.einsum('mngb,mnb->mng', (B, alpha))
    ret = ret.reshape([N, K])

    return ret, B


class BCQuantizer(nn.Module):

    def __init__(self, shape, groupsize=-1, wbits=3, rounds = 5, use_bst=True, apot_nums=1):
        super(BCQuantizer, self).__init__()

        self.wbits = wbits
        self.groupsize = groupsize
        self.rounds = rounds
        self.use_bst = use_bst
        self.apot_nums = apot_nums

        N, K = shape
        if groupsize == -1:
            num_group = 1
        else:
            # if K % groupsize != 0:
            #     raise ValueError(f'K % groupsize != 0, K = {K}, groupsize = {groupsize}')
            num_group = K // groupsize

        self.register_buffer('alpha', torch.zeros(N, num_group, wbits))


    def find_params(self, x, input=None):
        # WARNING: assert x is linear weight
        if len(x.shape) != 2:
            raise ValueError(r'x should be linear weight')
        # if input is None:
        #     wf = None
        # else:
        #     input = torch.abs(input)
        #     input = F.normalize(input, p=2, dim=0)
        #     wf = input.unsqueeze(0).expand_as(x)

        # _, _, self.alpha, _, _ = \
        #     quantize_shift(x, qbits=self.wbits, rounds=self.rounds, group_size=self.groupsize,
        #                    exponent=0.0, clipping=1.0, pruning=0.0, use_bst=self.use_bst, wf=None, apot_nums=self.apot_nums)
        _, _, self.alpha, _ = \
            bcq_quantize(x, qbits=self.wbits, rounds=self.rounds, group_size=self.groupsize, use_bst=self.use_bst)
        assert torch.all(torch.sort(self.alpha, dim=2, descending=True)[0] == self.alpha), "alpha should be in descending order, something wrong with 'quantize_shift'"

    def quantize(self, x):
        if not self.ready():
            self.find_params(x)
        return quantize(x, self.alpha, self.groupsize)


    def ready(self):
        return torch.any(self.alpha != 0)


if __name__ == "__main__":
    layer = nn.Linear(128, 256)
    quantizer = BCQuantizer(layer, groupsize=-1, wbits=3, rounds=5)
    quantizer.find_params(layer.weight.data)
    ret, B = quantizer.quantize(layer.weight.data)
    print(B - quantizer.B)
    print(torch.norm(B - quantizer.B))
    assert torch.all(abs(B - quantizer.B)<1e-4)
    assert torch.all(abs(ret - quantizer.ret.cpu())<1e-4)
    