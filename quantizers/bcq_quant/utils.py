import torch
import torch.nn as nn
import numpy as np

class Packer:
    def __init__(self):
        self.s = torch.from_numpy(np.array([1, 2, 4, 8, 16, 32, 64, 128])).view(
            [-1, 1])
        if torch.cuda.is_available():
            self.s = self.s.cuda()
        self.w_pool = {}

    def __get_weight(self, shape, dtype):
        key = np.prod(shape)
        if key not in self.w_pool.keys():
            self.w_pool[key] = torch.zeros(shape, dtype=dtype)
            if torch.cuda.is_available():
                self.w_pool[key] = self.w_pool[key].cuda()
        return self.w_pool[key].reshape(shape)

    def pack(self, b):
        shape = b.shape
        p_b = b
        if torch.cuda.is_available():
            p_b = p_b.cuda()
        p_b = (p_b + 1) / 2  # (-1., +1.) -> (0, 1)
        p_b = torch.reshape(p_b, [8, -1]).type(torch.uint8)
        p_b = p_b * self.s
        p_b = p_b.sum(0)
        p_b = p_b.type(torch.uint8)
        return p_b, shape

    def unpack(self, pb, shape, dtype=torch.float16):
        b = self.__get_weight(shape, dtype).view([8, -1])
        for i in range(8):
            b[i] = (pb & 1)  # (pB%2)
            pb = pb >> 1  # //2
        b = b * 2 - 1
        b = b.reshape(shape)
        return b


PACKER = Packer()

class CompressionParameter(nn.Parameter):
    def compress(self, **kwargs):
        raise NotImplemented

    def decompress(self, **kwargs):
        raise NotImplemented
