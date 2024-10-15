import numpy as np
import torch
import torch.nn as nn

import lutgemm
from quantizers.bcq_quant.bcq_shift import get_shift_and_sign

class LutLinear(nn.Module): 
    def __init__(self, infeatures:int,
                        outfeatures:int,
                        group_size:int = 128,
                        wbit:int = 3,
                        bias=None):

        super().__init__()

        self.M = 1
        self.K = infeatures
        self.N = outfeatures
        self.group_size = group_size
        self.wbit = wbit
        self.num_groups = self.K // group_size
        self.num_apot = 3

        self.register_buffer('binaryWeight', torch.randint(-2147483648, 2147483647, (self.K //32 , wbit, self.N), dtype=torch.int32))
        self.register_buffer('alpha', torch.randint(1,8,(self.N, wbit, self.num_groups, self.num_apot), dtype=torch.int8))

        self.bias = bias

    def quantize_to_apot(self, x):
        shift, sign = get_shift_and_sign(x, rounding = "deterministic")
        x_rounded = (2.0 ** shift) * sign
        result = [shift.to(torch.int8)]
        for _ in range(self.num_apot - 1):
            x_diff = x - x_rounded
            shift, sign = get_shift_and_sign(x_diff, rounding = "deterministic")
            x_rounded += (2.0 ** shift) * sign
            result.append(shift.to(torch.int8))
        return torch.stack(result, dim = -1)

    def forward(self, x):

        assert x.shape[-1] == x.numel(), "Only support batch_size = 1 and 2 dimension input"

        outshape = list(x.shape)
        output = self.bias.clone()
        outshape[-1] = self.bias.numel()
        dtype = x.dtype

        lutgemm.lutgemm_compute_block_shiftInt8(output, self.binaryWeight, self.alpha, \
        x, self.N, self.K, self.wbit, self.num_groups, 3)

        output = output.to(dtype).reshape(outshape)
        return output

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}, wbit={}, group_size={}'.format(
            self.K, self.N, self.bias is not None, self.wbit, self.group_size * 8
        )

def make_lut(module, names, name='', wbit = 3, group_size = 128):
    if isinstance(module, LutLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(module, attr, LutLinear(tmp.in_features, tmp.out_features, group_size, wbit, bias = tmp.bias))
    for name1, child in module.named_children():
        make_lut(child, names, name + '.' + name1 if name != '' else name1, wbit, group_size)

def load_lut(model, checkpoint='', wbit = 3, group_size = 128):

    def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res

    import transformers
    from transformers import OPTConfig, OPTForCausalLM 
    config = OPTConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = OPTForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['model.decoder.project_out', 'model.decoder.project_in', 'lm_head']:
        if name in layers:
            del layers[name]
    make_lut(model, layers, wbit = wbit, group_size = group_size)

    if checkpoint != '':
        print('Loading model ...')
        model.load_state_dict(torch.load(checkpoint))
        
    model.seqlen = model.config.max_position_embeddings
    print('Done.')

    return model