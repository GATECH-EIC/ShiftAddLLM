import numpy as np
import torch
import torch.nn as nn

import lutgemm

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

        self.register_buffer('binaryWeight', torch.randint(-2147483648, 2147483647, (self.K //32 , wbit, self.N), dtype=torch.int32))
        self.register_buffer('alpha', torch.randn((self.num_groups, wbit, self.N), dtype=torch.half))
        self.register_buffer('scale', torch.randn((self.num_groups, self.N), dtype=torch.half))

        self.bWeight_cal = None
        self.alpha_cal = None
        self.q_bias_cal = None

        # self.bias = None
        self.bias = bias
    # def pack(self, linear, qbit = 3, group_size = 128,):
    #     # TODO: do quantization here
    #     self.bias = linear.bias.clone() if linear.bias is not None
        
    #     _, intweight, alpha, _, scale = quantize_shift(linear.weight.data, qbits=qbit, group_size=group_size)

    def parsing(self):
        # device = self.binaryWeight.device
        # assert device != torch.device("cpu"), "Device should be cuda"
        # device = int(str(device).split(":")[-1])
        # self.bWeight_cal, self.alpha_cal, self.q_bias_cal = lutgemm.parsing(self.binaryWeight, self.alpha.view(-1), self.K, self.N, self.wbit, False, self.num_groups, self.q_bias.view(-1), device)

        self.bWeight_cal, self.alpha_cal, self.q_bias_cal = self.binaryWeight.view(-1), self.alpha.view(-1), self.q_bias.view(-1)

    def kernel_forward(self, x):
        assert x.size(0) == 1 and x.size(1) == self.K
        output = x.new_zeros(x.size(0), self.N)
        lutgemm.lutgemm_compute_shift_scale(output, self.binaryWeight, self.alpha, \
        x, self.N, self.K, self.wbit, self.num_groups)
        #print("out: ", output.size(), self.bias.size())
        return output 

    def forward(self, x):
        if self.bWeight_cal is None or self.alpha_cal is None or self.q_bias_cal is None:
            self.parsing()

        assert x.shape[-1] == x.numel(), "Only support batch_size = 1 and 2 dimension input"

        outshape = list(x.shape)
        output = self.bias.clone()
        outshape[-1] = self.bias.numel()
        dtype = x.dtype

        lutgemm.lutgemm_compute(output, self.bWeight_cal, self.alpha_cal, self.q_bias_cal, x, self.N, self.K, self.wbit, self.num_groups)

        output = output.to(dtype).reshape(outshape)
        return output

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}, wbit={}, group_size={}'.format(
            self.K, self.N, self.bias is not None, self.wbit, self.group_size
        )

def make_lut(module, names, name='', wbit = 3, group_size = 128):
    if isinstance(module, LutLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(module, attr, LutLinear(tmp.in_features, tmp.out_features, group_size, wbit))
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

def translate_lut(bW, alpha, n_groups, wbit, N, K, Q, bias=None):
    '''
    bW is a list, each of size [1, n_groups, group_size, wbit]
    alpha [K, n_groups, bit]
    all list with size K and original weight size [K, N]
    '''
    #print("inp size: ", len(bW), bW[0].size(), n_groups, N, K,Q.size(), wbit)
    #assert alpha.size(0) == K and alpha.size(1) == n_groups and alpha.size(-1) == wbit 
    #assert bW[0].size(1) == 1 and bW[0].size(2) == 1
    # bW.permute(2, 0, 1, 3)
    
    new_bW = torch.cat([bw.squeeze(1).squeeze(1).unsqueeze(-1) for bw in bW], dim=-1).transpose(0, 2)
    new_alpha = alpha.permute(1, 2, 0)

    '''
    new_bW [K, bit, N]
    new_bW [n_groups, groups, bit, N]
    new_alpha [N, n_groups, bit]
    '''
    groupsize = K // n_groups
    
    assert (new_bW == 1).bitwise_or(new_bW == -1).all().item()
    new_bW = (new_bW == 1).contiguous().view(-1, 32, wbit, N)
    mask = (2**torch.arange(32))[None,:,None,None].to(new_bW.device)
    compressed_bW = (new_bW * mask).sum(1).to(torch.int32)

    #print("constants: ", n_groups, groupsize, wbit, N, K, compressed_bW.size())
    #compressed_bW = torch.zeros(K // 32, wbit, N,dtype=torch.int32)
    def binary(x, bits):
        mask = 2**torch.arange(bits).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

    def recover_weight_bias(bW, alpha, NUM_BITS, num_groups, N):
        ret_weight = bW.new_zeros(K, N, dtype=torch.float)
        group_size = groupsize

        for i in range(num_groups):
            ret_weight[i * group_size:(i+1) * group_size,:].zero_()
            for b in range(NUM_BITS):
                bin0 = binary(bW[:,b,:], 32)  #bW[b]: [K//32, N] -> [K//32, N, 32] (1, 0) 
                
                bin = bin0.transpose(1,2).flatten(0,1) #K, N -> 1 or 0
                #print("nn: ", bin0.size(), bin.size(), ret_weight[i * group_size:(i+1) * group_size,:].size(), alpha[i:i+1,b,:].size(), bin[i * group_size:(i+1) * group_size].size())
                #print("bw: ", (2 * bin[i * group_size:(i+1) * group_size].int() - 1).size(), alpha[i,b,:][None,:].size(), ret_weight[i * group_size:(i+1) * group_size,:].size())
                ret_weight[i * group_size:(i+1) * group_size,:] += alpha[i:i+1,b,:] * (2 * bin[i * group_size:(i+1) * group_size].int() - 1)

        return ret_weight
    #print(new_alpha.max(), new_alpha.min())
    recovered_Q = recover_weight_bias(compressed_bW, new_alpha, wbit, n_groups, N).half()
    lutlinear = LutLinear(infeatures=K,
                        outfeatures=N,
                        group_size = groupsize,
                        wbit=wbit,
                        bias=bias)
    
    with torch.no_grad():
        lutlinear.binaryWeight.data.copy_(compressed_bW.data)
        lutlinear.alpha.data.copy_(new_alpha.data)
        
    a = torch.rand(1, K).cuda().half()
    lutlinear = lutlinear.cuda()


    error = lutlinear.kernel_forward(a) - a.mm(recovered_Q.cuda())
    print("rec: ", (recovered_Q.cuda() - Q.T.half()).abs().max(), (error).abs().max(), error.abs().mean(), lutlinear.kernel_forward(a).abs().mean())

    return lutlinear



if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import lutgemm
    
    N, K = 3072, 768
    group_size = 128
    n_groups = K // group_size
    wbit = 3

    bias = torch.randn(N, device='cuda', dtype=torch.float16)

    #g, n, c, b
    bWeight = torch.rand(N, n_groups, group_size, wbit, device='cuda', dtype=torch.float32)
    # bWeight = torch.rand(group_size, N, n_groups, wbit, device='cuda', dtype=torch.float32)
    bWeight = torch.sign(bWeight)
    #n, c, b
    alpha = torch.randn([N, n_groups, wbit], device='cuda', dtype=torch.float32)

    # Q = torch.einsum('gncb,ncb->ncg', (bWeight, alpha)).flatten(1)
    Q = torch.einsum('nijl,nil->nij', (bWeight, alpha)).flatten(1)
    print(Q.size())

    translate_lut(bWeight, alpha, n_groups=n_groups, wbit=wbit, N=N, K=K, Q = Q, bias=None)
