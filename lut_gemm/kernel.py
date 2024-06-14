import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm

# import lutgemm

def pack_binaryWeight(bWeight:torch.Tensor):
    # bWeight: [N, n_groups, group_size, wbit]

    assert bWeight.dim() == 4, "BinaryWeight shape should be [N, n_groups, group_size, wbit], but got {}".format(bWeight.size())

    N, n_groups, group_size, wbit = bWeight.shape

    new_bW = bWeight.reshape(N, -1, wbit).permute(1, 2, 0) # [n_groups * group_size, wbit, N]

    assert (new_bW == 1).bitwise_or(new_bW == -1).all().item(), "BinaryWeight should be 1 or -1, but got other values"
    new_bW = (new_bW == 1).contiguous().view(-1, 32, wbit, N)
    mask = (2**torch.arange(32))[None,:,None,None].to(new_bW.device)
    compressed_bW = (new_bW * mask).sum(1).to(torch.int32) # [K/32, wbit, N]

    return compressed_bW.permute(2, 0, 1).contiguous()


def unpack_weight(bW:torch.Tensor, alpha:torch.Tensor):
    # bW: [N, K/32, wbit]
    # alpha: [N, n_groups, wbit]
    bW = bW.permute(1, 2, 0).contiguous() # [K/32, wbit, N]
    alpha = alpha.permute(1, 2, 0).contiguous() # [n_groups, wbit, N]
    assert bW.dim() == 3, "BinaryWeight shape should be [K/32, wbit, N], but got {}".format(bW.size())
    assert alpha.dim() == 3, "Alpha shape should be [n_groups, wbit, N], but got {}".format(alpha.size())
    assert bW.size(1) == alpha.size(1), "BinaryWeight and Alpha should have the same number of bits, but got {} and {}".format(bW.size(1), alpha.size(1))
    assert bW.size(2) == alpha.size(2), "BinaryWeight and Alpha should have the same number of N channels, but got {} and {}".format(bW.size(2), alpha.size(2))

    def binary(x, bits):
        mask = 2**torch.arange(bits).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
    
    K = bW.size(0) * 32
    N = bW.size(-1)
    num_groups = alpha.size(0)
    NUM_BITS = alpha.size(1)
    group_size = K // num_groups

    ret_weight = bW.new_zeros(K, N, dtype=torch.float)
    
    for i in range(num_groups):
        ret_weight[i * group_size:(i+1) * group_size,:].zero_()
        for b in range(NUM_BITS):
            bin0 = binary(bW[:,b,:], 32)  #bW[b]: [K//32, N] -> [K//32, N, 32] (1, 0) 
            bin = bin0.transpose(1,2).flatten(0,1) #K, N -> 1 or 0
            ret_weight[i * group_size:(i+1) * group_size,:] += alpha[i:i+1,b,:] * (2 * bin[i * group_size:(i+1) * group_size].int() - 1)

    ret_weight = ret_weight.transpose(0, 1)
    return ret_weight

def packed_to_fake_binaryWeight(packed_bweight, n_groups=1):

    def binary(x, bits):
        mask = 2**torch.arange(bits).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

    wbit = packed_bweight.size(-1)
    packed_bweight = packed_bweight.permute(1, 2, 0).contiguous()
    weight_list = []
    for i in range(wbit):
        binaryWeight_perBit = binary(packed_bweight[:,i,:], 32) * 2 - 1
        binaryWeight_perBit = binaryWeight_perBit.transpose(0, 1).flatten(1).unsqueeze(-1)
        weight_list.append(binaryWeight_perBit)
    weight = torch.cat(weight_list, dim=-1)
    K,N,bit = weight.size()
    groupsize = N // n_groups
    weight = weight.view(K, n_groups, groupsize, bit)
    return weight

def load_shiftaddllm_weight(model, weights_dir, model_name, wbits, is_lat=False):
    print(f"Loading shiftaddllm low-bit weights from {weights_dir}, model_name: {model_name}, wbits: {wbits}")
    def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res

    assert is_lat, "Packed quantization weight only support lat method now"
    layers = model.model.decoder.layers
    for i in tqdm(range(len(layers)), desc="Loading shiftaddllm low-bit weights", leave=False):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            layer_name = f"{i}.{name}"
            temp_storage_pt = os.path.join(weights_dir, f"{model_name}_{layer_name}_{wbits}bit.pt")

            if os.path.exists(temp_storage_pt):
                # print(f"load from {temp_storage_pt}")
                checkpoint = torch.load(temp_storage_pt)
                BinaryWeight = checkpoint["bWeight"]
                alpha = checkpoint["alpha"]
                alpha = alpha.repeat_interleave(8, dim=0)
                W = unpack_weight(BinaryWeight, alpha)
                W = W.transpose(0, 1).contiguous()
                subset[name].weight.data = W.to(subset[name].weight.data.dtype)
            else:
                print(f"WARNING: no such file {temp_storage_pt}")

if __name__ == "__main__":
    N, K = 128, 768
    group_size = 128
    n_groups = K // group_size
    wbit = 3

    #g, n, c, b
    bWeight = torch.randn([N, n_groups, group_size, wbit], device='cuda', dtype=torch.float32)
    # print(bWeight)
    bWeight = torch.sign(bWeight)
    # bWeight = 2.0 * (bWeight > 0) - 1
    #n, c, b
    alpha = torch.abs(torch.randn([N, n_groups, wbit], device='cuda', dtype=torch.float32))

    Q = torch.einsum('nijl,nil->nij', (bWeight, alpha.clone())).flatten(1)
    binaryWeight = pack_binaryWeight(bWeight.clone())
    print(binaryWeight.shape)
    print(binaryWeight.dtype)
    print(alpha.shape)

    recovered_Q = unpack_weight(binaryWeight, alpha)
    print(recovered_Q.shape)
    # print(Q.shape)

    print((recovered_Q.cuda() - Q).abs().max())