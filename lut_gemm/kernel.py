import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from .quant import LutLinear, make_lut
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

def unpack_to_fake_binaryWeight(packed_bweight, n_groups=1):

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

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def load_shiftaddllm_weight(model, weights_dir, model_name, wbits, is_lat=False):
    print(f"Loading shiftaddllm low-bit weights from {weights_dir}, model_name: {model_name}, wbits: {wbits}")

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


def convert_weight(BinaryWeight, Alpha):
    import einops

    N = BinaryWeight.size(0)
    K = BinaryWeight.size(1) * 32
    wbit = BinaryWeight.size(2)
    num_groups = Alpha.size(1)
    group_size = N // num_groups

    bb = []
    for ii in range(32):
        bb.append(BinaryWeight.bitwise_and(1 << ii) != 0)
    bb = torch.cat(bb, dim=1).float() * 2 - 1
    bb = einops.rearrange(bb, "k (i c) b -> k (c i) b", i = 32).float()
    new_bW = einops.rearrange(bb, "k (c g) b -> k b (c g)", g=group_size) #, "g k c b -> k b (c g)")            
    new_bW = (new_bW == 1) #.contiguous() .view(K, wbit, N).contiguous()
    new_bW = new_bW.view(K // 32, 32, wbit, N).contiguous()
    mask = (2**torch.arange(32))[None,:,None,None].to(new_bW.device)
    compressed_bW = (new_bW * mask).sum(1).to(torch.int32)
    alpha = Alpha.transpose(1, 2).float()

    return compressed_bW, alpha

def load_shiftaddllm_weight_with_kernel(model, weights_dir, model_name, wbits, is_lat=False):
    print(f"Loading shiftaddllm low-bit weights from {weights_dir}, model_name: {model_name}, wbits: {wbits}")
    print(f"Convert weight and run with CUDA kernel")
    print(f"WARNING!!! make sure you have kernel installed")

    assert is_lat, "Packed quantization weight only support lat method now"
    layers = model.model.decoder.layers

    # get the wbit and group size information from the packed weight
    subset = find_layers(layers[0])
    name = list(subset.keys())[0]
    checkpoint = torch.load(os.path.join(weights_dir, f"{model_name}_{0}.{name}_{wbits}bit.pt"))
    BinaryWeight = checkpoint["bWeight"]
    alpha = checkpoint["alpha"]

    wbit_weight = BinaryWeight.size(2)
    group_size = BinaryWeight.size(0) // alpha.size(1)
    assert wbit_weight == wbits, f"Weight bit mismatch, expect {wbits} bits, but got {wbit_weight} bits"

    # convert
    make_lut(layers, find_layers(layers), wbit = wbits, group_size = group_size)
    print(model)

    # load the weight
    layers = model.model.decoder.layers # layers now are LutLinear
    for i in tqdm(range(len(layers)), desc="Converting shiftaddllm low-bit weights", leave=True):
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
                compressed_bW, alpha = convert_weight(BinaryWeight, alpha)
                
                subset[name].binaryWeight.data = compressed_bW.to(subset[name].binaryWeight.data.dtype)
                subset[name].alpha.data = subset[name].quantize_to_apot(alpha)

            else:
                print(f"WARNING: no such file {temp_storage_pt}")