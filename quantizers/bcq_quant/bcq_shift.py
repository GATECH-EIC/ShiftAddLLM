import time
import torch
import numpy as np
import gc
from tqdm import tqdm

def round(x, rounding='deterministic'):
    assert(rounding in ['deterministic', 'stochastic'])
    if rounding == 'stochastic':
        x_floor = x.floor()
        return x_floor + torch.bernoulli(x - x_floor)
    else:
        return x.round()

def get_shift_and_sign(x, rounding='deterministic'):
    sign = torch.sign(x)
    
    x_abs = torch.abs(x)
    if rounding == "floor":
        shift = torch.floor(torch.log(x_abs) / np.log(2))
    else:
        shift = round(torch.log(x_abs) / np.log(2), rounding)
    
    return shift, sign    

def round_power_of_2(x, rounding='deterministic', q_bias=None, scale=None):
    if q_bias is not None:
        q_bias = q_bias.unsqueeze(1).expand_as(x)
        x = x - q_bias
    if scale is not None:
        scale = scale.unsqueeze(1).expand_as(x)
        x = x / scale
    shift, sign = get_shift_and_sign(x, rounding)    
    x_rounded = (2.0 ** shift) * sign
    if scale is not None:
        x_rounded = x_rounded * scale
    if q_bias is not None:
        x_rounded = x_rounded + q_bias
    return x_rounded

def additive_power_of_2(x, nums = 2):
    shift, sign = get_shift_and_sign(x, rounding = "deterministic")
    x_rounded = (2.0 ** shift) * sign
    for _ in range(nums - 1):
        x_diff = x - x_rounded
        shift, sign = get_shift_and_sign(x_diff, rounding = "deterministic")
        x_rounded += (2.0 ** shift) * sign
    return x_rounded


def get_best_bias(x):
    x = torch.abs(x)
    assert len(x.shape) == 2, f'Weight shape should be [num, groupsize], but get: {x.shape}'
    quanted_x = round_power_of_2(x)
    bias = torch.mean(x, dim=1) - torch.mean(quanted_x, dim=1)
    return bias

def get_best_scale(x):
    x = torch.abs(x)
    assert len(x.shape) == 2, f'Weight shape should be [num, groupsize], but get: {x.shape}'
    quanted_x = round_power_of_2(x)
    scale = torch.max(x, dim=1)[0] / torch.max(quanted_x, dim=1)[0]
    # scale = torch.norm(x, dim=1) / torch.norm(quanted_x, dim=1)
    if torch.isnan(scale).any():
        # print(torch.max(x, dim=1))
        # print(torch.max(quanted_x, dim=1))
        scale[torch.isnan(scale)] = 1.0
    return scale

def get_best_scale_bias(x):
    x = torch.abs(x)
    assert len(x.shape) == 2, f'Weight shape should be [num, groupsize], but get: {x.shape}'
    quanted_x = round_power_of_2(x)
    scale = torch.max(x, dim=1)[0] / torch.max(quanted_x, dim=1)[0]
    if torch.isnan(scale).any():
        scale[torch.isnan(scale)] = 1.0
    # bias = torch.mean(x, dim=1) - torch.mean(quanted_x, dim=1) * scale
    bias = torch.mean(x, dim=1) - torch.mean(quanted_x, dim=1) 
    return scale, bias

@torch.inference_mode()
def quantize_shift(w, qbits, rounds=15, group_size=-1, transpose=False, use_bst=True, apot_nums = 3):
    '''
    rounds == 0: greedy algorithm
    rounds == 1: refined greedy algorithm
    rounds >= 2: alternating algorithm

    :param w: a weight tensor of layer
    :param qbits: number of quantization bits for the `w`
    :param rounds: number of iterations for refining both alpha and B
    :param group_size: number of weights in which a scaling factor can be shared
    :param transpose: if `transpose` is True, `w` is a transposed when using this method.
    :param use_bst: if `use_bst` is True(default), the binary matrix is calculated using BST algorithm.
                    if `use_bst` is False, the binary matrix is calculated with greedy algorithm.
    :param apot_nums: number of additive-shift weight for quantization.
    '''
    # w_ = w.clone()
    w_ = w
    w_ = w_.cuda()

    if transpose:
        assert len(w_.shape) == 2, f'Check your weight shape {w_.shape}'
        w_ = w_.transpose(1, 0).contiguous()
    
    orig_shape = w_.shape
    group_size = group_size if group_size > 0 else orig_shape[-1]
    w_ = w_.view([-1, group_size])

    wf = torch.ones(w_.shape, dtype=torch.float32, device=w_.device)    
    ret, B, alpha = greedy_mean_torch(w_, n_bits=qbits, wf=wf)
    if rounds > 0 and qbits > 1:
        for _ in range(rounds):
            ret, B, alpha = refine_mean_torch(w_, ret, B, alpha, wf=wf, use_bst=use_bst, apot_nums=apot_nums)

    ret = torch.einsum('ijl,il->ij', (B, alpha))
    ret = ret.view(orig_shape) 
    if transpose:
        ret = ret.transpose(1, 0).contiguous()

    del w_

    B = B.reshape([orig_shape[0], orig_shape[1] // group_size, group_size, qbits])
    alpha = alpha.reshape([orig_shape[0], orig_shape[1] // group_size, qbits])

    B = B.to('cpu')
    # alpha = alpha.to('cpu')

    torch.cuda.empty_cache()
    return ret, B, alpha

def greedy_mean_torch(w, n_bits=1, wf=None, q_bias=None, scale=None, shift = False):
    B = torch.zeros(w.shape + (n_bits,), device=w.device, dtype=w.dtype)
    Alpha = torch.zeros(w.shape[0], n_bits, device=w.device, dtype=w.dtype)
  
    r, w_hat = w.clone(), 0.
    for i in range(n_bits):
        b = r.sign()
        
        if wf is not None:
            # a1sum = torch.sum(wf, dim=1)
            alpha = (r.abs()*wf).sum(dim=1) / torch.sum(wf, dim=1)
            # alpha = (r.abs()*wf).max(dim=1)[0] / 2
            alpha[torch.isnan(alpha)] = 0.
            alpha = alpha.view(alpha.shape[0], 1)
            if shift:
                alpha = round_power_of_2(alpha, q_bias=q_bias, scale=scale)
        else:
            alpha = r.abs().mean(dim=1, keepdim=True)
            if shift:
                alpha = round_power_of_2(alpha, q_bias=q_bias, scale=scale)
        
        r -= b * alpha
        w_hat += b * alpha
        B[:,:,i] = b
        Alpha[:,i] = alpha.view(-1)
    
    # del r, b, alpha
    # gc.collect()
    # torch.cuda.empty_cache()

    return w_hat, B, Alpha

def refine_mean_torch(w, w_hat, B, Alpha, wf=None, use_bst=True, apot_nums=1):
    w = w.float()
    d1, d2 = w.shape
    with torch.no_grad():
        n_bits = B.shape[-1]
        Bt = B.transpose(1, 2)
        if wf is not None:
            Bt = Bt * wf.unsqueeze(1)
        B_cov = Bt.bmm(B)
        Btw = Bt.bmm(w.unsqueeze(-1)).view(d1, n_bits)

        Alpha_new = batch_cg_torch(B_cov, Btw, x=Alpha)
        Alpha_new, _ = Alpha_new.abs().sort(descending=True)

        Alpha_new = additive_power_of_2(Alpha_new, nums=apot_nums)

        torch.cuda.empty_cache()
        if use_bst == False:
            r = w.clone()
            B_new = torch.zeros_like(B)
            for i in range(n_bits):
                B_new[:, :, i] = r.sign()
                r -= B_new[:, :, i] * Alpha_new[:, i].view([-1, 1])
            del r
        else:
            B_new = find_B_torch(w, Alpha_new)
            B_new = B_new * (wf != 0.0).unsqueeze(-1)
        w_hat_new = torch.einsum('ijl,il->ij', (B_new, Alpha_new))

    return w_hat_new, B_new, Alpha_new

def list_binary_vecs(n):
    ListBinaryVecs = {0 : [[]]}
    for m in range(1, n+1):
        ListBinaryVecs[m] = [[1.] + l for l in ListBinaryVecs[m-1]] + [[-1.] + l for l in ListBinaryVecs[m-1]]
    return ListBinaryVecs

def find_B_torch(w, Alpha):
    '''Find optimal quantization assignment via binary search (torch)'''
    n_bits = Alpha.shape[-1]

    ListBinaryVecs = list_binary_vecs(n_bits)
    bin_mat = torch.from_numpy(np.vstack(ListBinaryVecs[n_bits]).astype(np.float32)).to(w.device)

    d1, d2 = w.shape
    row_inds = torch.arange(d1, dtype=torch.long).view(d1, 1).repeat([1, d2]).view(-1)
    # w is d1xd2, Alpha is d1xk, v is d1x2^k
    v = Alpha.mm(bin_mat.t())
    v_sorted, inds = torch.sort(v)
    # Binary search to find nearest neighbor
    w_flat = w.view([-1])
    Left = torch.zeros(d1*d2, dtype=torch.int, device=w.device)
    Right = torch.ones(d1*d2, dtype=torch.int, device=w.device) * (2 ** n_bits - 1)
    for i in range(n_bits):
        Mid_Left = torch.div(Left + Right - 1, 2, rounding_mode='trunc')
        Mid_Right = Mid_Left + 1
        mid_vals = (v_sorted[row_inds, Mid_Left] + v_sorted[row_inds, Mid_Right]) / 2
        inds_left = (w_flat < mid_vals)
        Right[inds_left] = Mid_Left[inds_left]
        Left[~inds_left] = Mid_Right[~inds_left]
    assignment_inds = inds[row_inds, Left].view(d1, d2)
    return bin_mat[assignment_inds, :]

def batch_cg_torch(A, b, x=None):
    '''Batch conjugate gradient for solving Ax = b'''
    d1, k, _ = A.shape
    # Initialize
    x = x.clone().view(d1, k, 1)
    b = b.view(d1, k, 1)
    r = b - A.bmm(x)
    rtr_new = r.transpose(1, 2).bmm(r)
    p = r.clone()
    # Perform batch CG
    for i in range(k):
        rtr = rtr_new
        Ap = A.bmm(p)
        alpha = rtr / (p.transpose(1, 2).bmm(Ap) + 1e-6)
        x += alpha * p
        r -= alpha * Ap
        rtr_new = r.transpose(1, 2).bmm(r)
        beta = rtr_new / (rtr + 1e-6)
        p = r + beta * p
    return x.view(d1, k)
