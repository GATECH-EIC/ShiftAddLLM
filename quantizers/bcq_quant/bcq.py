import time
import torch
import numpy as np
import gc
from tqdm import tqdm

@torch.inference_mode()
def quantize(w, qbits, rounds=15, group_size=-1, transpose=False, exponent=0.0, clipping=1.0, pruning=0.0, use_bst=True):
    '''
    Post-training Weighted Quantization (BCQ format)
    https://openreview.net/pdf?id=2Id6XxTjz7c

    rounds == 0: greedy algorithm
    rounds == 1: refined greedy algorithm
    rounds >= 2: alternating algorithm

    :param w: a weight tensor of layer
    :param qbits: number of quantization bits for the `w`
    :param rounds: number of iterations for refining both alpha and B
    :param group_size: number of weights in which a scaling factor can be shared
    :param transpose: if `transpose` is True, `w` is a transposed when using this method.
    :param exponent: the exponent term of weighted factor.
                     if `exponent` is zero, this method is exactly the same as conventional BCQ method.
    :param clipping: the clipping importance term(0 <= clipping <= 1) of weighted factor.
    :param pruning: the pruning ratio(0 <= pruning <= 1) of weighted factor.
    :param use_bst: if `use_bst` is True(default), the binary matrix is calculated using BST algorithm.
                    if `use_bst` is False, the binary matrix is calculated with greedy algorithm.
    '''
    w_ = w.clone().float()
    w_ = w_.cuda()

    if transpose:
        assert len(w_.shape) == 2, f'Check your weight shape {w_.shape}'
        w_ = w_.transpose(1, 0).contiguous()
    
    orig_shape = w_.shape
    group_size = group_size if group_size > 0 else orig_shape[-1]
    w_ = w_.view([-1, group_size])
 
    # init weighted
    w_abs = w_.abs()
    ws, _ = w_abs.view(-1).sort()
    wf = torch.ones(w_.shape, dtype=torch.float32, device=w.device)
    if pruning > 0.0:
        wf = wf * (w_ != 0.0)
    if exponent > 0.0 or clipping < 1.0:
        wf = w_abs / w_abs.max()
    # weighted factor for C
    if clipping < 1.0:
        c_th = ws[int(ws.size(0) * clipping)].item()
        wf = wf * w_abs.max() / c_th
        wf[wf > 1.0] = 1.0
    # weighted factor for E
    if exponent > 0.0:
        wf = wf ** exponent
    # weighted factor for P
    if pruning > 0.0:
        p_th = ws[int(ws.shape[0] * pruning)].item()
        wf[w_abs <= p_th] = 0.0
        w_[w_abs <= p_th] = 0.0

    wf = wf.to(w_.device)
    # greedy & alternating algo.
    ret, B, alpha = greedy_mean_torch(w_, n_bits=qbits, wf=wf)
    if rounds > 0 and qbits > 1:
        # for _ in range(rounds):
        for _ in range(rounds):
            ret, B, alpha = refine_mean_torch(w_, ret, B, alpha, wf=wf, use_bst=use_bst)

    ret = ret.view(orig_shape) 
    if transpose:
        ret = ret.transpose(1, 0).contiguous()

    del w_
    
    B = B.reshape([orig_shape[0], orig_shape[1] // group_size, group_size, qbits])
    alpha = alpha.reshape([orig_shape[0], orig_shape[1] // group_size, qbits])

    # B = B.to('cpu')
    # alpha = alpha.to('cpu')
    # torch.cuda.empty_cache()

    return ret, B, alpha, (wf != 0.0)

def greedy_mean_torch(w, n_bits=1, wf=None):
    B = torch.zeros(w.shape + (n_bits,), device=w.device)
    Alpha = torch.zeros(w.shape[0], n_bits, device=w.device)
  
    r, w_hat = w.clone(), 0.
    for i in range(n_bits):
        b = r.sign()
        
        if wf is not None:
            a1sum = torch.sum(wf, dim=1)
            alpha = (r.abs()*wf).sum(dim=1) / torch.sum(wf, dim=1)
            alpha[torch.isnan(alpha)] = 0.
            alpha = alpha.view(alpha.shape[0], 1)
        else:
            alpha = r.abs().mean(dim=1, keepdim=True)
        
        r -= b * alpha
        w_hat += b * alpha
        B[:,:,i] = b
        Alpha[:,i] = alpha.view(-1)
    
    # del r, b, alpha
    # gc.collect()
    # torch.cuda.empty_cache()

    return w_hat, B, Alpha

def refine_mean_torch(w, w_hat, B, Alpha, wf=None, use_bst=True):
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
    Left = torch.zeros(d1*d2, dtype=torch.long, device=w.device)
    Right = torch.ones(d1*d2, dtype=torch.long, device=w.device) * (2 ** n_bits - 1)
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
