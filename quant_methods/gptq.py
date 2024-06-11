import math
import time
import os

from tqdm import tqdm
import torch
import torch.nn as nn
import transformers

from quantizers.quant import *
from quantizers.bcq_quant.quantizer import quantize as bcq_quantize

from .quip_method import QuantMethod

DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ(QuantMethod):

    def analyse(self, percdamp=0.01):
        result = {
            "rowwise": {"w": {"max": None, "min": None, "mean": None, "std": None}, # analysis of weight
                        "wa" : { "max": None, "min": None, "mean": None, "std": None}, # analysis of weight * activation
                        "wh" : { "max": None, "min": None, "mean": None, "std": None}, # analysis of weight / hessian
                        "wh_origin" : { "max": None, "min": None, "mean": None, "std": None}, # analysis of weight / hessian
                        "wn" : { "max": None, "min": None, "mean": None, "std": None}, # analysis of weight * norm
            },
            "columnWise": {"w": {"max": None, "min": None, "mean": None, "std": None}, # analysis of weight
                        "wa" : { "max": None, "min": None, "mean": None, "std": None}, # analysis of weight * activation
                        "wh" : { "max": None, "min": None, "mean": None, "std": None}, # analysis of weight / hessian
                        "wh_origin" : { "max": None, "min": None, "mean": None, "std": None}, # analysis of weight / hessian
                        "wn" : { "max": None, "min": None, "mean": None, "std": None}, # analysis of weight * norm
            },
            "mean": {"w": None, "wa": None, "wn": None, "wh": None, "wh_origin": None}, # mean of weight, weight * activation, weight * norm, weight / hessian
            "std": {"w": None, "wa": None, "wn": None, "wh": None, "wh_origin": None}, # std of weight, weight * activation, weight * norm, weight / hessian
            "max": {"w": None, "wa": None, "wn": None, "wh": None, "wh_origin": None}, # max of weight, weight * activation, weight * norm, weight / hessian
            "min": {"w": None, "wa": None, "wn": None, "wh": None, "wh_origin": None}, # min of weight, weight * activation, weight * norm, weight / hessian
            "norm": {"w": None, "wa": None, "wn": None, "wh": None, "wh_origin": None}, # norm of weight, weight * activation, weight * norm, weight / hessian
        }
        W = self.layer.weight.data.clone()
        W = W.float()
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)

        # analysis of weight
        result["rowwise"]["w"]["max"] = W.max(dim=1).values
        result["rowwise"]["w"]["min"] = W.min(dim=1).values
        result["rowwise"]["w"]["mean"] = W.mean(dim=1)
        result["rowwise"]["w"]["std"] = W.std(dim=1)
        result["columnWise"]["w"]["max"] = W.max(dim=0).values
        result["columnWise"]["w"]["min"] = W.min(dim=0).values
        result["columnWise"]["w"]["mean"] = W.mean(dim=0)
        result["columnWise"]["w"]["std"] = W.std(dim=0)
        result["mean"]["w"] = W.mean()
        result["std"]["w"] = W.std()
        result["max"]["w"] = W.max()
        result["min"]["w"] = W.min()
        result["norm"]["w"] = torch.norm(W)

        # analysis of weight * activation
        inputMean = torch.mean(self.input, 1)
        weightAct = W * inputMean.repeat(self.rows, 1) 
        result["rowwise"]["wa"]["max"] = weightAct.max(dim=1).values
        result["rowwise"]["wa"]["min"] = weightAct.min(dim=1).values
        result["rowwise"]["wa"]["mean"] = weightAct.mean(dim=1)
        result["rowwise"]["wa"]["std"] = weightAct.std(dim=1)
        result["columnWise"]["wa"]["max"] = weightAct.max(dim=0).values
        result["columnWise"]["wa"]["min"] = weightAct.min(dim=0).values
        result["columnWise"]["wa"]["mean"] = weightAct.mean(dim=0)
        result["columnWise"]["wa"]["std"] = weightAct.std(dim=0)
        result["mean"]["wa"] = weightAct.mean()
        result["std"]["wa"] = weightAct.std()
        result["max"]["wa"] = weightAct.max()
        result["min"]["wa"] = weightAct.min()
        result["norm"]["wa"] = torch.norm(weightAct)

        inputNorm = torch.norm(self.input, dim=1, p=2)
        weightActNorm = W * inputNorm.repeat(self.rows, 1) 
        result["rowwise"]["wn"]["max"] = weightActNorm.max(dim=1).values
        result["rowwise"]["wn"]["min"] = weightActNorm.min(dim=1).values
        result["rowwise"]["wn"]["mean"] = weightActNorm.mean(dim=1)
        result["rowwise"]["wn"]["std"] = weightActNorm.std(dim=1)
        result["columnWise"]["wn"]["max"] = weightActNorm.max(dim=0).values
        result["columnWise"]["wn"]["min"] = weightActNorm.min(dim=0).values
        result["columnWise"]["wn"]["mean"] = weightActNorm.mean(dim=0)
        result["columnWise"]["wn"]["std"] = weightActNorm.std(dim=0)
        result["mean"]["wn"] = weightActNorm.mean()
        result["std"]["wn"] = weightActNorm.std()
        result["max"]["wn"] = weightActNorm.max()
        result["min"]["wn"] = weightActNorm.min()
        result["norm"]["wn"] = torch.norm( torch.abs(W) * inputNorm.repeat(self.rows, 1) )


        weightH = W / torch.diag(H).repeat(self.rows, 1) 
        result["rowwise"]["wh"]["max"] = weightH.max(dim=1).values
        result["rowwise"]["wh"]["min"] = weightH.min(dim=1).values
        result["rowwise"]["wh"]["mean"] = weightH.mean(dim=1)
        result["rowwise"]["wh"]["std"] = weightH.std(dim=1)
        result["columnWise"]["wh"]["max"] = weightH.max(dim=0).values
        result["columnWise"]["wh"]["min"] = weightH.min(dim=0).values
        result["columnWise"]["wh"]["mean"] = weightH.mean(dim=0)
        result["columnWise"]["wh"]["std"] = weightH.std(dim=0)
        result["mean"]["wh"] = weightH.mean()
        result["std"]["wh"] = weightH.std()
        result["max"]["wh"] = weightH.max()
        result["min"]["wh"] = weightH.min()
        result["norm"]["wh"] = torch.norm(torch.abs(W) / torch.diag(H).repeat(self.rows, 1))

        weightH_origin = W ** 2 / torch.diag(torch.matmul(self.input, self.input.T)).repeat(self.rows, 1)
        result["rowwise"]["wh_origin"]["max"] = weightH_origin.max(dim=1).values
        result["rowwise"]["wh_origin"]["min"] = weightH_origin.min(dim=1).values
        result["rowwise"]["wh_origin"]["mean"] = weightH_origin.mean(dim=1)
        result["rowwise"]["wh_origin"]["std"] = weightH_origin.std(dim=1)
        result["columnWise"]["wh_origin"]["max"] = weightH_origin.max(dim=0).values
        result["columnWise"]["wh_origin"]["min"] = weightH_origin.min(dim=0).values
        result["columnWise"]["wh_origin"]["mean"] = weightH_origin.mean(dim=0)
        result["columnWise"]["wh_origin"]["std"] = weightH_origin.std(dim=0)
        result["mean"]["wh_origin"] = weightH_origin.mean()
        result["std"]["wh_origin"] = weightH_origin.std()
        result["max"]["wh_origin"] = weightH_origin.max()
        result["min"]["wh_origin"] = weightH_origin.min()
        result["norm"]["wh_origin"] = torch.norm(weightH_origin)

        return result

    def fasterquant(
        self, args, model_name, layer_name
    ):  
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        groupsize = args.groupsize
        tick = time.time()

        if args.static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if args.act_order:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        H = self.H
        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        for i1 in tqdm(range(0, self.columns, args.blocksize), desc=layer_name, leave=False):
            i2 = min(i1 + args.blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if args.lut_eval:
                    if groupsize != -1:
                        idx = i1 + i
                        if args.act_order:
                            idx = perm[idx]
                        group = idx // groupsize
                    else:
                        group = 0
                    alpha = self.quantizer.alpha[:,group,:].unsqueeze(1)
                    q, BinaryWeight = bcq_quantize(w.unsqueeze(1), alpha, groupsize=-1)
                    q = q.flatten()
                else:
                    if groupsize != -1:
                        if not args.static_groups:
                            if (i1 + i) % groupsize == 0:
                                self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                        else:
                            idx = i1 + i
                            if args.act_order:
                                idx = perm[idx]
                            self.quantizer = groups[idx // groupsize]

                    q = quantize(
                        w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if args.act_order:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        self.postproc()
        self.error_compute(W, self.layer.weight.data)
        
    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
