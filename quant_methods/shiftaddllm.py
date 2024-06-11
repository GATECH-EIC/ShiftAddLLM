import math
import time
import os

from tqdm import tqdm
import torch
import torch.nn as nn
import transformers

from quantizers.quant import *

from quantizers.bcq_quant.quantizer import quantize as bcq_quantize
from quantizers.bcq_quant.bcq_shift import quantize_shift
from .quip_method import QuantMethod
from lut_gemm.kernel import pack_binaryWeight, unpack_weight

DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class ShiftAddLLM(QuantMethod):

	def analyse(self, percdamp=0.01):
			result = {
					"rowwise": {"wh" : { "max": None, "min": None, "mean": None, "std": None}}, # analysis of weight / hessian
					"columnWise": {"wh" : { "max": None, "min": None, "mean": None, "std": None}}, # analysis of weight / hessian
					"mean": {"wh": None}, 
					"std": {"wh": None},
					"max": {"wh": None}, 
					"min": {"wh": None},
					"norm": {"wh": None}, 
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

			return result

	def fasterquant(
			self, args, model_name, layer_name
	):  
		temp_storage_pt = ""
		parent_dir = ""
		if args.temp_storage is not None:
			assert args.block_quant, "temp_storage only work for blockwise (i.e lat. method) quantization"
			parent_dir = os.path.join(args.temp_storage, f"{model_name}")
			os.makedirs(parent_dir, exist_ok=True)
			temp_storage_pt = os.path.join(parent_dir, f"{layer_name}_{self.quantizer.wbits}bit_{args.groupsize}groupsize.pt")

		if os.path.exists(temp_storage_pt):
			print(f"load from {temp_storage_pt}")
			checkpoint = torch.load(temp_storage_pt)
			BinaryWeight = checkpoint["bWeight"].to(self.layer.weight.data.device)
			alpha = checkpoint["alpha"].to(self.layer.weight.data.device)
			if args.lat:
				alpha = alpha.repeat_interleave(8, dim=0)
			W = unpack_weight(BinaryWeight, alpha).transpose(0, 1)
			self.layer.weight.data = W.to(self.layer.weight.data.dtype)
			self.postproc()
			return

		W = self.layer.weight.data.clone()
		if isinstance(self.layer, nn.Conv2d):
				W = W.flatten(1)
		if isinstance(self.layer, transformers.Conv1D):
				W = W.t()
		W = W.float()
		N, K = W.size()

		groupsize = args.groupsize
		if args.block_quant: # we set 8 column & 1/8 row for a block, which keeps the same quant param as rowwise.
			if groupsize == -1: # for shfitaddllm, groupsize work columnwise
				groupsize = int(N // 8)
			else:
				groupsize = int(args.groupsize // 8)

		tick = time.time()
		H = self.H

		if args.act_order:
			if args.acc:
				perm = torch.argsort(torch.diag(H), descending=True)
				W = W[:, perm]
				H = H[perm][:, perm]
				invperm = torch.argsort(perm)
			elif args.lat:
				W = W.reshape(N, K//8, 8)
				perm = torch.argsort(torch.diag(H).reshape(-1, 8).sum(-1), descending=True)
				W = W[:, perm, : ]
				W = W.reshape(N, K)
				H = H.reshape(K//8, 8, K//8, 8)
				H = H[perm,:,:,:][:,:,perm,:]
				H = H.reshape(K,K)
				invperm = torch.argsort(perm)

		Losses = torch.zeros_like(W)
		Q = torch.zeros_like(W)

		H = torch.linalg.cholesky(H)
		H = torch.cholesky_inverse(H)
		H = torch.linalg.cholesky(H, upper=True)
		Hinv = H

		BinaryWeight_list = []
		alpha_list = []

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

				if args.block_quant:
					if i % 8 == 0:
						w_8column = W1[:, i:i+8].flatten()
						# Get alpha
						q, BinaryWeight, alpha = quantize_shift(w_8column.unsqueeze(0),
										qbits=self.quantizer.wbits,
										group_size=groupsize * 8 if groupsize != -1 else -1, 
										rounds=self.quantizer.rounds, 
										use_bst=self.quantizer.use_bst, apot_nums=self.quantizer.apot_nums)
						alpha_list.append(alpha.cpu())
					# Get the correct bW
					q, BinaryWeight = bcq_quantize(w.unsqueeze(0), alpha, groupsize=groupsize, use_bst=self.quantizer.use_bst)
					BinaryWeight_list.append(BinaryWeight.cpu())
					q = q.flatten()
				else:
					q, BinaryWeight, alpha = quantize_shift(w.unsqueeze(0),\
									qbits=self.quantizer.wbits, 
									group_size=groupsize, rounds=self.quantizer.rounds, 
									use_bst=self.quantizer.use_bst, apot_nums=self.quantizer.apot_nums)
					q = q.flatten()
					BinaryWeight_list.append(BinaryWeight.cpu())
					alpha_list.append(alpha.cpu())

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

		# pack binary weight
		if temp_storage_pt != "" and args.block_quant:
			BinaryWeight = torch.cat(BinaryWeight_list, dim=0)
			alpha = torch.cat(alpha_list, dim=0)
			if args.act_order:
				invperm = invperm.cpu()
				K, n_group, groupsize, wbits = BinaryWeight.size()
				BinaryWeight = BinaryWeight.reshape(K//8, 8, n_group, groupsize, wbits)
				BinaryWeight = BinaryWeight[invperm, :, :, :, :]
				BinaryWeight = BinaryWeight.reshape(K, n_group, groupsize, wbits)
				alpha = alpha[invperm, :, :]
			BinaryWeight = pack_binaryWeight(BinaryWeight)
			
			if args.block_quant:
				alpha_origin = alpha.repeat_interleave(8, dim=0)
			recover_weight = unpack_weight(BinaryWeight, alpha_origin).transpose(0, 1)
			print("weight pack loss:" , (recover_weight.cuda() - Q).abs().max().item())
			torch.save({
					"bWeight": BinaryWeight,
					"alpha": alpha
			}, temp_storage_pt)

		if args.act_order:
				if args.acc:
					Q = Q[:, invperm]
				elif args.lat:
					Q = Q.reshape(N, K//8, 8)
					Q = Q[:, invperm, : ]
					Q = Q.reshape(N, K)

		if isinstance(self.layer, transformers.Conv1D):
				Q = Q.t()
		self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

		if DEBUG:
				print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

		self.postproc()
