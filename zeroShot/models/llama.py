import transformers
import torch
from .models_utils import BaseLM, find_layers
from transformers import LlamaForCausalLM, AutoTokenizer
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
from .quant import *
from .gptq import GPTQ


class LlamaClass(BaseLM):
		def __init__(self, args):

				super().__init__()

				self.args = args
				self._device = torch.device("cuda")
				self.model_name = args.model
				self.batch_size_per_gpu = args.batch_size

				self.model = LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype='auto')
				# self.seqlen = self.model.config.max_position_embeddings
				self.seqlen = 2048
				self.model.eval()

				# pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
				self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
				self.vocab_size = self.tokenizer.vocab_size
				print('Llama vocab size: ', self.vocab_size)

		@property
		def eot_token_id(self):
				# we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
				return self.tokenizer.eos_token_id

		@property
		def max_length(self):
				return 2048
		@property
		def max_gen_toks(self):
				print('max_gen_toks fn')
				return 256

		@property
		def batch_size(self):
				# TODO: fix multi-gpu
				return self.batch_size_per_gpu  # * gpus

		@property
		def device(self):
				# TODO: fix multi-gpu
				return self._device

		def tok_encode(self, string: str):
				return self.tokenizer.encode(string)

		def tok_decode(self, tokens):
				return self.tokenizer.decode(tokens)

		def _model_call(self, inps):
				"""
				inps: a torch tensor of shape [batch, sequence]
				the size of sequence may vary from call to call
				returns: a torch tensor of shape [batch, sequence, vocab] with the
				logits returned from the model
				"""
				with torch.no_grad():
						return self.model(inps)[0][:, :, :128001]

		@torch.no_grad()
		def _model_logits_on_dataset(self, dataset_inps):
				print('Evaluating ...')

				nsamples = len(dataset_inps)
				print("Number of samples: ", nsamples)

				model = self.model
				dev = self.device

				use_cache = model.config.use_cache
				model.config.use_cache = False
				layers = model.model.layers

				model.model.embed_tokens = model.model.embed_tokens.to(dev)
				layers[0] = layers[0].to(dev)

				dtype = next(iter(model.parameters())).dtype
				inps = []
				outs = []
				for batch_idx, batch in enumerate(dataset_inps):
						inps.append(torch.zeros(
								(batch.shape[1], self.model.config.hidden_size), dtype=dtype,
						))
						outs.append(torch.zeros(
								(batch.shape[1], self.model.config.hidden_size), dtype=dtype,
						))

				cache = {'i': 0, 'attention_masks': [], 'position_ids': []}

				class Catcher(nn.Module):
						def __init__(self, module):
								super().__init__()
								self.module = module

						def forward(self, inp, **kwargs):
								inps[cache['i']] = inp
								cache['i'] += 1
								cache['attention_masks'].append(kwargs['attention_mask'].detach().cpu())
								cache['position_ids'].append(kwargs['position_ids'].detach().cpu())
								raise ValueError

				layers[0] = Catcher(layers[0])
				for i in tqdm(range(nsamples)):
						batch = dataset_inps[i].to(dev)
						try:
								model(batch)
						except ValueError:
								pass
				layers[0] = layers[0].module

				layers[0] = layers[0].cpu()
				model.model.embed_tokens = model.model.embed_tokens.cpu()
				torch.cuda.empty_cache()

				attention_mask = cache['attention_masks']
				position_ids = cache['position_ids']

				for i in tqdm(range(len(layers))):
					layer = layers[i].to(dev)
					
					if self.args.nearest:
							subset = find_layers(layer)
							for name in subset:
									quantizer = Quantizer()
									quantizer.configure(
											self.args.wbits, perchannel=True, sym=False, mse=False
									)
									W = subset[name].weight.data
									quantizer.find_params(W, weight=True)
									subset[name].weight.data = quantize(
											W, quantizer.scale, quantizer.zero, quantizer.maxq
									).to(next(iter(layer.parameters())).dtype)

					for j in range(nsamples):
							outs[j] = layer(inps[j].to(self.device), attention_mask=attention_mask[j].to(self.device), position_ids=position_ids[j].to(self.device), cache_position=position_ids[j].squeeze().to(self.device))[0].detach().cpu()
					layers[i] = layer.cpu()
					del layer
					torch.cuda.empty_cache()
					inps, outs = outs, inps

				if model.model.norm is not None:
						model.model.norm = model.model.norm.to(dev)
				model.lm_head = model.lm_head.to(dev)

				if self.model.model.norm is not None:
						self.model.model.norm = self.model.model.norm.to(dev)
				self.model.lm_head = self.model.lm_head.to(dev)

				dataset_logits = []

				for i in tqdm(range(nsamples), desc='Last Layer'):
						hidden_states = inps[i].unsqueeze(0).to(self.device)
						if self.model.model.norm is not None:
								hidden_states = self.model.model.norm(hidden_states)
						batch_logits = F.log_softmax(self.model.lm_head(hidden_states)[0][:, :, :128001], dim=-1).cpu()
						dataset_logits.append(batch_logits)
				model.config.use_cache = use_cache
				return dataset_logits

		def _model_generate(self, context, max_length, eos_token_id):
				return self.model.generate(
						context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
				)

# for backwards compatibility
Llama = LlamaClass