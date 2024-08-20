import time
import math
import bisect

import torch
import torch.nn as nn
import numpy as np
import json

from quant_methods.shiftaddllm import ShiftAddLLM
from modelutils import find_layers, DEV

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')
    print('Analysing ...')

    analysis = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        sequential = [list(full.keys())]
        for names in sequential:
            subset = {n: full[n] for n in names}

        shiftaddllm = {}
        for name in subset:
            shiftaddllm[name] = ShiftAddLLM(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                shiftaddllm[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, cache_position = position_ids.squeeze())[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(" ====== ",i, name, " ====== ")
            start_time = time.time()
            result = shiftaddllm[name].analyse(args.percdamp)
            end_time = time.time()
            print("Analysis Time: ", end_time - start_time)
            analysis['model.layers.%d.%s' % (i, name)] = result
            shiftaddllm[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, cache_position = position_ids.squeeze())[0]

        layers[i] = layer.cpu()
        del layer
        del shiftaddllm 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return analysis

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        '--dataset', type=str, default="wikitext2",choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
	)
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )

    parser.add_argument(
        '--load_weights', type=str, default=None,
        help='weights file to load. None for no weights.'
    )
    parser.add_argument(
        '--load_safetensor', action='store_true', default=True,
        help='Whether to load from a safetensor file.'
    )

    parser.add_argument(
        '--load_analyse_result', action='store_true', default=False,
        help='Whether to load the analyse middle result from .pth at default path.'
    )
    parser.add_argument(
        '--mix_bits', type=str, default="0.0,0.8,0.2",
        help=f'The proportion of each bits. index=0 for 1bits quant, index=1 for 2bits quant. as default, 80% 2bit and 20% 3bit'
    )
    args = parser.parse_args()

    args.mix_bits = str(args.mix_bits).split(",")
    args.mix_bits = [float(each) for each in args.mix_bits]
    assert sum(args.mix_bits) == 1, "The sum of the ratios of the mixed bits should be 1"
    # get cumsum
    args.mix_bits = np.cumsum(np.array(args.mix_bits)).tolist()
    print(args.mix_bits)

    if not args.load_analyse_result:
        model = get_llama(args.model)
        model.eval()
        print(model)

        if args.load_weights is not None:
            if args.load_safetensor:
                from safetensors import safe_open
                ckpt_weights = {}
                with safe_open(args.load_weights, framework="pt", device='cpu') as f:
                    for k in f.keys():
                        ckpt_weights[k] = f.get_tensor(k)
                model.load_state_dict(ckpt_weights, strict=False)
            else:
                model.load_state_dict(torch.load(args.load_weights), strict=True)

        dataloader, testloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
        )

        analysis_result = llama_sequential(model, dataloader, DEV)
        model_name = str(args.model).split("/")[-1]
        torch.save(analysis_result, f"./sensitivity/mixbit/{model_name}.pth")
    else:
        model_name = str(args.model).split("/")[-1]
        analysis_result = torch.load(f"./sensitivity/mixbit/{model_name}.pth")

    model_quant_config = {}
    weight_score = []
    layers = []
    for each in analysis_result.keys():
        if "fc" in each:
            layers.extend(each for _ in range(4))
        else:
            layers.append(each)
    

    for each in layers:
        layer_quant_config = {
            "bits": 2,
            "columnwise": True,
        }
        layer_norm = analysis_result[each]['norm']["wh"]
        layer_std = analysis_result[each]["std"]["wh"]
        weight_score.append(layer_norm * layer_std ** 2)
        model_quant_config[each] = layer_quant_config

    _, weight_score_index = torch.sort(torch.tensor(weight_score))

    mix_bits = [each *  len(weight_score_index) for each in args.mix_bits]
    for i, each in enumerate(weight_score_index):
        model_quant_config[layers[each]]["bits"] = bisect.bisect(mix_bits, i) + 1

    print(model_quant_config)

    # Statistical mixed bits averages
    count = bits_sum = 0
    if "Llama-3" in model_name:
        ffn_ratio = 3.5
    else:
        ffn_ratio = 2.7
    for each in model_quant_config.keys():
        if "gate" in each or "up_proj" in each or "down_proj" in each:
            count += ffn_ratio
            bits_sum += ffn_ratio * model_quant_config[each]["bits"]
        else:
            count += 1
            bits_sum += model_quant_config[each]["bits"]
    print(f"mix bits avg: {bits_sum/count} bits")

    with open(f"./sensitivity/mixbit/{model_name}.json", "w") as f:
        json.dump(model_quant_config, f)