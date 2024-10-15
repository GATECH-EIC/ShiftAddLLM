import sys
import time
import math
sys.path.append('.')
sys.path.append('..')

import torch
import torch.nn as nn

from quant_methods.gptq import *
from quant_methods.shiftaddllm import *
from modelutils import *
from parsers import parse_args

from quantizers.quant import *
from quant_methods.quant_model_bcq import quant_model
from quantizers.bcq_quant.quantizer import BCQuantizer
from lut_gemm.kernel import load_shiftaddllm_weight, load_shiftaddllm_weight_with_kernel

def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model

@torch.no_grad()
def opt_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
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
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    quant_config_dict = None
    if args.quant_config:
        import json
        with open(args.quant_config, "r") as f:
            quant_config_dict = json.load(f)
        print(f"quant_config: {quant_config_dict}")

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        quant_method = {}
        for name in subset:
            if args.gptq or args.lut_eval:
                quant_method[name] = GPTQ(subset[name])
            else:
                quant_method[name] = ShiftAddLLM(subset[name])

            if quant_config_dict is not None:
                wbits = quant_config_dict['model.decoder.layers.%d.%s' % (i, name)]["bits"]
            else:
                wbits = args.wbits

            if args.gptq:
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False, trits=args.trits
                )
            else:
                quant_method[name].quantizer = BCQuantizer(subset[name].weight.data.size(),
                                                    groupsize=args.groupsize, 
                                                    wbits=wbits,
                                                    rounds=args.bcq_round,
                                                    use_bst=args.use_bst, 
                                                    apot_nums=args.apot_nums)

        def add_batch(name):
            def tmp(_, inp, out):
                quant_method[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()
        for name in subset:
            quant_method[name].post_batch()

        for name in subset:
            print(" ====== ",i, name, " ====== ")
            quant_method[name].preproc(
                    preproc_gptqH=args.pre_gptqH, percdamp=args.percdamp,
                    preproc_rescale=args.pre_rescale, 
                    preproc_proj=args.pre_proj, preproc_proj_extra=args.pre_proj_extra)
            
            if quant_config_dict is not None:
                is_column = quant_config_dict['model.decoder.layers.%d.%s' % (i, name)]["columnwise"]
            # if "fc1" in name or "fc2" in name or "out_proj" in name:
                if not is_column:
                    args.columnwise = False
                    args.block_quant = False
                    args.lut_eval = True
                else:
                    if args.acc:
                        args.columnwise = True
                        args.block_quant = False
                    elif args.lat:
                        args.block_quant = True
                        args.columnwise = False
                    args.lut_eval = False

            quant_method[name].fasterquant(
                args, model_name=str(args.model).split("/")[-1], layer_name=f"{i}.{name}"
            )
            quantizers['model.decoder.layers.%d.%s' % (i, name)] = quant_method[name].quantizer
            quant_method[name].free()
        
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del quant_method 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

@torch.no_grad()
def opt_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
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
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        # print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        # plot_distribution(outs, file_path=f"./plot_activation/layer_{i}.png")
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

# TODO: perform packing on GPU
def opt_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers, faster=args.faster_kernel)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model

def load_quant3(model, checkpoint):
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
    make_quant3(model, layers, faster=args.faster_kernel)

    print('Loading model ...')
    model.load_state_dict(torch.load(checkpoint))
    model.seqlen = model.config.max_position_embeddings
    print('Done.')

    return model

def opt_multigpu(model, gpus):
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(gpus[0])
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(gpus[0])
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(gpus[0])
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(gpus[-1])
    if hasattr(model.model.decoder, 'final_layer_norm') and model.model.decoder.final_layer_norm:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(gpus[-1])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[-1])

    cache = {'mask': None}

    class MoveModule(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if cache['mask'] is None or cache['mask'].device != self.dev:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.decoder.layers
    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers)):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))

    model.gpus = gpus

@torch.no_grad()
def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV) # [1,128]
    torch.cuda.synchronize()

    cache = {'past': None}
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    for i, layer in enumerate(model.model.decoder.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()
            

    attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
    times = []

    for _ in range(100): #warmup
        out = model(
            input_ids[:, 0].reshape((1,-1)),
            past_key_values=cache['past'],
            attention_mask=attention_mask[:, :1].reshape((1, -1))
        )
        sync()

    for i in range(input_ids.numel()):
        tick = time.time()
        out = model(
            input_ids[:, i].reshape((1,-1)),
            past_key_values=cache['past'],
            attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1))
        )
        sync()
        times.append(time.time() - tick)
        # print(i, times[-1])
        if check and i != input_ids.numel() - 1:
            tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
        cache['past'] = list(out.past_key_values)
        del out
    sync()
    import numpy as np
    print('Median:', np.median(times))
    if check:
        print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())


if __name__ == '__main__':
    from datautils import *
    args = parse_args()
    
    if args.temp_storage is not None:
        os.makedirs(args.temp_storage, exist_ok=True)

    model = get_opt(args.model)
    if args.load:
        model.load_state_dict(torch.load(args.load))
    model.eval()
    print(model)

    if args.infer_kernel and args.load_temp_storage is not None:  # load quantized weight and infer with cuda kernel
        assert args.block_quant, "quantized weight only work for blockwise (i.e lat. method) quantization"
        load_shiftaddllm_weight_with_kernel(model, args.load_temp_storage, model_name=str(args.model).split("/")[-1],
                                wbits=args.wbits, is_lat=args.lat)

    elif args.load_temp_storage is not None: # load quantized weight and infer with Fp16
        assert args.block_quant, "temp_storage only work for blockwise (i.e lat. method) quantization"
        load_shiftaddllm_weight(model, args.load_temp_storage, model_name=str(args.model).split("/")[-1],
                                wbits=args.wbits, is_lat=args.lat)



    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    
    if args.wbits < 16 and not args.nearest and args.load_temp_storage is None:
        tick = time.time()
        if args.bcq:
            print("quantizing with bcq")
            model = quant_model(model, qbits=args.wbits, group_size=args.groupsize, rounds=args.bcq_round)
        else:
            quantizers = opt_sequential(model, dataloader, DEV)
        print("full quantization time: ",time.time() - tick)

    if args.benchmark:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            opt_multigpu(model, gpus)
        else:
            model = model.to(DEV)
        
        input_ids = next(iter(dataloader))[0][:, :args.benchmark]
        print(input_ids.shape)
        benchmark(model, input_ids, check=args.check)
        exit()
            
    if args.save:
        torch.save(model.state_dict(), args.save)
        
    datasets = ['wikitext2', 'ptb'] 
    if args.new_eval:
        datasets = ['wikitext2', 'ptb-new', 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV)
