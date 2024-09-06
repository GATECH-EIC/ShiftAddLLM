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
from lut_gemm.kernel import load_shiftaddllm_weight


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
            
            # Handle attention_mask
            if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                cache['attention_mask'] = torch.as_tensor(kwargs['attention_mask']).to(dev)
            else:
                cache['attention_mask'] = None
            
            # Handle position_ids
            if 'position_ids' in kwargs and kwargs['position_ids'] is not None:
                cache['position_ids'] = torch.as_tensor(kwargs['position_ids']).to(dev)
            else:
                cache['position_ids'] = None
            
            # Move all other kwargs to the device as well
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    kwargs[k] = v.to(dev)
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
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]
        for names in sequential:
            subset = {n: full[n] for n in names}

        quant_method = {}
        for name in subset:
            if args.gptq or args.lut_eval:
                quant_method[name] = GPTQ(subset[name])
            else:
                quant_method[name] = ShiftAddLLM(subset[name])

            if quant_config_dict is not None:
                wbits = quant_config_dict['model.layers.%d.%s' % (i, name)]["bits"]
            else:
                wbits = args.wbits

            if args.gptq:
                quant_method[name].quantizer = Quantizer()
                quant_method[name].quantizer.configure(
                    wbits, perchannel=True, sym=args.sym, mse=False, trits=args.trits
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
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, cache_position = position_ids.squeeze())[0]
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
            
            quant_method[name].fasterquant(
                args, model_name=str(args.model).split("/")[-1], layer_name=f"{i}.{name}"
            )

            quantizers['model.layers.%d.%s' % (i, name)] = quant_method[name].quantizer
            quant_method[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, cache_position = position_ids.squeeze())[0]

        layers[i] = layer.cpu()
        del layer
        del quant_method 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    model = model.to(dev)

    for module in model.modules():
        module.to(dev)

    testenc = testenc.to(dev)
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
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
            
            # Handle attention_mask
            if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                cache['attention_mask'] = torch.as_tensor(kwargs['attention_mask']).to(dev)
            else:
                cache['attention_mask'] = None
            
            # Handle position_ids
            if 'position_ids' in kwargs and kwargs['position_ids'] is not None:
                cache['position_ids'] = torch.as_tensor(kwargs['position_ids']).to(dev)
            else:
                cache['position_ids'] = None
            
            # Move all other kwargs to the device as well
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    kwargs[k] = v.to(dev)
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
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, cache_position = position_ids.squeeze())[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
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

def llama_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model

def override_rotary_emb(model):
    original_forward = model.model.embed_tokens.rotary_emb.forward
    
    def new_forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        return original_forward(x.to(DEV), seq_len)
    
    model.model.embed_tokens.rotary_emb.forward = types.MethodType(new_forward, model.model.embed_tokens.rotary_emb)


if __name__ == '__main__':
    from datautils import *
    args = parse_args()

    if args.temp_storage is not None:
        os.makedirs(args.temp_storage, exist_ok=True)

    model = get_llama(args.model)
    if args.load:
        model.load_state_dict(torch.load(args.load))
    model.eval()
    print(model)

    if args.load_temp_storage is not None:
        assert args.block_quant, "temp_storage only work for blockwise (i.e lat. method) quantization"
        load_shiftaddllm_weight(model, args.load_temp_storage, model_name=str(args.model).split("/")[-1],
                                wbits=args.wbits, groupsize=args.groupsize)

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    
    if args.wbits < 16 and not args.nearest and not args.load:
        tick = time.time()
        if args.bcq:
            print("quantizing with bcq")
            model = quant_model(model, qbits=args.wbits, group_size=args.groupsize)
        else:
            quantizers = llama_sequential(model, dataloader, DEV)
        print("full quantization time: ",time.time() - tick)
    
    if args.save:
        #llama_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save)
    
    # Call this function before evaluation
    # print("ensuring model is on the right device")
    # override_rotary_emb(model)

    datasets = ['wikitext2', 'ptb'] 
    if args.new_eval:
        datasets = ['wikitext2', 'ptb-new', 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        llama_eval(model, testloader, DEV)
