# from rtn_parameter import RTNParameter
from quantizers.bcq_quant.bcq_parameter import BCQParameter


layers = ["q_proj","k_proj","v_proj","out_proj","fc1","fc2","o_proj","gate_proj","up_proj","down_proj"]

def quant_model(model, qbits:int = 4, group_size:int = 128, rounds=50):
    parameters  = model.state_dict()
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            quant_model(module, qbits, group_size)

        if any(x in name for x in layers):
        # if True:
            print(name)
            original_weight = module.weight.clone().detach()
            # INT4 Quantization -> BCQ
            w_bcq = BCQParameter(original_weight)
            alpha, binary, binary_shape = w_bcq.compress(
                do_packing=False, in_ch_wise=True, qbits=qbits,
                rounds=rounds, group_size=group_size)
            w_bcq.decompress(alpha, binary, binary_shape, do_packing=False, in_ch_wise=True)
            parameters[name + ".weight"] = w_bcq.data.clone().detach()
    model.load_state_dict(parameters)

    return model
