from tqdm import tqdm
from hw_utils import get_OPs_HW_metric


MODEL_CONFIG = {
    "opt-125m":{
        "type": "opt",
        "dim": 768,
        "final_dim": 50272,
        "num_layer": 12
    },
    "opt-350m":{
        "type": "opt",
        "dim": 1024,
        "final_dim": 50272,
        "num_layer": 24
    },
    "opt-1.3b":{
        "type": "opt",
        "dim": 2048,
        "final_dim": 50272,
        "num_layer": 24
    },
    "opt-2.7b":{
        "type": "opt",
        "dim": 2560,
        "final_dim": 50272,
        "num_layer": 32
    },
    "opt-6.7b":{
        "type": "opt",
        "dim": 4096,
        "final_dim": 50272,
        "num_layer": 32
    },
    "opt-13b":{
        "type": "opt",
        "dim": 5120,
        "final_dim": 50272,
        "num_layer": 40
    },
    "opt-30b":{
        "type": "opt",
        "dim": 7168,
        "final_dim": 50272,
        "num_layer": 48
    },
    "opt-66b":{
        "type": "opt",
        "dim": 9216,
        "final_dim": 50272,
        "num_layer": 64
    },

    "llama-7b":{
        "type": "llama",
        "dim": 4096,
        "fc_dim": 11008,
        "final_dim": 32000,
        "num_layer": 32
    },
    "llama2-7b":{
        "type": "llama",
        "dim": 4096,
        "fc_dim": 11008,
        "final_dim": 32000,
        "num_layer": 32
    },
    "llama3-8b":{
        "type": "llama3",
        "dim": 4096,
        "fc_dim": 14336,
        "final_dim": 128256,
        "num_layer": 32
    },
    "llama-13b":{
        "type": "llama",
        "dim": 5120,
        "fc_dim": 13824,
        "final_dim": 32000,
        "num_layer": 40
    },
    "llama2-13b":{
        "type": "llama",
        "dim": 5120,
        "fc_dim": 13824,
        "final_dim": 32000,
        "num_layer": 40
    },
    "llama-70b":{
        "type": "llama",
        "dim": 8192,
        "fc_dim": 21845,
        "final_dim": 32000,
        "num_layer": 80
    },
    "llama2-70b":{
        "type": "llama",
        "dim": 8192,
        "fc_dim": 28672,
        "final_dim": 32000,
        "num_layer": 80
    },
    "llama3-70b":{
        "type": "llama3",
        "dim": 8192,
        "fc_dim": 28672,
        "final_dim": 128256,
        "num_layer": 80
    },
}

def get_op(input, output, bits, OPtype='FC', name="FP_Mult", batch=1):
    op = {
        "name": name,
        "type": OPtype,
        "wbits": bits,
        "batch": batch,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "input_H": 1,
        "input_W": 1,
        "input_C": input,
        "output_E": 1,
        "output_F": 1,
        "output_M": output
    }
    return op

def get_quant_op(input, output, bits, quant_type="fp"):
    if quant_type == "fp":
        return [get_op(input, output, bits, OPtype='FC', name="FP_Mult"), get_op(input, output, bits, OPtype='Add', name="FP_Add"),]
    elif quant_type == "gptq":
        return [get_op(input, output, bits, OPtype='FC', name="FP_Mult"), get_op(input, output, 16, OPtype='Add', name="FP_Add"),]
    elif quant_type == "lut":
        build_lut_op = [get_op(8, 1, 16, OPtype='FC', name="LUT-Build", batch=int(input//8)), # 8->1 for base result, another 8->1 for offset (2*inp)
                        get_op(8, 1, 16, OPtype='Add', name="LUT-Build", batch=int(input//8)), # 8->1 for base result,
                        get_op(1, 255, 16, OPtype='Add', name="LUT-Build", batch=int(input//8)), # 8->1 for offset
                        ] 
        look_up_op = [get_op(output, bits, 16, OPtype='LUT', name="LookUp", batch=int(input//8)),
                    get_op(output, bits, 16, OPtype='FC', name="FP_Mult", batch=int(input//8)),
                    get_op(int(input//8), output, 16, OPtype='Add', name="FP_Add", batch=1),
                    ]
        return build_lut_op + look_up_op
    elif quant_type == "shiftaddllm":
        build_lut_op = [get_op(8, 1, 16, OPtype='FC', name="LUT-Build", batch=int(input//8)), # 8->1 for base result, another 8->1 for offset (2*inp)
                        get_op(8, 1, 16, OPtype='Add', name="LUT-Build", batch=int(input//8)), # 8->1 for base result,
                        get_op(1, 255, 16, OPtype='Add', name="LUT-Build", batch=int(input//8)), # 8->1 for offset
                        ] 
        look_up_op = [get_op(output, bits, 16, OPtype='LUT', name="LookUp", batch=int(input//8)),
                    get_op(output, bits, 16, OPtype='Shift', name="FP_Shift", batch=int(input//8)),
                    get_op(int(input//8), output, 16, OPtype='Add', name="FP_Add", batch=1),
                    ]   
        return build_lut_op + look_up_op

def get_op_list(model_type, quant_type, dim, fc_dim, wbits):
    OPs_list = []
    if model_type == "opt":
        if wbits == "mix":
            for i in range(2):
                linear_q_op = get_quant_op(dim, dim, bits=2, quant_type=quant_type)
                OPs_list.extend(linear_q_op)
            for i in range(2):
                linear_q_op = get_quant_op(dim, dim, bits=3, quant_type=quant_type)
                OPs_list.extend(linear_q_op)
            linear_q_op = get_quant_op(dim, fc_dim, bits=2, quant_type=quant_type)
            OPs_list.extend(linear_q_op)
            linear_q_op = get_quant_op(fc_dim, dim, bits=2, quant_type=quant_type)
            OPs_list.extend(linear_q_op)
        else:
            for i in range(4):
                linear_q_op = get_quant_op(dim, dim, bits=wbits, quant_type=quant_type)
                OPs_list.extend(linear_q_op)
            linear_q_op = get_quant_op(dim, fc_dim, bits=wbits, quant_type=quant_type)
            OPs_list.extend(linear_q_op)
            linear_q_op = get_quant_op(fc_dim, dim, bits=wbits, quant_type=quant_type)
            OPs_list.extend(linear_q_op)
        # bias
        bias_op = [get_op(dim, 1, bits=16, OPtype='Add', name="FP_Add") for i in range(5)] + [get_op(4 * dim, 1, bits=16, OPtype='Add', name="FP_Add")]
        OPs_list.extend(bias_op)

    elif model_type == "llama":
        if wbits == "mix":
            for i in range(2):
                linear_q_op = get_quant_op(dim, dim, bits=2, quant_type=quant_type)
                OPs_list.extend(linear_q_op)
            for i in range(2):
                linear_q_op = get_quant_op(dim, dim, bits=3, quant_type=quant_type)
                OPs_list.extend(linear_q_op)
            linear_q_op = get_quant_op(dim, fc_dim, bits=2, quant_type=quant_type)
            OPs_list.extend(linear_q_op)
            linear_q_op = get_quant_op(dim, fc_dim, bits=2, quant_type=quant_type)
            OPs_list.extend(linear_q_op)
            OPs_list.append(get_op(fc_dim, 1, bits=16, OPtype='Mult', name="FP_Mult"))
            linear_q_op = get_quant_op(fc_dim, dim, bits=2, quant_type=quant_type)
            OPs_list.extend(linear_q_op)
        else:
            for i in range(4):
                linear_q_op = get_quant_op(dim, dim, bits=wbits, quant_type=quant_type)
                OPs_list.extend(linear_q_op)
            linear_q_op = get_quant_op(dim, fc_dim, bits=wbits, quant_type=quant_type)
            OPs_list.extend(linear_q_op)
            linear_q_op = get_quant_op(dim, fc_dim, bits=wbits, quant_type=quant_type)
            OPs_list.extend(linear_q_op)
            OPs_list.append(get_op(fc_dim, 1, bits=16, OPtype='Mult', name="FP_Mult"))
            linear_q_op = get_quant_op(fc_dim, dim, bits=wbits, quant_type=quant_type)
            OPs_list.extend(linear_q_op)

    elif model_type == "llama3":
        if wbits == "mix":
            linear_q_op = get_quant_op(dim, dim // 8, bits=2, quant_type=quant_type)
            OPs_list.extend(linear_q_op)
            linear_q_op = get_quant_op(dim, dim // 8, bits=3, quant_type=quant_type)
            OPs_list.extend(linear_q_op)
            linear_q_op = get_quant_op(dim, dim, bits=2, quant_type=quant_type)
            OPs_list.extend(linear_q_op)
            linear_q_op = get_quant_op(dim, dim, bits=3, quant_type=quant_type)
            OPs_list.extend(linear_q_op)

            linear_q_op = get_quant_op(dim, fc_dim, bits=2, quant_type=quant_type)
            OPs_list.extend(linear_q_op)
            linear_q_op = get_quant_op(dim, fc_dim, bits=2, quant_type=quant_type)
            OPs_list.extend(linear_q_op)
            OPs_list.append(get_op(fc_dim, 1, bits=16, OPtype='Mult', name="FP_Mult"))
            linear_q_op = get_quant_op(fc_dim, dim, bits=2, quant_type=quant_type)
            OPs_list.extend(linear_q_op)

        else:
            for i in range(2):
                linear_q_op = get_quant_op(dim, dim, bits=wbits, quant_type=quant_type)
                OPs_list.extend(linear_q_op)
            for i in range(2):
                linear_q_op = get_quant_op(dim, dim // 8, bits=wbits, quant_type=quant_type)
                OPs_list.extend(linear_q_op)

            linear_q_op = get_quant_op(dim, fc_dim, bits=wbits, quant_type=quant_type)
            OPs_list.extend(linear_q_op)
            linear_q_op = get_quant_op(dim, fc_dim, bits=wbits, quant_type=quant_type)
            OPs_list.extend(linear_q_op)
            OPs_list.append(get_op(fc_dim, 1, bits=16, OPtype='Mult', name="FP_Mult"))
            linear_q_op = get_quant_op(fc_dim, dim, bits=wbits, quant_type=quant_type)
            OPs_list.extend(linear_q_op)

    return OPs_list



def main(model, quant_type, wbits):

    model_type = MODEL_CONFIG[model]["type"]
    dim = MODEL_CONFIG[model]["dim"]
    fc_dim = MODEL_CONFIG[model].get("fc_dim", dim*4)
    final_dim = MODEL_CONFIG[model]["final_dim"]
    num_layer = MODEL_CONFIG[model]["num_layer"]

    OPs_list = get_op_list(model_type, quant_type, dim, fc_dim, wbits)
    breakdown_energy = {
        "FP_Mult": 0,
        "FP_Add": 0,
        "FP_Shift": 0,
        "LUT-Build": 0,
        "LookUp": 0,
    }

    total_energy_per_layer = 0
    total_latency_per_layer = 0
    for item in tqdm(OPs_list):
        energy, latency, breakdown, min_energy, min_latency = get_OPs_HW_metric(item, v_stats=False, v_show_optimal=False, v_align=True)
        breakdown_energy[item["name"]] += energy

        total_energy_per_layer += energy
        total_latency_per_layer +=  latency

    total_energy = total_energy_per_layer * num_layer
    total_latency = total_latency_per_layer * num_layer
    for key in breakdown_energy.keys():
        breakdown_energy[key] *= num_layer

    final_linear_op = get_quant_op(dim, final_dim, bits=16, quant_type="fp")
    for item in tqdm(final_linear_op):
        energy, latency, breakdown, min_energy, min_latency = get_OPs_HW_metric(item, v_stats=False, v_show_optimal=False, v_align=True)
        breakdown_energy[item["name"]] += energy

        total_energy += energy
        total_latency +=  latency

    print("total energy: {} mJ, total latency: {} ms".format(total_energy, total_latency))
    print("energy breakdown: ", breakdown_energy)
    with open(f"energy_llama.txt", "a") as f:
        f.write(" ======== model: {}, quant type: {}, wbits: {} ========\n".format(model, quant_type, wbits))
        f.write("total energy: {} mJ, total latency: {} ms\n".format(total_energy, total_latency))
        f.write("energy breakdown: {}\n".format(breakdown_energy))
        f.write("\n")

if __name__ == "__main__":

    for model in MODEL_CONFIG.keys():
        main(model, "fp", 16)
        main(model, "shiftaddllm", "mix")

        for method in ["fp", "gptq", "lut"]:
            for wbits in [2, 3]:
                main(model, method, wbits)