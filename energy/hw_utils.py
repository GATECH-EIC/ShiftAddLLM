import os
import math
import copy
import csv
import numpy as np


def valid_tiling(tiling_factor, FC_flag, num_bits=32):

    if num_bits == 32:
        mem_factor = 1
        cmp_factor = 1
    elif num_bits == 16:
        mem_factor = 1
        cmp_factor = 1
    else:
        mem_factor = 2
        cmp_factor = 4

    valid = True
    N3 = tiling_factor["N3"]
    N0 = tiling_factor["N0"]
    N = tiling_factor["N"]
    if not (N == N3*N0):
        valid = False
    valid = True
    M3 = tiling_factor["M3"]
    M2 = tiling_factor["M2"]
    M1 = tiling_factor["M1"]
    M0 = tiling_factor["M0"]
    M = tiling_factor["M"]
    if not (M == M3*M2*M1*M0):
        valid = False
    if not (M0 <= 16 * mem_factor):          # output RF
        valid = False
    C3 = tiling_factor["C3"]
    C2 = tiling_factor["C2"]
    C1 = tiling_factor["C1"]
    C0 = tiling_factor["C0"]
    C = tiling_factor["C"]
    if not (C == C3*C2*C1*C0):
        valid = False
    if not (C3 == 1):
        valid = False
    E1 = tiling_factor["E1"]
    E3 = tiling_factor["E3"]
    E = tiling_factor["E"]
    if not (E == E1*E3):
        valid = False
    if not FC_flag:
        if not (E1 != 1):
            valid = False
    R = tiling_factor["R"]
    S = tiling_factor["S"]
    if not (M1*C1*E1*R < 16*12 * cmp_factor): # number of PEs
        valid = False
    if not (C0*S < 12*mem_factor):          # input RF
        valid = False
    # print('check 2: ', valid)
    if not (M0*C0*S < 192*5*mem_factor):      # weight RF
        valid = False
    stride = tiling_factor["stride"]
    F = tiling_factor["F"]
    if not (N0*C1*C0*((E1-1)*stride+R)*((F-1)*stride+S) + N0*M2*M1*M0*E1*F < 65536*mem_factor): # SRAM
        valid = False
    return valid

def get_latency(tiling_factor, unit_latency):
    N3 = tiling_factor["N3"]
    N0 = tiling_factor["N0"]
    N = tiling_factor["N"]
    M3 = tiling_factor["M3"]
    M2 = tiling_factor["M2"]
    M1 = tiling_factor["M1"]
    M0 = tiling_factor["M0"]
    M = tiling_factor["M"]
    C3 = tiling_factor["C3"]
    C2 = tiling_factor["C2"]
    C1 = tiling_factor["C1"]
    C0 = tiling_factor["C0"]
    C = tiling_factor["C"]
    E1 = tiling_factor["E1"]
    E3 = tiling_factor["E3"]
    E = tiling_factor["E"]
    R = tiling_factor["R"]
    S = tiling_factor["S"]
    stride = tiling_factor["stride"]
    F = tiling_factor["F"]

    latency = N3*N0*M3*M2*M0*C3*C2*C0*E3*F*S*unit_latency
    return latency

def get_energy(tiling_factor, unit_energy):
    N3 = tiling_factor["N3"]
    N0 = tiling_factor["N0"]
    N = tiling_factor["N"]
    M3 = tiling_factor["M3"]
    M2 = tiling_factor["M2"]
    M1 = tiling_factor["M1"]
    M0 = tiling_factor["M0"]
    M = tiling_factor["M"]
    C3 = tiling_factor["C3"]
    C2 = tiling_factor["C2"]
    C1 = tiling_factor["C1"]
    C0 = tiling_factor["C0"]
    C = tiling_factor["C"]
    E1 = tiling_factor["E1"]
    E3 = tiling_factor["E3"]
    E = tiling_factor["E"]
    R = tiling_factor["R"]
    S = tiling_factor["S"]
    stride = tiling_factor["stride"]
    F = tiling_factor["F"]

    H = tiling_factor["H"]
    W = tiling_factor["W"]
    num_ifmap = N*H*W*C # input feature map size
    num_weight = R*S*C*M # weight size
    num_ofmap = N*E*F*M # output size

    computation = N*E*F*M*R*S*C
    DRAM_ifmap = M3 * num_ifmap
    DRAM_weight = N3 * E3 * num_weight
    DRAM_ofmap = ( max((2*C3-1), 1) ) * num_ofmap
    GB_ifmap = M3 * M2 * num_ifmap
    GB_ofmap = ( max(2 * C3 * (C2-1), 1) ) * num_ofmap
    NoC_ifmap = M3 * M2 * M1 * R * E / H * num_ifmap
    NoC_weight = N3 * E3 * E1 * num_weight
    NoC_ofmap = ( max(C3 * C2 * (C1 * R - 1), 1) ) * num_ofmap
    RF_ifmap = M3 * M2 * M1 * R * E / H * M0 * S * F / W * num_ifmap
    RF_weight = N3 * E3 * E1 * F * N0 * num_weight
    RF_ofmap = ( max(C3 * C2 * C1 * R * (C0 * S - 1 ) * 2, 1) ) * num_ofmap
    energy = computation * unit_energy["unit_comp"] \
             + (DRAM_ifmap + DRAM_weight + DRAM_ofmap) * unit_energy["unit_DRAM"] \
             + (DRAM_ifmap + DRAM_ofmap) * unit_energy["unit_DRAM_GB"] \
             + (GB_ifmap + GB_ofmap) * unit_energy["unit_GB"] \
             + (NoC_ifmap + NoC_weight) * unit_energy["unit_NoC"] \
             + (NoC_ofmap) * unit_energy["unit_NoC_psum"] \
             + (RF_ifmap + RF_ofmap) * unit_energy["unit_RF"] \
             + (RF_weight) * unit_energy["unit_RF_weight"]
    return energy, [computation * unit_energy["unit_comp"], \
        (DRAM_ifmap + DRAM_weight + DRAM_ofmap) * unit_energy["unit_DRAM"] \
        + (DRAM_ifmap + DRAM_ofmap) * unit_energy["unit_DRAM_GB"] \
        + (GB_ifmap + GB_ofmap) * unit_energy["unit_GB"] \
        + (NoC_ifmap + NoC_weight) * unit_energy["unit_NoC"] + (NoC_ofmap) * unit_energy["unit_NoC_psum"] \
        +(RF_ifmap + RF_ofmap) * unit_energy["unit_RF"] + (RF_weight) * unit_energy["unit_RF_weight"] \
        + DRAM_ifmap * (unit_energy["unit_DRAM"] + unit_energy["unit_DRAM_GB"]) + GB_ifmap * unit_energy["unit_GB"] + NoC_ifmap * unit_energy["unit_NoC"] + RF_ifmap * unit_energy["unit_RF"] \
        + DRAM_weight * unit_energy["unit_DRAM"] + NoC_weight * unit_energy["unit_NoC"] + RF_weight * unit_energy["unit_RF_weight"] \
        + DRAM_ofmap * (unit_energy["unit_DRAM"] + unit_energy["unit_DRAM_GB"]) + GB_ofmap * unit_energy["unit_GB"] + NoC_ofmap * unit_energy["unit_NoC_psum"] + RF_ofmap * unit_energy["unit_RF"]]


# Refer: https://www.geeksforgeeks.org/print-all-prime-factors-of-a-given-number/
def primeFactors(n):
    prime_list = []
    # Print the number of two's that divide n
    while n % 2 == 0:
        prime_list.append(2)
        n = n / 2

    # n must be odd at this point
    # so a skip of 2 ( i = i + 2) can be used
    for i in range(3,int(math.sqrt(n))+1,2):

        # while i divides n , print i ad divide n
        while n % i== 0:
            prime_list.append(int(i))
            n = n / i

    # Condition if n is a prime
    # number greater than 2
    if n > 2:
        prime_list.append(int(n))
    return prime_list

def possible_mul(x,l):
    if len(l) == 1:
        raw_list = [x*l[0], x*1]
        clean_list = list(dict.fromkeys(raw_list))
        return clean_list
    else:
        raw_list = possible_mul(x*l[0], l[1:]) + possible_mul(x*1, l[1:])
        clean_list = list(dict.fromkeys(raw_list))
        return clean_list

def tile(num, tile_size):
    if tile_size == 1:
        return [[num]]
    else:
        if num == 1:
            prime_list = [1]
        else:
            prime_list = primeFactors(num)

        tile_list = []
        selected_list = possible_mul(1, prime_list)
        for selected in selected_list:
            # select 1 for current the first position
            for options in tile(int(num/selected), tile_size-1):
                to_append = [selected,] + options
                if to_append not in tile_list:
                    tile_list.append(to_append)
        return tile_list

def get_unit_energy(layer_dict, num_bits=32):

    assert num_bits in [32, 16, 4, 3, 2]

    if num_bits == 32: # fp32
        unit_mult = 3.7
        unit_add = 0.9
        unit_shift = 0.1
    else:
        unit_mult = 1.1
        unit_add = 0.4
        unit_additive_shift = 0.15
    unit_lut = 0.37

    unit_energy = {}
    if (layer_dict["type"] == "AvgP") or (layer_dict["type"] == "MaxP"):
        unit_energy["unit_comp"] = 0.0/(1e9) # mJ/MAC
    elif (layer_dict["type"] == "Add"):
        unit_energy["unit_comp"] = unit_add/(1e9) # mJ/MAC
    elif (layer_dict["type"] == "Shift"):
        unit_energy["unit_comp"] = unit_additive_shift/(1e9) # mJ/MAC
    elif (layer_dict["type"] == "LUT"):
        unit_energy["unit_comp"] = unit_lut/(1e9) # mJ/MAC
    else:
        unit_energy["unit_comp"] = unit_mult/(1e9) # mJ/MAC

    if layer_dict["type"] == "Shift":
        num_bits = (16 + 5) // 2
        unit_energy["unit_DRAM"] = 200/(1e9) * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_DRAM_GB"] = 0.0/(1e9) * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_GB"] = 6.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_NoC"] = 2.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_NoC_psum"] = 1.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_RF"] = 1.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_RF_weight"] = 2.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits

    elif layer_dict["type"] == "Add" or layer_dict["type"] == "Mult" or layer_dict["type"] == "LUT":
        # on chip
        unit_energy["unit_DRAM"] = 0 # mJ/16 bits
        unit_energy["unit_DRAM_GB"] = 0 # mJ/16 bits
        unit_energy["unit_GB"] = 0 # mJ/16 bits
        unit_energy["unit_NoC"] = 0 # mJ/16 bits
        unit_energy["unit_NoC_psum"] = 1.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_RF"] = 1.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_RF_weight"] = 2.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits

    else:
        unit_energy["unit_DRAM"] = 200/(1e9) * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_DRAM_GB"] = 0.0/(1e9) * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_GB"] = 6.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_NoC"] = 2.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_NoC_psum"] = 1.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_RF"] = 1.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits
        unit_energy["unit_RF_weight"] = 2.0/(1e9) * 3.7 * (num_bits / 16) # mJ/16 bits

    # need to confirm
    # unit_latency = 1.0 / 1e9 # 1GHz
    if layer_dict["type"] == "Shift":
        # do not need to store back to off-chip memory (plus 16 bits benefits)
        unit_latency = 1.0 / (250e6) / (32 / num_bits) # 250MHz
    else:
        unit_latency = 1.0 / (250e6) # 250MHz

    return unit_energy, unit_latency


# gives the energy (mJ), latency (ms)
def get_OPs_HW_metric(layer_dict, v_stats=False,v_show_optimal=False,v_align=False):
    # constant defination
    num_bits = layer_dict["wbits"]
    if layer_dict["type"] == "FC":
        FC_flag = True
    elif (layer_dict['input_H'] == 1) and (layer_dict['input_W'] == 1) and (layer_dict['output_E'] == 1) and (layer_dict['output_F'] == 1):
        FC_flag = True
    else:
        FC_flag = False

    unit_energy, unit_latency = get_unit_energy(layer_dict, num_bits)

    # Add basic information to tiling_factor
    base_tiling_factor = {}
    base_tiling_factor["N"] = layer_dict["batch"]
    base_tiling_factor["H"] = layer_dict["input_H"]
    base_tiling_factor["W"] = layer_dict["input_W"]
    base_tiling_factor["C"] = layer_dict["input_C"]
    base_tiling_factor["R"] = layer_dict["kernel_size"]
    base_tiling_factor["S"] = layer_dict["kernel_size"]
    base_tiling_factor["M"] = layer_dict["output_M"]
    base_tiling_factor["E"] = layer_dict["output_E"]
    base_tiling_factor["F"] = layer_dict["output_F"]
    base_tiling_factor["stride"] = layer_dict["stride"]
    # tile N to N0 * N3
    N_tile_list = tile(base_tiling_factor["N"], 2)
    # tile M to M0 * M1 * M2 * M3
    M_tile_list = tile(base_tiling_factor["M"], 4)
    # filter out M0 > 16 options
    for tile_option in M_tile_list:
        if tile_option[0] > 16:
            M_tile_list.remove(tile_option)
    # tile C to C0 * C1 * C2 * C3
    C_tile_list = tile(base_tiling_factor["C"], 4)
    # filter out C3 != 1 options
    for tile_option in C_tile_list:
        if tile_option[3] != 1:
            C_tile_list.remove(tile_option)
    # tile E to E1 * E3
    E_tile_list = tile(base_tiling_factor["E"], 2)
    # filter out E1 == 1 options
    if not FC_flag:
        for tile_option in E_tile_list:
            if tile_option[0] == 1:
                E_tile_list.remove(tile_option)

    energy_list = []
    breakdown_list = []
    latency_list = []
    tiling_factor_list = []

    for N_tile in N_tile_list:
        for M_tile in M_tile_list:
            for C_tile in C_tile_list:
                for E_tile in E_tile_list:
                    tiling_factor = copy.deepcopy(base_tiling_factor)
                    tiling_factor["N0"] = N_tile[0]
                    tiling_factor["N3"] = N_tile[1]
                    tiling_factor["M0"] = M_tile[0]
                    tiling_factor["M1"] = M_tile[1]
                    tiling_factor["M2"] = M_tile[2]
                    tiling_factor["M3"] = M_tile[3]
                    tiling_factor["C0"] = C_tile[0]
                    tiling_factor["C1"] = C_tile[1]
                    tiling_factor["C2"] = C_tile[2]
                    tiling_factor["C3"] = C_tile[3]
                    tiling_factor["E1"] = E_tile[0]
                    tiling_factor["E3"] = E_tile[1]
                    if valid_tiling(tiling_factor, FC_flag, num_bits=num_bits):
                        energy, breakdown  =  get_energy(tiling_factor, unit_energy)
                        latency = get_latency(tiling_factor, unit_latency)

                        energy_list.append(energy)
                        breakdown_list.append(breakdown)
                        latency_list.append(latency)
                        tiling_factor_list.append(tiling_factor)

    # tiling factor search M, C, E
    max_energy = max(energy_list)
    min_energy = min(energy_list)

    max_latency = max(latency_list)
    min_latency = min(latency_list)

    min_normal_metric = (energy_list[0]-min_energy)/max_energy + (latency_list[0]-min_latency)/max_latency

    total_optimal_tiling_factor_idx = [0]
    latency_optimal_tiling_factor_idx = []
    energy_optimal_tiling_factor_idx = []

    for i in range(len(tiling_factor_list)):
        tiling_factor = tiling_factor_list[i]
        energy = energy_list[i]
        latency = latency_list[i]

        normal_metric = (energy_list[i]-min_energy)/max_energy + (latency_list[i]-min_latency)/max_latency

        # update total optimal
        if normal_metric < min_normal_metric:
            min_normal_metric = normal_metric
            total_optimal_tiling_factor_idx = [i]
        elif normal_metric == min_normal_metric:
            total_optimal_tiling_factor_idx.append(i)
        if latency == min_latency:
            latency_optimal_tiling_factor_idx.append(i)
        if energy == min_energy:
            energy_optimal_tiling_factor_idx.append(i)

    if v_stats:
        print("max_energy: {} mJ, min_energy: {} mJ".format(max_energy, min_energy))
        print("max_latency: {} ms, min_latency: {} ms".format(max_latency, min_latency))

    if len(latency_optimal_tiling_factor_idx) > 1:
        if v_show_optimal:
            print("Notice!!!!!, There are multiple latency optimal tiling factor")
    for optimal_idx in latency_optimal_tiling_factor_idx:
        if v_show_optimal:
            print("==Latency OPTIMAL==")
            print("Latency optimal tiling factor: {}, with Energy: {} mJ, Latency: {} ms".format(tiling_factor_list[optimal_idx], energy_list[optimal_idx], latency_list[optimal_idx]))
            print("==Latency OPTIMAL==")

    if len(energy_optimal_tiling_factor_idx) > 1:
        if v_show_optimal:
            print("Notice!!!!!, There are multiple energy optimal tiling factor")
    for optimal_idx in energy_optimal_tiling_factor_idx:
        if v_show_optimal:
            print("==Energy OPTIMAL==")
            print("Energy optimal tiling factor: {}, with Energy: {} mJ, Latency: {} ms".format(tiling_factor_list[optimal_idx], energy_list[optimal_idx], latency_list[optimal_idx]))
            print("==Energy OPTIMAL==")

    if len(total_optimal_tiling_factor_idx) > 1:
        if v_align:
            print("Notice!!!!!, There are multiple total optimal tiling factor")
    for optimal_idx in total_optimal_tiling_factor_idx:
        if ((optimal_idx in latency_optimal_tiling_factor_idx) and (optimal_idx in energy_optimal_tiling_factor_idx)):
            if v_align:
                print("==TOTAL OPTIMAL==")
                print("Good News, this tiling factor is latency optimal + energy optimal:")
                print("Total optimal tiling factor: {}, with Energy: {} (min: {}) mJ, Latency: {} (min: {}) ms".format(tiling_factor_list[optimal_idx], energy_list[optimal_idx], min_energy, latency_list[optimal_idx], min_latency))
                print("==TOTAL OPTIMAL==")
        else:
            if v_align:
                print("==TOTAL OPTIMAL==")
                print("Bad News, this tiling factor is NOT latency optimal + energy optimal:")
                print("Total optimal tiling factor: {}, with Energy: {} (min: {}) mJ, Latency: {} (min: {}) ms".format(tiling_factor_list[optimal_idx], energy_list[optimal_idx], min_energy, latency_list[optimal_idx], min_latency))
                print("==TOTAL OPTIMAL==")
    return energy_list[total_optimal_tiling_factor_idx[0]], latency_list[total_optimal_tiling_factor_idx[0]], breakdown_list[total_optimal_tiling_factor_idx[0]], min_energy, min_latency