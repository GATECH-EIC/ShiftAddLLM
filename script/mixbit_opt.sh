# =========== analysis the weight and find the important part, set higher bits =========
CUDA_VISIBLE_DEVICES=0 python opt_analysis.py \
    facebook/opt-125m \
    --mix_bits "0.0,0.8,0.2"
    # --load_analyse_result

# =========== quantize the model with mixbit config =========
CUDA_VISIBLE_DEVICES=0 python model/opt.py \
    facebook/opt-125m \
    --wbits 3 \
    --groupsize -1 \
		--lat \
    --bcq_round 50 \
    --quant_config sensitivity/mixbit/opt-125m.json
