
# =========== quantize with original bcq method ===========
CUDA_VISIBLE_DEVICES=0 python model/opt.py \
    facebook/opt-125m \
    --wbits 3 \
    --groupsize -1 \
    --bcq \
    --bcq_round 50 \
    --use_bst \


# =========== quantize with bcq+gptq method ===========
CUDA_VISIBLE_DEVICES=0 python model/opt.py \
    facebook/opt-125m \
    --wbits 3 \
    --groupsize -1 \
    --lut_eval \
    --bcq_round 50 \
    --use_bst