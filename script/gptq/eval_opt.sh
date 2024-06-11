CUDA_VISIBLE_DEVICES=0 python model/opt.py \
    facebook/opt-125m \
    --gptq \
    --wbits 3 \
    --groupsize -1 \