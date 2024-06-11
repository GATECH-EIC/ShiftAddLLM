CUDA_VISIBLE_DEVICES=0 python main.py \
    meta-llama/Meta-Llama-3-70B \
    wikitext2 \
    --load Llama-3-70b-optq-wbits3-groupsize128.pt