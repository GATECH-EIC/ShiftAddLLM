CUDA_VISIBLE_DEVICES=3 python main.py \
    facebook/opt-2.7b \
    --load ../weight/opt2.7b-3bit-optq.pth

    # --load_shiftaddllm "../weight/tempweight/opt-125m-block-3bit/" \