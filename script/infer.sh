CUDA_VISIBLE_DEVICES=3 python model/opt.py \
    facebook/opt-125m \
    --wbits 3 \
    --lat \
    --load_temp_storage ./weight \
    --infer_kernel \
    --benchmark 128