CUDA_VISIBLE_DEVICES=0 python model/llama.py \
    meta-llama/Llama-2-13b-hf \
    --wbits 3 \
    --groupsize 128 \
		--lat \
    --bcq_round 50 # bcq_round 20 works too, bigger - slower - maybe better

    # --temp_storage ./weight \