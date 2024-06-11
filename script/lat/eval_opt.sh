CUDA_VISIBLE_DEVICES=0 python model/opt.py \
    facebook/opt-125m \
    --wbits 3 \
    --groupsize -1 \
		--lat \
    --bcq_round 50 # bcq_round 20 works too, bigger - slower - maybe better

    # --temp_storage ./weight \