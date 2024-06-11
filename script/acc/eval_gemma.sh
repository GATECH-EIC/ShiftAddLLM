CUDA_VISIBLE_DEVICES=0 python model/gemma.py \
    google/gemma-2b \
    --wbits 3 \
    --groupsize -1 \
		--acc \
    --bcq_round 50 # bcq_round 20 works too, bigger - slower - maybe better