import argparse

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument(
			'model', type=str,
			help='LlaMa model to load; pass location of hugginface converted checkpoint.'
	)
	parser.add_argument(
			'--dataset', type=str, default="wikitext2", choices=['wikitext2', 'ptb', 'c4'],
			help='Where to extract calibration data from.'
	)
	parser.add_argument(
			'--load', type=str,
			help='local checkpoint'
	)
	parser.add_argument(
			'--seed',
			type=int, default=0, help='Seed for sampling the calibration data.'
	)
	parser.add_argument(
			'--nsamples', type=int, default=128,
			help='Number of calibration data samples.'
	)
	parser.add_argument(
			'--percdamp', type=float, default=.01,
			help='Percent of the average Hessian diagonal to use for dampening.'
	)
	parser.add_argument(
			'--nearest', action='store_true',
			help='Whether to run the RTN baseline.'
	) 
	parser.add_argument(
			'--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
			help='#bits to use for quantization; use 16 for evaluating base model.'
	)
	parser.add_argument(
			'--groupsize', type=int, default=-1,
			help='Groupsize to use for quantization; default uses full row.'
	)
	parser.add_argument(
			'--blocksize', type=int, default=128,
			help='Blocksize to use for quantization.'
	)

	parser.add_argument(
			'--benchmark', type=int, default=0,
			help='Number of tokens to use for benchmarking.'
	)
	parser.add_argument(
			'--check', action='store_true',
			help='Whether to compute perplexity during benchmarking for verification.'
	)
	parser.add_argument(
			'--new-eval', action='store_true',
			help='Whether to use the new PTB and C4 eval.'
	)
	parser.add_argument(
			'--act-order', action='store_true',
			help='Whether to apply the activation order GPTQ heuristic'
	)
	parser.add_argument(
			'--true-sequential', action='store_true',
			help='Whether to run in true sequential model.'
	)

	# GPTQ method
	parser.add_argument(
			'--gptq', action='store_true',
			help='Whether to use gptq method.'
	)
	parser.add_argument(
			'--sym', action='store_true',
			help='Whether to perform symmetric quantization.'
	)
	parser.add_argument(
			'--save', type=str, default='',
			help='Save quantized checkpoint under this name.'
	)
	parser.add_argument(
			'--static-groups', action='store_true',
			help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
	)
	parser.add_argument(
        '--trits', action='store_true',
        help='Whether to use trits for quantization.'
    )

	# bcq quant - LUT-gemm
	parser.add_argument(
			'--bcq', action='store_true', help='Quantize weight with bcq.'
	)
	parser.add_argument(
			'--lut_eval', action='store_true', help='Use lut+gptq to evaluate model.'
	)
	parser.add_argument(
			'--bcq_round', type=int, default=50,
			help='Steps to iterate bcq quantization.'
	)

	# shiftaddllm quant

	parser.add_argument(
			'--acc', action='store_true', default=False,
			help='whether to use ours(acc.) quantization method'
	)
	parser.add_argument(
			'--lat', action='store_true', default=False,
			help='whether to use ours(lat.) quantization method'
	)

	parser.add_argument(
			'--columnwise', action='store_true',
			help='Use columnwise - bcq - round to power of 2 - quantization to evaluate model. Can be used with new cuda kernel.'
	)
	parser.add_argument(
			'--block_quant', action='store_true',
			help='!!Only work when columnwise, Use blockwise (8 column for 1 quantize param) - bcq - round to power of 2 - quantization to evaluate model.'
	)
	parser.add_argument(
			'--use_bst', action='store_true',default=False,
			help='Use bst of get BinaryWeight'
	)
	parser.add_argument(
			'--apot_nums', type=int, default=3,
			help='set nums shift weight for quantization.'
	)

	parser.add_argument(
			'--temp_storage', type=str, default=None,
			help='Save temporary quantized weight with low bit under this dir. only work for lat. method.'
	)
	parser.add_argument(
			'--load_temp_storage', type=str, default=None,
			help='load from temporary quantized weight with low bit under this dir.'
	)
	parser.add_argument(
			'--infer_kernel', action='store_true',default=False,
			help='whether to inference with cuda kernel.'
	)


	# mix precision
	parser.add_argument(
		'--quant_config', type=str, default=None,
		help='path for mix bit quantization config.'
	)
	parser.add_argument(
		'--record_error', type=str, default=None,
		help='whether record the error of the quantization. Used for rank mixbit allocation result'
	)

	# QUIP method
	parser.add_argument(
			'--pre_gptqH',
			action='store_true',
			default=True,
			help='preprocessing')
	parser.add_argument(
			'--pre_rescale',
			action='store_true',
			help='preprocessing')
	parser.add_argument(
			'--pre_proj',
			action='store_true',
			help='preprocessing')
	parser.add_argument(
			'--pre_proj_extra',
			type=int,
			default=0,
			choices=[0, 1, 2],
			help='Extra options to control pre_proj step.')
	parser.add_argument(
			'--incoh_processing',
			action='store_true',
			help='incoherence processing')

	args = parser.parse_args()
		
	if args.acc:
		args.incoh_processing = True
		args.act_order = True
		args.use_bst = True
		args.columnwise = True
		args.block_quant = False
	elif args.lat:
		args.incoh_processing = False
		args.act_order = True
		args.use_bst = True
		args.block_quant = True
		args.columnwise = False

	if args.incoh_processing:
		args.pre_gptqH   = True
		args.pre_rescale = True
		args.pre_proj    = True
		args.pre_proj_extra = 0

	return args