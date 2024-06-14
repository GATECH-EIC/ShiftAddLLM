This folder contains code to reproduce the FewShot tasks. We follow the structure of 
[this](https://github.com/EleutherAI/lm-evaluation-harness) repository for implementing 
our tasks and the evaluation framework.

We implement the following tasks:
- [x] ARC-easy
- [x] ARC-challenge
- [x] COPA
- [x] BOOL-Q
- [x] StoryCloze-2018
- [x] RTE


To add new tasks, please follow [this](https://github.com/EleutherAI/lm-evaluation-harness#code-structure) 
instruction.

## Dependencies

* `torch`
* `transformers`
* `datasets`
* `sacrebleu`
* `scikit-learn`

# Usage

To use the code, you need to simply run the following command:

```bash 
python3 main.py  <model_name> --load <ckpt_path>

python3 main.py  <model_name> --load_shiftaddllm <packWeight_dir>
```
