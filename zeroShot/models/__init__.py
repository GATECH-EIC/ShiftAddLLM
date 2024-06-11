from . import opt
from . import bloom
from . import llama

MODEL_REGISTRY = {
    'opt': opt.OPT,
    'bloom': bloom.BLOOM,
    'Llama': llama.Llama
}


def get_model(model_name):
    if 'opt' in model_name:
        return MODEL_REGISTRY['opt']
    elif 'bloom' in model_name:
        return MODEL_REGISTRY['bloom']
    elif 'Llama' in model_name:
        return MODEL_REGISTRY['Llama']
    return MODEL_REGISTRY[model_name]
