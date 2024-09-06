from tqdm import tqdm
from quantizers.bcq_quant.bcq_parameter import BCQParameter

# Define the layers that will be quantized
# These are the attention and MLP layers 
layers = ["q_proj","k_proj","v_proj","out_proj","fc1","fc2","o_proj","gate_proj","up_proj","down_proj"]

def quant_model(model, qbits:int = 4, group_size:int = 128, rounds=50):
    """
    Quantize a pre-trained model using Binary-coding Quantization (BCQ).
    
    Args:
    - model: The pre-trained model to be quantized
    - qbits: Number of bits for quantization (default: 4)
    - group_size: Size of groups for group-wise quantization (default: 128)
    - rounds: Number of alternating optimization rounds (default: 50)
    
    Returns:
    - Quantized model
    """
    # Get the current state of the model
    parameters = model.state_dict()
    
    # Iterate through all named children (layers) of the model
    for name, module in tqdm(list(model.named_children()), desc="Quantizing model"):
        # If the module has children, recursively quantize them
        if len(list(module.children())) > 0:
            quant_model(module, qbits, group_size)

        # Check if the current layer is one we want to quantize
        if any(x in name for x in layers):
            # Clone and detach the original weight to avoid modifying the original
            original_weight = module.weight.clone().detach()
            
            # Create a BCQParameter object for quantization
            w_bcq = BCQParameter(original_weight)
            
            # Compress the weights using BCQ
            # This step quantizes the weights into binary matrices and scaling factors

            # alpha: Scaling factors for each binary matrix, shape (qbits,)
            # binary: Quantized binary matrices, where each element is either -1 or +1
            # binary_shape: Original shape of the binary matrices before flattening
            alpha, binary, binary_shape = w_bcq.compress(
                do_packing=False,       # Don't pack the binary values. This saves space as 
                                        # multiple binary values can  be packed into the integer value
                in_ch_wise=True,        # Perform channel-wise quantization
                qbits=qbits,            # Use the specified number of bits
                rounds=rounds,          # Number of alternating optimization rounds
                group_size=group_size   # Size of groups for quantization
            )
          
            # At this point, the weight matrix W is approximated as:
            # W ≈ Σ(αᵢ * Bᵢ) for i = 1 to qbits
            # where αᵢ are the scaling factors (alpha)
            # and Bᵢ are the binary matrices (binary) with elements in {-1, +1}
            
            # Decompress the weights
            # This step reconstructs the weights from binary matrices and scaling factors
            # It's done to update the BCQParameter object with the quantized weights
            w_bcq.decompress(alpha, binary, binary_shape, do_packing=False, in_ch_wise=True)
            
            # Update the model's state dict with the quantized weights
            parameters[name + ".weight"] = w_bcq.data.clone().detach()
    
    # Load the updated state (with quantized weights) back into the model
    model.load_state_dict(parameters)

    return model
