import torch
import torch.nn as nn
import math

class FlexRoundParametrization(nn.Module):
    """
    A learnable parametrization for FlexRound.
    
    This class defines:
      - s1: The global grid scale (a single learnable scalar).
      - scale: A per-weight learnable tensor, same shape as the weight.
    
    The forward pass does:  W_rounded = s1 * round( W / (s1 * scale) )
    
    You can expand this to include s3, s4, etc., if you want channel-wise or group-wise factors.
    """
    def __init__(self, weight_shape, init_scale=1.0):
        super().__init__()
        
        # One global scale
        self.s1 = nn.Parameter(torch.tensor(init_scale))
        
        # Element-wise scale
        # Same shape as weight, all ones initially
        self.scale = nn.Parameter(torch.ones(weight_shape) * init_scale)

    def forward(self, W_fullprecision):
        # We do a clamp to avoid division by zero
        denom = torch.clamp(self.s1 * self.scale, min=1e-8)
        
        # 1) Divide
        w_div = W_fullprecision / denom
        
        # 2) Round
        w_rounded = torch.round(w_div)
        
        # 3) Scale back up
        W_quant = self.s1 * w_rounded
        return W_quant

class FlexRoundQuantizer:
    """
    Similar to HWPQ, but uses FlexRound to quantize. 
    Also does optional pruning, returning (quantized_weights, mask).
    """
    def __init__(self, structured_sparsity=False):
        """
        Args:
            structured_sparsity: If True, apply 2:4 structured pruning. 
                                 (Or adapt as needed)
        """
        self.structured_sparsity = structured_sparsity
        
        # For demonstration, no separate "compute_contribution" method here;
        # you could re-use the HWPQ approach or your own magnitude-based approach.

    def prune_and_flexround_weights(self, weights, sparsity_level=0.5):
        """
        1) Prunes 'weights' to achieve `sparsity_level`.
        2) Applies our FlexRoundParametrization for quantization 
           (using simple defaults).
        
        Returns:
            pruned_and_quantized_weights, mask
        """
        # Make a copy for safety
        pruned_weights = weights.clone()
        mask = torch.ones_like(weights, dtype=torch.bool)
        
        total_weights = weights.numel()
        target_prune = int(total_weights * sparsity_level)
        
        # == Simple unstructured pruning example (lowest magnitude) ==
        #   You could adapt to row-wise or contribution-based.
        
        # Flatten
        w_flat = weights.view(-1)
        abs_w_flat = torch.abs(w_flat)
        
        # Sort by magnitude (lowest first)
        sorted_indices = torch.argsort(abs_w_flat)
        
        # Prune
        if target_prune >= total_weights:
            target_prune = max(0, total_weights - 1)
        prune_indices = sorted_indices[:target_prune]
        
        # Create mask
        mask_flat = torch.ones_like(w_flat, dtype=torch.bool)
        mask_flat[prune_indices] = 0
        # Make sure not everything is pruned
        if (mask_flat == 0).all():
            # keep the largest magnitude
            max_idx = sorted_indices[-1]
            mask_flat[max_idx] = 1
        
        # Reshape mask
        mask = mask_flat.view_as(weights)
        
        # ========== FlexRound QUANTIZATION ==========
        # We can't do "learning" on s1, scale in a single pass, 
        # but we can define them, then do a forward pass:
        
        # Build the parametrization
        flexround_param = FlexRoundParametrization(weights.shape)
        
        # Typically you'd "train" flexround_param.s1 and flexround_param.scale 
        # to minimize an error measure. For demonstration, 
        # let's do a single forward pass with default initialization:
        
        # Zero out pruned elements
        pruned = weights * mask
        
        # Forward pass (no training for now)
        with torch.no_grad():
            quantized = flexround_param(pruned)
        
        pruned_weights.copy_(quantized)
        
        # Return
        return pruned_weights, mask


def flexround_pruning(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    """
    A function analogous to 'hwpq_pruning' for MASE:
      - Creates a FlexRoundQuantizer
      - Returns a boolean mask (True = keep, False = prune)
    
    The MASE framework calls this to get the mask, then uses Parametrizations, etc.
    """
    # You might read structured_sparsity from info
    structured_sparsity = info.get("structured_sparsity", False)
    
    flex = FlexRoundQuantizer(structured_sparsity=structured_sparsity)
    
    # We want ONLY the mask for MASE
    _, mask = flex.prune_and_flexround_weights(tensor, sparsity)
    return mask


class FlexRoundParameterization(nn.Module):
    """
    A param wrapper if you want to separately store the mask and do the FlexRound
    quantization for forward passes in the model (like HWPQParameterization).
    
    This means:
      - 'mask' is kept in a buffer
      - 'flexround_param' is a submodule that has s1, scale, etc.
    
    Then, PyTorch uses it in forward pass to compute quantized W. 
    """
    def __init__(self, weight_shape, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        
        # Our learnable param (could even be the same class or second instance)
        self.flexround_param = FlexRoundParametrization(weight_shape)

    def forward(self, W_fullprecision):
        # Zero out pruned weights
        pruned = self.mask * W_fullprecision
        # Apply the flexible rounding
        quantized = self.flexround_param(pruned)
        return quantized

    def state_dict(self, *args, **kwargs):
        """
        If we want to avoid duplicating the mask in state_dict, 
        we can override or store minimal info.
        """
        return super().state_dict(*args, **kwargs)
