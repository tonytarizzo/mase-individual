import torch
import math

class FlexRound:
    """
    FlexRound: A flexible rounding scheme for quantization
    
    This class implements the FlexRound quantization method, which provides better
    quantization by dynamically adjusting the rounding behavior based on the value
    distribution.
    """
    
    def __init__(self, 
                 bit_width=8, 
                 frac_width=4, 
                 alpha=0.5, 
                 beta=0.25, 
                 use_adaptive_scale=True):
        """
        Initialize FlexRound with given parameters
        
        Args:
            bit_width: Total bit width for quantization
            frac_width: Fractional bit width for quantization
            alpha: FlexRound parameter controlling interpolation between round-to-nearest and probabilistic rounding
            beta: FlexRound parameter controlling range preservation
            use_adaptive_scale: Whether to use adaptive scaling based on tensor statistics
        """
        self.bit_width = bit_width
        self.frac_width = frac_width
        self.alpha = alpha
        self.beta = beta
        self.use_adaptive_scale = use_adaptive_scale
        
        # Calculate quantization parameters
        self.scale = 2.0 ** frac_width
        self.max_val = 2.0 ** (bit_width - frac_width - 1) - 2.0 ** (-frac_width)
        self.min_val = -2.0 ** (bit_width - frac_width - 1)
        
    def _compute_adaptive_scale(self, tensor):
        """
        Compute an adaptive scale factor based on tensor statistics
        
        Args:
            tensor: Input tensor
            
        Returns:
            Adaptive scale factor
        """
        abs_max = tensor.abs().max().item()
        if abs_max < 1e-6:  # Avoid division by zero
            return 1.0
            
        # Calculate optimal scale to utilize the available range
        range_max = max(abs(self.max_val), abs(self.min_val))
        return range_max / abs_max
        
    def _flex_rounding(self, x):
        """
        Apply FlexRound rounding scheme to tensor x
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor after applying FlexRound
        """
        # Floor values (truncation)
        x_floor = torch.floor(x)
        
        # Residual (fractional part)
        residual = x - x_floor
        
        # Standard rounding: round to nearest with 0.5 threshold
        nearest_round = torch.where(residual >= 0.5, x_floor + 1.0, x_floor)
        
        # Probabilistic rounding based on residual
        rand = torch.rand_like(x)
        prob_round = torch.where(rand < residual, x_floor + 1.0, x_floor)
        
        # FlexRound: interpolate between nearest and probabilistic rounding
        # based on value magnitude (scaled by alpha and beta)
        x_abs = torch.abs(x)
        relative_pos = torch.clamp(x_abs / self.max_val, 0.0, 1.0)
        
        # Weight factor based on beta and relative position in range
        weight = torch.clamp(self.beta * (1.0 - relative_pos), 0.0, self.alpha)
        
        # Interpolate between nearest rounding and probabilistic rounding
        flex_rounded = torch.lerp(nearest_round, prob_round, weight)
        
        return flex_rounded
    
    def quantize(self, tensor):
        """
        Quantize a tensor using FlexRound
        
        Args:
            tensor: Input tensor to be quantized
            
        Returns:
            Quantized tensor
        """
        # Copy tensor to avoid modifying the original
        result = tensor.clone()
        
        # Apply adaptive scaling if enabled
        scale_factor = 1.0
        if self.use_adaptive_scale:
            scale_factor = self._compute_adaptive_scale(tensor)
            result = result * scale_factor
        
        # Scale to fixed-point representation
        result = result * self.scale
        
        # Apply FlexRound rounding scheme
        result = self._flex_rounding(result)
        
        # Scale back to floating point
        result = result / self.scale
        
        # Apply adaptive scaling correction if enabled
        if self.use_adaptive_scale:
            result = result / scale_factor
        
        # Clamp to representable range
        result = torch.clamp(result, self.min_val, self.max_val)
        
        return result


class FLEXROUND:
    """
    A class that combines pruning and FlexRound quantization.
    Similar to HWPQ but using FlexRound for quantization.
    """
    def __init__(self, alpha=0.5, beta=0.25, bit_width=8, frac_width=4, structured_sparsity=False):
        """
        Initialize FLEXROUND with parameters
        
        Args:
            alpha: FlexRound parameter controlling interpolation
            beta: FlexRound parameter controlling range preservation
            bit_width: Total bit width for quantization
            frac_width: Fractional bit width for quantization
            structured_sparsity: Whether to use structured sparsity
        """
        self.alpha = alpha
        self.beta = beta
        self.bit_width = bit_width
        self.frac_width = frac_width
        self.structured_sparsity = structured_sparsity
        self.flexround = FlexRound(bit_width, frac_width, alpha, beta)
        
    def compute_contribution(self, weights):
        """
        Compute contribution metric L = w_i^2 / (1 - x_i^2/S) for each weight
        
        Args:
            weights: Tensor of weights in a single layer
            
        Returns:
            Tensor of contribution metrics for each weight
        """
        # Compute S (sum of squared weights)
        S = torch.sum(weights**2)
        
        # Compute the contribution metric for each weight
        x_squared = weights**2
        denominator = 1 - x_squared / (S + 1e-10)  # Add epsilon to avoid division by zero
        # Avoid division by zero or negative values
        denominator = torch.clamp(denominator, min=1e-10)
        
        contributions = x_squared / denominator
        return contributions, S
    
    def prune_and_quantize_weights(self, weights, sparsity_level=0.5):
        """
        Apply FLEXROUND to prune and quantize weights
        
        Args:
            weights: The weight tensor to be pruned and quantized
            sparsity_level: Target sparsity level (0.0 to 1.0)
            
        Returns:
            Pruned and quantized tensor and mask
        """
        # Make a copy of the weights to modify
        pruned_weights = weights.clone()
        mask = torch.ones_like(weights, dtype=torch.bool)
        
        # Track statistics
        total_weights = weights.numel()
        total_kept = 0
        
        print(f"\nFLEXROUND pruning details:")
        print(f"  Input tensor shape: {weights.shape}")
        print(f"  Target sparsity: {sparsity_level:.2%}")
        print(f"  Structured sparsity: {self.structured_sparsity}")
        
        # Process each row independently
        for i in range(weights.shape[0]):
            row_weights = weights[i].flatten()
            
            # Get contribution metrics
            contributions, S = self.compute_contribution(row_weights)
            
            # Create a mask for this row
            row_mask = torch.ones_like(row_weights, dtype=torch.bool)
            
            # Count for achieving target sparsity
            n_weights = row_weights.numel()
            target_prune = int(n_weights * sparsity_level)
            
            if target_prune >= n_weights:
                # Avoid pruning all weights
                target_prune = max(0, n_weights - 1)
            
            pruned_count = 0
            if self.structured_sparsity and n_weights >= 4 and abs(sparsity_level - 0.5) < 0.01:
                # Only use 2:4 structured sparsity when sparsity is close to 50%
                for start_idx in range(0, n_weights, 4):
                    end_idx = min(start_idx + 4, n_weights)
                    chunk_size = end_idx - start_idx
                    
                    if chunk_size < 4:  # Handle incomplete chunks differently
                        if chunk_size > 1:  # If at least 2 weights, prune proportionally
                            prune_in_chunk = max(1, int(chunk_size * 0.5))  # Prune ~50%
                            group_contrib = contributions[start_idx:end_idx]
                            _, indices = torch.topk(group_contrib, prune_in_chunk, largest=False)
                            indices = indices + start_idx
                            row_mask[indices] = 0
                            pruned_count += prune_in_chunk
                    else:
                        # Full chunk of 4 - prune 2
                        group_contrib = contributions[start_idx:end_idx]
                        _, indices = torch.topk(group_contrib, 2, largest=False)
                        indices = indices + start_idx
                        row_mask[indices] = 0
                        pruned_count += 2
            else:
                # For unstructured pruning or non-50% sparsity
                # Sort contributions to ensure we prune exactly the target number
                sorted_indices = torch.argsort(contributions)
                
                # Prune the weights with lowest contributions up to target sparsity
                prune_indices = sorted_indices[:target_prune]
                row_mask[prune_indices] = 0
                pruned_count = len(prune_indices)
            
            # Ensure we're not pruning everything
            if (row_mask == 0).all():
                # Keep at least one weight (the one with highest contribution)
                max_idx = torch.argmax(contributions)
                row_mask[max_idx] = 1
                pruned_count -= 1
            
            # Quantize the remaining weights using FlexRound
            result = torch.zeros_like(row_weights)
            # Apply the mask
            masked_weights = row_weights * row_mask
            # Quantize the masked weights using FlexRound
            result[row_mask] = self.flexround.quantize(masked_weights[row_mask])
            
            # Count non-zeros after quantization
            nonzeros_after_quant = (result != 0).sum().item()
            kept_weights = row_mask.sum().item()
            
            # Print row statistics
            if i < 3 or i == weights.shape[0] - 1:  # Print first 3 rows and last row
                print(f"  Row {i}: kept {kept_weights}/{n_weights} weights " 
                    f"({kept_weights/n_weights:.2%}), "
                    f"non-zeros after quant: {nonzeros_after_quant}")
            elif i == 3:
                print("  ...")
            
            # Update total statistics
            total_kept += kept_weights
            
            # Save the result and reshape the mask
            pruned_weights[i] = result.reshape(weights[i].shape)
            mask[i] = row_mask.reshape(weights[i].shape)
        
        # Overall statistics
        actual_sparsity = 1 - (total_kept / total_weights)
        print(f"  Overall: kept {total_kept}/{total_weights} weights, "
            f"sparsity = {actual_sparsity:.4f}")
        
        return pruned_weights, mask

def flexround_pruning(tensor: torch.Tensor, info: dict, sparsity: float) -> torch.Tensor:
    """
    FlexRound pruning ranking function for MASE pruning framework.
    
    Args:
        tensor: Weight tensor to be pruned
        info: Dictionary with metadata for the tensor
        sparsity: Target sparsity level
        
    Returns:
        Boolean mask indicating which weights to keep (True) or prune (False)
    """
    structured_sparsity = info.get("structured_sparsity", True)
    alpha = info.get("alpha", 0.5)
    beta = info.get("beta", 0.25)
    bit_width = info.get("bit_width", 8)
    frac_width = info.get("frac_width", 4)
    
    flexround = FLEXROUND(
        alpha=alpha, 
        beta=beta, 
        bit_width=bit_width, 
        frac_width=frac_width,
        structured_sparsity=structured_sparsity
    )
    
    # Apply FLEXROUND to get pruned weights and mask
    _, mask = flexround.prune_and_quantize_weights(tensor, sparsity)
    
    return mask

# For use with quantized modules
class FlexRoundParameterization(torch.nn.Module):
    """
    Parametrization for FLEXROUND. This applies both pruning and FlexRound quantization.
    """
    def __init__(self, mask, alpha=0.5, beta=0.25, bit_width=8, frac_width=4):
        super().__init__()
        self.register_buffer("mask", mask)
        self.flexround = FlexRound(
            bit_width=bit_width,
            frac_width=frac_width,
            alpha=alpha,
            beta=beta
        )
        
    def forward(self, x):
        assert self.mask.shape == x.shape
        pruned = self.mask * x
        quantized = self.flexround.quantize(pruned)
        return quantized
        
    def state_dict(self, *args, **kwargs):
        # Avoid double saving masks
        return {} 