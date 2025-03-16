# FlexRound: Flexible Rounding Quantization for MASE

FlexRound is a quantization method implemented for the MASE framework that provides improved quantization performance through adaptive rounding. This implementation combines pruning and quantization in a single method, similar to HWPQ.

## Features

- Adjustable rounding behavior controlled by α (alpha) and β (beta) parameters
- Fixed-point quantization with configurable bit width and fractional width
- Optional adaptive scaling based on tensor statistics
- Integrated with MASE pruning framework for combined pruning and quantization
- Support for structured sparsity (2:4 pattern when sparsity=0.5)

## How FlexRound Works

FlexRound provides a flexible rounding approach that interpolates between nearest rounding and probabilistic rounding:

1. **Standard rounding**: Round to nearest integer (0.5 threshold)
2. **Probabilistic rounding**: Round up with probability equal to the fractional part
3. **FlexRound**: Interpolate between these two methods based on:
   - The magnitude of the value (relative to the representable range)
   - The alpha and beta parameters

This approach helps:
- Maintain statistical properties of the original distribution
- Reduce quantization error for values of different magnitudes
- Preserve the dynamic range of the original weights

## Using FlexRound

### As a Pruning Method

You can use FlexRound as a pruning method in your pruning config:

```json
{
  "method": "flexround",
  "sparsity": 0.5,
  "structured_sparsity": true,
  "alpha": 0.5,
  "beta": 0.25,
  "bit_width": 8,
  "frac_width": 4
}
```

### As a Quantization Method

You can use FlexRound as a quantization method in your quantization config:

```json
{
  "by": "type",
  "default": {
    "config": {
      "name": "flexround"
    }
  },
  "linear": {
    "config": {
      "name": "flexround",
      "weight_width": 8,
      "weight_frac_width": 4,
      "weight_alpha": 0.5,
      "weight_beta": 0.25,
      "data_in_width": 8,
      "data_in_frac_width": 4,
      "data_in_alpha": 0.5,
      "data_in_beta": 0.25,
      "bias_width": 8,
      "bias_frac_width": 4,
      "bias_alpha": 0.5,
      "bias_beta": 0.25
    }
  }
}
```

### Combined Pruning and Quantization

For the best results, use both pruning and quantization passes in sequence:

```python
# Apply pruning
mg, _ = passes.prune_transform_pass(
    mg,
    pass_args={
        "weight": {
            "method": "flexround",
            "sparsity": 0.5,
            "structured_sparsity": True,
            "alpha": 0.5,
            "beta": 0.25,
            "bit_width": 8,
            "frac_width": 4
        }
    }
)

# Apply quantization (optional, for finer control)
mg, _ = passes.quantize_transform_pass(
    mg,
    pass_args={
        "by": "type",
        "linear": {
            "config": {
                "name": "flexround",
                "weight_width": 8,
                "weight_frac_width": 4,
                "weight_alpha": 0.5,
                "weight_beta": 0.25,
                "data_in_width": 8,
                "data_in_frac_width": 4,
                "data_in_alpha": 0.5,
                "data_in_beta": 0.25
            }
        }
    }
)
```

## Example Usage

Check the example script `flexround_example.py` to see FlexRound in action:

```bash
python docs/custom_scripts_nyal/flexround_example.py
```

This will demonstrate:
1. Basic FlexRound quantization
2. Combined pruning and quantization
3. Integration with MASE

## Parameters

- **bit_width**: Total bit width for quantization (default: 8)
- **frac_width**: Fractional bit width for quantization (default: 4)
- **alpha**: Controls the maximum amount of interpolation between rounding modes (default: 0.5)
- **beta**: Controls how the interpolation weight varies with value magnitude (default: 0.25)
- **structured_sparsity**: Whether to use structured sparsity patterns (default: False)

## Customization

You can customize the FlexRound implementation by:

1. Modifying the `_flex_rounding` method to implement different interpolation schemes
2. Changing the `_compute_adaptive_scale` method to use different scaling strategies
3. Implementing additional sparsity patterns in the `prune_and_quantize_weights` method 