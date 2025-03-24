# flexround_new_example.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCTC, Wav2Vec2Processor
from datasets import load_dataset, Dataset, DatasetDict
from pyctcdecode import build_ctcdecoder
from tqdm import tqdm

# CHOP imports
from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_logger, get_trainer
from chop.passes.graph.transforms.quantize.quantize import quantize_transform_pass

# FlexRound modules
from chop.nn.quantized.modules import quantized_module_map
from chop.nn.quantized.modules.flexround_modules import (
    LinearFlexRound, Conv2dFlexRound, Conv1dFlexRound
)

logger = get_logger(__name__)
logger.setLevel("INFO")

class FlexRoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Straight-Through Estimator

def calibrate_flexround(model, calibration_data, device="cuda", iterations=500, lr=1e-3):
    """Calibrate FlexRound parameters using layer-wise reconstruction"""
    model = model.to(device)
    model.eval()
    
    # Freeze all parameters except FlexRound parameters
    for name, param in model.named_parameters():
        if not any(p in name for p in ["s1", "S2", "s3", "s4", "act_scale"]):
            param.requires_grad_(False)
            
    # Collect FlexRound layers and create optimizers
    layers = []
    optimizers = []
    for name, module in model.named_modules():
        if isinstance(module, (LinearFlexRound, Conv1dFlexRound, Conv2dFlexRound)):
            layers.append(module)
            params = [module.s1, module.S2]
            if hasattr(module, 's3'): params.append(module.s3)
            if hasattr(module, 's4'): params.append(module.s4)
            if module.act_quant is not None: params.append(module.act_scale)
            optimizers.append(torch.optim.Adam(params, lr=lr))
    
    # Get original layer outputs
    original_outputs = []
    hooks = []
    def hook_fn(module, input, output):
        original_outputs.append(output.detach())
    
    with torch.no_grad():
        for layer in layers:
            hooks.append(layer.register_forward_hook(hook_fn))
        model(calibration_data["input_values"].to(device),
             calibration_data["attention_mask"].to(device)))
        for hook in hooks:
            hook.remove()

    # Layer-wise reconstruction training
    for layer_idx, (layer, optimizer) in enumerate(zip(layers, optimizers)):
        print(f"Calibrating layer {layer_idx+1}/{len(layers)}")
        layer.train()
        
        # Get layer input and original output
        input_hook_value = None
        def input_hook(module, input):
            nonlocal input_hook_value
            input_hook_value = input[0].detach()
        
        hook = layer.register_forward_pre_hook(input_hook)
        with torch.no_grad():
            model(calibration_data["input_values"].to(device),
                 calibration_data["attention_mask"].to(device)))
        hook.remove()
        
        # Training loop
        progress = tqdm(range(iterations), desc=f"Layer {layer_idx+1}")
        for _ in progress:
            optimizer.zero_grad()
            
            # Forward with current quantization parameters
            quant_out = layer(input_hook_value.to(device))
            loss = F.mse_loss(quant_out, original_outputs[layer_idx].to(device))
            
            loss.backward()
            optimizer.step()
            
            # Apply parameter constraints
            with torch.no_grad():
                layer.s1.data.clamp_(min=1e-8)
                layer.S2.data.clamp_(min=1e-8)
                if hasattr(layer, 's3'): layer.s3.data.clamp_(min=1e-8)
                if hasattr(layer, 's4'): layer.s4.data.clamp_(min=1e-8)
                if layer.act_quant is not None: 
                    layer.act_scale.data.clamp_(min=1e-8)
            
            progress.set_postfix(loss=loss.item())

    # Unfreeze parameters
    for param in model.parameters():
        param.requires_grad_(True)
        
    return model

def main():
    # Load model and dataset
    checkpoint = "facebook/wav2vec2-base-960h"
    dataset_name = "nyalpatel/condensed_librispeech_asr"
    
    processor = Wav2Vec2Processor.from_pretrained(checkpoint)
    model = AutoModelForCTC.from_pretrained(checkpoint)
    
    # Prepare data
    dataset = load_dataset(dataset_name, split="validation.clean").select(range(50))
    tokenized_dataset = dataset.map(
        lambda x: processor(
            audio=x["audio"]["array"], 
            sampling_rate=x["audio"]["sampling_rate"],
            text=x["text"],
            return_tensors="pt",
            padding=True
        ),
        remove_columns=["audio", "speaker_id", "file", "id", "chapter_id"]
    ).with_format("torch")

    # Build MASE graph
    mg = MaseGraph(model.wav2vec2)
    dummy_in = {
        "input_values": torch.zeros((1, 16000)),
        "attention_mask": torch.ones((1, 16000), dtype=torch.long)
    }
    mg, _ = passes.init_metadata_analysis_pass(mg)
    mg, _ = passes.add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})

    # Quantization config
    quant_config = {
        "by": "type",
        "default": {"config": {"name": None}},
        "linear": {
            "config": {
                "name": "flexround",
                "weight_width": 8,
                "data_in_width": 8,
                "weight_only": False
            }
        },
        "conv1d": {
            "config": {
                "name": "flexround",
                "weight_width": 8,
                "data_in_width": 8,
                "weight_only": False
            }
        }
    }

    # Apply quantization
    mg_quant, _ = quantize_transform_pass(mg, quant_config)

    # Calibration
    calibration_data = {
        "input_values": torch.stack(tokenized_dataset["input_values"]),
        "attention_mask": torch.stack(tokenized_dataset["attention_mask"])
    }
    mg_quant.model = calibrate_flexround(
        mg_quant.model, 
        calibration_data,
        iterations=500,
        lr=1e-3
    )

    # Rebuild full model
    class QuantizedCTC(nn.Module):
        def __init__(self, encoder, head):
            super().__init__()
            self.encoder = encoder
            self.head = head
            self.decoder = build_ctcdecoder(processor.tokenizer.get_vocab())

        def forward(self, input_values, attention_mask=None, labels=None):
            hidden = self.encoder(input_values, attention_mask=attention_mask).last_hidden_state
            logits = self.head(hidden)
            return {"logits": logits}

    quant_model = QuantizedCTC(mg_quant.model, model.lm_head)

    # Evaluation
    trainer = get_trainer(
        model=quant_model,
        tokenized_dataset=DatasetDict({"train": tokenized_dataset, "test": tokenized_dataset}),
        tokenizer=processor.tokenizer,
        data_collator=lambda x: processor.pad(x, return_tensors="pt"),
        evaluate_metric="wer"
    )
    
    print("\n===== Evaluating =====")
    results = trainer.evaluate()
    print(f"Final WER: {results['eval_wer']:.2f}%")

if __name__ == "__main__":
    main()