# HunYuan MoE: Weight Conversion and Loading Debug Case Study

## Context and Problem

When converting HunYuanVideo-PromptRewrite, a Mixture of Experts (MoE) model, from PyTorch to MLX-LM, we encountered the following issues:

1. The MoE components initialized correctly but became None during inference
2. LayerNorm weights failed to load with path mismatches
3. Missing bias terms caused weight loading to fail

The root cause was a mismatch between how PyTorch and MLX handle MoE architecture weights, combined with silent failures in the weight loading system.

## Conversion Chain

```
PyTorch Binary
    → Hugging Face Safetensors
    → MLX-LM Format
    → MLX Runtime
```

## Root Causes

1. **Initial MoE Weight Structure**
   - PyTorch treats experts separately with individual tensors per expert
   ```
   model.layers.0.mlp.experts.0.up_proj.weight
   model.layers.0.mlp.experts.1.up_proj.weight
   ```
   - MLX expects stacked experts in single tensors
   ```
   model.layers.0.mlp.switch_mlp.up_proj.weight [num_experts, dim...]
   ```

2. **LayerNorm Architecture Differences**
   ```python
   # Original (PyTorch)
   input_layernorm = nn.LayerNorm(args.hidden_size)  # With bias
   
   # MLX-LM Needed
   input_layernorm = nn.LayerNorm(args.hidden_size, bias=False)  # No bias
   ```

3. **Silent Weight Loading Failures**
   - MLX's weight loading requires exact path matches
   - Mismatched paths caused components to silently become None
   - No immediate error at initialization time

## The Fix

1. **Corrected Weight Path Structure**
   ```python
   def sanitize(self, weights):
       # Remove bias terms globally
       weights = {k: v for k, v in weights.items() if not k.endswith('.bias')}
       
       if any('switch_mlp' in k for k in weights.keys()):
           # Weights already in stacked format, verify structure
           for l in range(self.args.num_hidden_layers):
               prefix = f"model.layers.{l}.mlp"
               required = {
                   'switch_mlp': ['up_proj', 'down_proj', 'gate_proj'],
                   'shared_mlp': ['up_proj', 'down_proj', 'gate_proj'],
                   'gate': ['wg']
               }
               # Verification code...
       return weights
   ```

2. **Modified Layer Architecture**
   ```python
   class DecoderLayer(nn.Module):
       def __init__(self, args: ModelArgs, kv_proj: bool):
           super().__init__()
           # No-bias layer norms to match weights
           self.input_layernorm = nn.LayerNorm(args.hidden_size, bias=False)
           self.post_attention_layernorm = nn.LayerNorm(args.hidden_size, bias=False)
   ```

## What Should Have Been Done Differently

### 1. Initial Conversion
- Inspect PyTorch weight structure before conversion
```bash
for f in *.bin; do 
  echo "=== $f ==="
  python -c "import torch; print(torch.load('$f').keys())"
done
```

### 2. Safetensors Validation
- Validate MoE structure after HF conversion
```python
def verify_moe_structure(path):
    weights = safetensors.torch.load_file(path)
    moe_keys = [k for k in weights.keys() if 'mlp' in k]
    print(f"Found {len(moe_keys)} MoE keys")
    for k in moe_keys[:5]:
        print(f"{k}: {weights[k].shape}")
```

### 3. MLX-LM Development
- Add explicit debug logging during initialization:
```python
class MoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        logging.info("Initializing MoeBlock")
        # Initialize components
        if not hasattr(self, 'switch_mlp'):
            logging.error("Failed to create switch_mlp")
```

- Add weight structure verification:
```python
def verify_weights(model_path: str):
    weights = {}
    for shard in glob.glob(f"{model_path}/model-*.safetensors"):
        weights.update(mx.load(shard))
        
    # Check for required MoE components
    components = {
        'switch_mlp': ['up_proj', 'down_proj', 'gate_proj'],
        'shared_mlp': ['up_proj', 'down_proj', 'gate_proj'],
        'gate': ['wg']
    }
    
    for layer in range(64):
        for comp, parts in components.items():
            prefix = f"model.layers.{layer}.mlp.{comp}"
            for part in parts:
                key = f"{prefix}.{part}.weight"
                if key not in weights:
                    print(f"Missing: {key}")
```

### 4. More Robust Component Tracking
- Add component state verification in MoeBlock:
```python
class MoeBlock(nn.Module):
    def verify_state(self):
        """Verify MoE component state"""
        if not hasattr(self, 'switch_mlp'):
            return False
        if not hasattr(self, 'shared_mlp'):
            return False
        if not hasattr(self, 'gate'):
            return False
        return True
```

### 5. Development Process Changes
1. Always inspect weight structures before conversion:
   ```bash
   python inspect_weights.py original_model.bin
   python inspect_weights.py converted.safetensors
   ```

2. Use a staged conversion with validation at each step
   ```
   PyTorch Weights 
     → Verify expert structure
     → HuggingFace Safetensors 
     → Verify MoE components
     → MLX-LM Format 
     → Test basic inference
   ```

3. Add debug logging for weight loading
   - Track which weights are found/missing
   - Log actual vs expected paths
   - Verify component initialization

## Lessons Learned

1. **Architecture Differences Matter**
   - Different frameworks handle MoE differently
   - Weight structures need explicit mapping
   - Silent failures need proactive detection

2. **Validation at Every Step**
   - Check weight structure before conversion
   - Verify component existence after loading
   - Test with small inputs before full inference

3. **Debug Infrastructure**
   - Add component state verification
   - Track initialization success
   - Log weight paths and shapes

4. **Framework Specifics**
   - MLX expects exact path matches
   - Layer architecture must match weights
   - Handle framework-specific bias patterns

## Future Work Recommendations

1. Add automatic MoE structure verification to `convert.py`
2. Create weight structure validator for MLX-LM
3. Add explicit error on weight path mismatches
4. Create test suite for MoE model loading

## References
- MLX-LM weight loading implementation
- HunYuan model architecture
- PyTorch MoE documentation
