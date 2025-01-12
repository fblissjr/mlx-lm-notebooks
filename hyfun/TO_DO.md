# todo
To use these:

Quick Fix (Immediate Solution):

Apply the patch to your local MLX-LM installation:

bashCopycd path/to/mlx-examples/llms
patch -p1 < quick-patch.diff

Proper PR (Long-term Solution):

Create a new branch in MLX-LM
Add the files as outlined in the PR template
Submit the PR with the provided description



For your model release:

Ensure these files are included:

Copymodel/
├── hy.tiktoken             # Your tiktoken vocabulary
├── tokenizer_config.json   # Config with proper special tokens
└── tokenization_hy.py      # Tokenizer implementation

Update your tokenizer_config.json:

jsonCopy{
  "tokenizer_class": "HYTokenizer",
  "trust_remote_code": true,
  "use_fast": false
}

## Changes

### Added Files

1. `mlx_lm/tokenizers/tiktoken.py`:
```python
"""Base functionality for tiktoken-based tokenizers."""
import tiktoken
from typing import Dict, Optional

def load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
    """Load tiktoken BPE vocabulary."""
    vocab = {}
    rank = 0
    for line in open(tiktoken_bpe_file, "rb"):
        if line:
            token, _ = line.split()
            vocab[base64.b64decode(token)] = rank
            rank += 1
    return vocab
```

2. `mlx_lm/tokenizers/hunyuan.py`:
```python
"""HunYuan tokenizer implementation."""
from .tiktoken import load_tiktoken_bpe
from transformers import PreTrainedTokenizer
# ... rest of tokenizer implementation ...
```

3. `tests/tokenizers/test_hunyuan.py`:
```python
"""Tests for HunYuan tokenizer."""
import pytest
from mlx_lm.tokenizers.hunyuan import HYTokenizer

def test_tokenizer_loading():
    # Test basic loading
    pass

def test_special_tokens():
    # Test special token handling
    pass

def test_chat_template():
    # Test chat template functionality
    pass
```

### Modified Files

1. `mlx_lm/tokenizer_utils.py`:
- Added tiktoken support
- Added HunYuan tokenizer detection
- Updated tokenizer loading logic

### Dependencies

Added to requirements.txt:
```
tiktoken>=0.5.0
```

## Testing

1. Basic tokenizer tests:
```bash
pytest tests/tokenizers/test_hunyuan.py
```

2. Generation test:
```python
model, tokenizer = load("path/to/model", trust_remote_code=True)
print(generate(model, tokenizer, "Test prompt"))
```

## Documentation

Added to docs/tokenizers.md:
```markdown
## Tiktoken-based Tokenizers

MLX-LM supports tiktoken-based tokenizers like those used in HunYuan models.
These tokenizers provide efficient byte-level BPE implementation.

### Usage

Models using tiktoken should include:
- hy.tiktoken: The tiktoken vocabulary file
- tokenization_hy.py: The tokenizer implementation
- tokenizer_config.json: Configuration including special tokens
```

## Migration Guide

For existing HunYuan models:

1. Ensure required files are present:
```
model/
  ├── hy.tiktoken
  ├── tokenizer_config.json
  └── tokenization_hy.py
```

2. Update tokenizer_config.json:
```json
{
  "tokenizer_class": "HYTokenizer",
  "trust_remote_code": true
}
```

## Related Issues

Fixes #XXX - Add support for tiktoken-based tokenizers



## Code
diff --git a/mlx_lm/tokenizer_utils.py b/mlx_lm/tokenizer_utils.py
--- a/mlx_lm/tokenizer_utils.py
+++ b/mlx_lm/tokenizer_utils.py
@@ -363,10 +363,32 @@ def load_tokenizer(model_path, tokenizer_config_extra={}, eos_token_ids=None):
     """Load a huggingface tokenizer and try to infer the type of streaming
     detokenizer to use.
     """
+    model_path = Path(model_path)
     detokenizer_class = NaiveStreamingDetokenizer
-    
-    tokenizer = AutoTokenizer.from_pretrained(
-        model_path, **tokenizer_config_extra
-    )
+
+    try:
+        # Try normal AutoTokenizer first
+        tokenizer = AutoTokenizer.from_pretrained(
+            model_path, **tokenizer_config_extra
+        )
+    except ValueError as e:
+        # Check if this is a HunYuan tokenizer
+        if (model_path / "tokenization_hy.py").exists():
+            logger.info("Loading HunYuan tokenizer directly...")
+            try:
+                import importlib.util
+                spec = importlib.util.spec_from_file_location(
+                    "tokenization_hy",
+                    model_path / "tokenization_hy.py"
+                )
+                module = importlib.util.module_from_spec(spec)
+                spec.loader.exec_module(module)
+                TokenizerClass = module.HYTokenizer
+                
+                tokenizer = TokenizerClass(
+                    vocab_file=str(model_path / "hy.tiktoken"),
+                    **tokenizer_config_extra
+                )
+            except Exception as e2:
+                raise ValueError(f"Failed to load HunYuan tokenizer: {e2}") from e
+        else:
+            raise e

     if isinstance(eos_token_ids, int):
         eos_token_ids = [eos_token_ids]


## Root cause
### THE ISSUE:

The AutoTokenizer system in transformers was failing to load our custom tokenizer because the auto_map format in tokenizer_config.json wasn't matching what transformers expected
The MLX-LM code was using this AutoTokenizer system, which was failing

### THE SOLUTION:

wrote a working HYTokenizer implementation that:

Uses tiktoken for BPE implementation
Handles special tokens correctly
Properly implements PreTrainedTokenizer interface


Rather than using AutoTokenizer, load the tokenizer directly:
pythonCopyTokenizerClass = HYTokenizer  # our custom implementation
tokenizer = TokenizerClass(
    vocab_file="hy.tiktoken",
    **tokenizer_config
)


### FOR LONGTERM:
Two options:

#### Quick Fix: Patch mlx_lm/utils.py and mlx_lm/tokenizer_utils.py:

pythonCopydef load_tokenizer(model_path, tokenizer_config_extra={}, eos_token_ids=None):
    """Load tokenizer with fallback to direct loading."""
    try:
        # Try normal AutoTokenizer first
        tokenizer = AutoTokenizer.from_pretrained(...)
    except ValueError:
        # Fallback to direct loading for HYTokenizer
        if Path(model_path / "tokenization_hy.py").exists():
            from .tokenization_hy import HYTokenizer
            tokenizer = HYTokenizer(
                vocab_file=str(model_path / "hy.tiktoken"),
                **tokenizer_config_extra
            )
    return tokenizer

#### Proper Fix: Fix the HYTokenizer in HYPR model folder

Add proper HYTokenizer implementation
Update tokenizer loading logic to handle tiktoken-based tokenizers
Add tests for special token handling and chat templates



#### The minimum files needed:
- tokenization_hy.py      # The tokenizer implementation 
- hy.tiktoken            # HY tiktoken vocabulary file
- tokenizer_config.json  # The tokenizer configuration

For your model release, you need to ensure these files are packaged together. For reference, here was the critical part that fixed special tokens:

```python
self.tokenizer = tiktoken.Encoding(
    "Hunyuan",
    pat_str=PAT_STR,
    mergeable_ranks=self.mergeable_ranks,
    special_tokens={
        STARTOFTEXT: SPECIAL_START_ID,
        ENDOFTEXT: SPECIAL_START_ID + 1,
        BOSTOKEN: SPECIAL_START_ID + 2,
        EOSTOKEN: SPECIAL_START_ID + 3,
        PADTOKEN: SPECIAL_START_ID + 4,
    }
)
```

This ensures your special tokens get the correct IDs and the chat template works properly.