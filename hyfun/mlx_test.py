#!/usr/bin/env python3
"""
Modified MLX testing script with patched tokenizer loading.
"""

import os
import sys 
import json
import logging
import importlib.util
from pathlib import Path
from typing import Tuple
import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm import generate 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def load_model_only(model_path: str, **kwargs):
    """Load only the model part from MLX."""
    try:
        from mlx_lm.utils import load_model
        model, config = load_model(Path(model_path), **kwargs)
        return model, config
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def load_tokenizer_direct(model_path: Path):
    """Load the tokenizer directly from our implementation."""
    # Load the tokenizer class
    spec = importlib.util.spec_from_file_location(
        "tokenization_hy",
        model_path / "tokenization_hy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    TokenizerClass = module.HYTokenizer

    # Get config
    with open(model_path / "tokenizer_config.json") as f:
        tokenizer_config = json.load(f)

    # Create tokenizer
    tokenizer = TokenizerClass(
        vocab_file=str(model_path / "hy.tiktoken"),
        **{k: v for k, v in tokenizer_config.items() 
           if k not in ['auto_map', 'tokenizer_class']}
    )

    return tokenizer

def patched_load(model_path: str, **kwargs) -> Tuple:
    """Patched version of mlx_lm.load that uses our direct tokenizer."""
    model_path = Path(model_path)
    
    # Load model
    model, _ = load_model_only(model_path, **kwargs)
    
    # Load tokenizer directly
    tokenizer = load_tokenizer_direct(model_path)
    
    return model, tokenizer

def test_generation(model_path: Path):
    """Test the model with our patched loading."""
    print("\nTesting generation with patched loader...")
    
    try:
        # Load with our patched function
        model, tokenizer = patched_load(model_path)
        
        print("Successfully loaded model and tokenizer")
        print(f"Tokenizer type: {type(tokenizer)}")
        
        # Test chat template
        messages = [
            {"role": "user", "content": "Write a video description."},
            {"role": "assistant", "content": "Shrek playing a violin while walking down the runway"}
        ]
        
        if hasattr(tokenizer, "chat_template"):
            print("\nTesting chat template:")
            formatted = tokenizer.apply_chat_template(messages)
            print(formatted)
            tokens = tokenizer.encode(formatted)
            print(f"Tokens: {tokens}")
            
        # Test generation
        prompt = "Shrek playing a violin"
        print(f"\nTesting generation with prompt: {prompt}")
        
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=512,
            verbose=True
        )
        
        print(f"Generated response: {response}")
        
    except Exception as e:
        print(f"Error in generation test: {e}")
        raise

def main():
    if len(sys.argv) != 2:
        print("Usage: python mlx_test.py /path/to/model")
        sys.exit(1)
        
    model_path = Path(sys.argv[1])
    test_generation(model_path)

if __name__ == "__main__":
    main()