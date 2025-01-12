#!/usr/bin/env python3
"""
Test script for HunYuan tokenizer implementation.
Usage: python test_tokenizer.py /path/to/model/directory
"""

import os
import sys 
import json
import logging
import importlib.util
from pathlib import Path
from mlx_lm import load, generate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def print_separator():
    print("\n" + "="*80 + "\n")

def verify_files(model_path: Path):
    """Check required files exist."""
    print("Verifying required files...")
    
    required_files = [
        "tokenization_hy.py",
        "hy.tiktoken",
        "tokenizer_config.json",
        "config.json"
    ]
    
    files_found = {}
    for file in required_files:
        path = model_path / file
        exists = path.exists()
        files_found[file] = exists
        print(f"{'✓' if exists else '✗'} Found {file}")
        if exists and file == "tokenizer_config.json":
            with open(path) as f:
                print("\nTokenizer config:")
                print(json.dumps(json.load(f), indent=2))
                
    print_separator()
    return all(files_found.values())

def load_tokenizer_class(model_path: Path):
    """Load the HYTokenizer class directly from the source file."""
    print("Loading tokenizer class directly...")
    
    tokenizer_path = model_path / "tokenization_hy.py"
    
    # Load the module directly
    spec = importlib.util.spec_from_file_location(
        "tokenization_hy",
        tokenizer_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Print available attributes
    print("\nModule contents:")
    for attr in dir(module):
        if not attr.startswith('__'):
            print(f"- {attr}")
            
    if hasattr(module, 'HYTokenizer'):
        print("\nFound HYTokenizer class")
        return module.HYTokenizer
    else:
        raise ValueError("HYTokenizer class not found in module")

def test_direct_tokenizer(model_path: Path, TokenizerClass):
    """Test loading tokenizer directly."""
    print("\nTesting direct tokenizer creation...")
    
    try:
        # Load configs
        with open(model_path / "tokenizer_config.json") as f:
            tokenizer_config = json.load(f)
            
        # Create tokenizer instance
        tokenizer = TokenizerClass(
            vocab_file=str(model_path / "hy.tiktoken"),
            **tokenizer_config
        )
        
        print(f"Successfully created tokenizer instance")
        print(f"Vocab size: {len(tokenizer)}")
        print(f"Special tokens map: {tokenizer.special_tokens_map}")
        
        # Test basic tokenization
        test_texts = [
            "Hello, this is a test sentence!",
            "你好，世界",
            "<|startoftext|>Test prompt<|extra_0|>"
        ]
        
        print("\nTesting tokenization:")
        for text in test_texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            print(f"\nInput:   {text}")
            print(f"Tokens:  {tokens}")
            print(f"Decoded: {decoded}")
        
        # Return tokenizer for MLX testing
        return tokenizer
            
    except Exception as e:
        print(f"Error testing tokenizer: {e}")
        raise

def test_mlx_integration(model_path: Path):
    """Test tokenizer with MLX model loading."""
    print("\nTesting MLX integration...")
    
    try:
        model, tokenizer = load(
            model_path,
            tokenizer_config={
                "trust_remote_code": True,
                "use_fast": False
            }
        )
        
        print(f"Successfully loaded model and tokenizer")
        print(f"Tokenizer type: {type(tokenizer)}")
        
        # Test template if available
        if hasattr(tokenizer, "chat_template"):
            print("\nTesting chat template:")
            messages = [
                {"role": "user", "content": "Write a video description."},
                {"role": "assistant", "content": "A scenic mountain landscape."}
            ]
            formatted = tokenizer.apply_chat_template(messages)
            print("\nFormatted chat:")
            print(formatted)
            tokens = tokenizer.encode(formatted)
            print(f"Tokens: {tokens}")
        
        # Test generation
        prompt = "Describe a mountain scene:"
        print(f"\nTesting generation with prompt: {prompt}")
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=50,
            verbose=True
        )
        print(f"Generated: {response}")
        
    except Exception as e:
        print(f"Error testing MLX integration: {e}")
        raise

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_tokenizer.py /path/to/model")
        sys.exit(1)
        
    model_path = Path(sys.argv[1])
    
    try:
        # Verify files
        if not verify_files(model_path):
            print("Missing required files!")
            sys.exit(1)
            
        # Load tokenizer class
        TokenizerClass = load_tokenizer_class(model_path)
        
        # Test direct instantiation
        tokenizer = test_direct_tokenizer(model_path, TokenizerClass)
        
        # Test with MLX
        test_mlx_integration(model_path)
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\nTesting failed with error: {e}")
        raise

if __name__ == "__main__":
    main()