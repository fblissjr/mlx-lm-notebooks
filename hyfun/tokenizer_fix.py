import os
import sys
import json
import shutil
import logging
import importlib
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any

import mlx.core as mx
from transformers import PreTrainedTokenizer
from mlx_lm import load, generate

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_module_contents(module_path: str):
    """Debug what's in the tokenizer module."""
    logger.info(f"Examining module at: {module_path}")
    
    # Read the file content
    with open(module_path, 'r') as f:
        content = f.read()
        
    logger.info("Module content:")
    logger.info("=" * 40)
    logger.info(content[:500] + "...") # Show first 500 chars
    logger.info("=" * 40)
    
    # Try to import
    spec = importlib.util.spec_from_file_location("tokenization_hy", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Check module contents
    logger.info("\nModule attributes:")
    for attr in dir(module):
        if not attr.startswith('__'):
            logger.info(f"- {attr}: {type(getattr(module, attr))}")
            
    return module

def ensure_tokenizer_class(model_path: str):
    """Ensure the tokenizer class is properly defined and available."""
    model_path = Path(model_path)
    tokenizer_file = model_path / "tokenization_hy.py"
    
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_file}")
    
    # Try to debug the module
    module = debug_module_contents(str(tokenizer_file))
    
    # Check for common class definition issues
    with open(tokenizer_file, 'r') as f:
        content = f.read()
        
        # Check if class is defined
        if 'class HYTokenizer' not in content:
            logger.error("HYTokenizer class not found in file!")
            if 'HYTokenizer =' in content:
                logger.error("Found HYTokenizer assignment but no class definition")
                
        # Check imports
        if 'from transformers import PreTrainedTokenizer' not in content:
            logger.warning("PreTrainedTokenizer import missing")
    
    return module

def create_tokenizer(model_path: str) -> PreTrainedTokenizer:
    """Create a tokenizer instance with the correct configuration."""
    model_path = Path(model_path)
    
    # Debug and get the module
    module = ensure_tokenizer_class(model_path)
    
    # Get the Tokenizer class
    TokenizerClass = getattr(module, 'HYTokenizer', None)
    if TokenizerClass is None:
        # Check if it's exported under a different name
        tokenizer_classes = [
            obj for name, obj in module.__dict__.items()
            if isinstance(obj, type) and issubclass(obj, PreTrainedTokenizer)
        ]
        if tokenizer_classes:
            TokenizerClass = tokenizer_classes[0]
            logger.info(f"Found tokenizer class with name: {TokenizerClass.__name__}")
        else:
            raise AttributeError("No tokenizer class found in module")
    
    # Load config
    with open(model_path / "tokenizer_config.json", "r") as f:
        tokenizer_config = json.load(f)
        
    logger.info(f"Creating tokenizer with config: {tokenizer_config}")
    
    # Create tokenizer instance
    tokenizer = TokenizerClass(
        vocab_file=str(model_path / "hy.tiktoken"),
        errors="replace",
        bod_token="<|startoftext|>",
        eod_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        add_bod_token=True,
        add_eod_token=True,
    )
    
    # Add special tokens from config
    if "additional_special_tokens" in tokenizer_config:
        tokenizer.additional_special_tokens = tokenizer_config["additional_special_tokens"]
    
    # Set chat template
    if "chat_template" in tokenizer_config:
        tokenizer.chat_template = tokenizer_config["chat_template"]
        
    return tokenizer

def test_tokenizer(model_path: str) -> None:
    """Test the tokenizer setup and functionality."""
    try:
        logger.info("Creating tokenizer...")
        tokenizer = create_tokenizer(model_path)
        
        logger.info(f"\nCreated tokenizer of type: {type(tokenizer)}")
        logger.info(f"Vocab size: {len(tokenizer)}")
        logger.info(f"Special tokens: {tokenizer.special_tokens_map}")
        
        # Basic encoding test
        test_text = "Hello world"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        logger.info(f"\nBasic test:")
        logger.info(f"Text: {test_text}")
        logger.info(f"Tokens: {tokens}")
        logger.info(f"Decoded: {decoded}")
        
        # Try MLX load
        logger.info("\nTesting MLX integration...")
        model, mlx_tokenizer = load(
            model_path,
            tokenizer_config={
                "trust_remote_code": True,
                "use_fast": False
            }
        )
        
        logger.info("MLX load successful!")
        
    except Exception as e:
        logger.error(f"Error testing tokenizer: {e}", exc_info=True)
        raise

def main():
    if len(sys.argv) != 2:
        print("Usage: python tokenizer_fix.py /path/to/model")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_tokenizer(model_path)

if __name__ == "__main__":
    main()