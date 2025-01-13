import argparse
from mlx_lm import load, generate
import mlx.core as mx
from typing import get_type_hints
from pathlib import Path
from enum import Enum
import logging
from transformers import AutoConfig
import tiktoken

class PromptMode(str, Enum):
    NORMAL = "normal"
    MASTER = "master"
    RAW = "raw"
    
def debug_tokenizer_type(tokenizer):
    """Debug what type of tokenizer we're actually using."""
    print("\nTokenizer Debug:")
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"Base tokenizer class: {tokenizer._tokenizer.__class__.__name__ if hasattr(tokenizer, '_tokenizer') else None}")
    
    # Check if tiktoken is being used
    if hasattr(tokenizer, 'tokenizer'):
        if isinstance(tokenizer.tokenizer, tiktoken.Encoding):
            print("✓ Using tiktoken Encoding")
            print(f"Tiktoken encoding name: {tokenizer.tokenizer.name}")
            print(f"Tiktoken merges: {len(tokenizer.tokenizer._mergeable_ranks)}")
        else:
            print(f"✗ Not using tiktoken, found: {type(tokenizer.tokenizer)}")
            
    # Test tokenization
    test_text = "Hello世界"  # Test both English and Chinese
    print(f"\nTest tokenization of '{test_text}':")
    try:
        tokens = tokenizer.encode(test_text)
        print(f"Tokens: {tokens}")
        print(f"Token bytes: {[tokenizer.decode([t]) for t in tokens]}")
    except Exception as e:
        print(f"Tokenization error: {e}")

def print_tokens(tokenizer, text: str, label: str = ""):
    """Helper to show both token IDs and special tokens"""
    if isinstance(text, list):
        tokens = text
    else:
        tokens = tokenizer.encode(text)
    
    decoded = []
    for t in tokens:
        # Try to identify special tokens
        token_str = tokenizer.decode([t])
        if t >= 127957:  # SPECIAL_START_ID
            decoded.append(f"[{token_str}]")
        else:
            decoded.append(token_str)
            
    print(f"\n{label} Tokens:")
    print(f"IDs: {tokens}")
    print(f"Decoded: {''.join(decoded)}")
    print("-" * 80)

def format_raw_message(text: str, role: str = "user") -> str:
    """Format message with correct special tokens based on role."""
    if role == "system":
        return f"<|startoftext|>{text}<|extra_4|>"
    elif role == "user":
        return f"<|startoftext|>{text}<|extra_0|>"
    elif role == "assistant":
        # Use extra_5 instead of eos if that's what the model expects
        return f"{text}<|extra_5|>"
    return text

def format_for_model(tokenizer, text: str, mode: PromptMode, use_chat_template: bool = False) -> str:
    """Format text with appropriate template and tokens."""
    # First get the template
    prompt = get_template_prompt(text, mode)
    
    if mode == PromptMode.RAW:
        # In raw mode, just add the single startoftext and extra_0
        return f"<|startoftext|>{text}<|extra_0|>"
        
    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return prompt

def debug_chat_format(tokenizer, messages, label="Chat Format Debug"):
    """Debug the chat template formatting."""
    if hasattr(tokenizer, "chat_template"):
        print(f"\n{label}:")
        print("Raw messages:", messages)
        print("\nChat template:", tokenizer.chat_template)
        
        # Show both raw formatted and tokenized versions
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("\nFormatted message (raw):")
        print(formatted)
        
        # Show token breakdown
        print("\nToken breakdown:")
        parts = formatted.split(">")  # Split on special token marker
        for part in parts:
            if part:
                if part.startswith("<|"):
                    print(f"Special token: {part}>'")
                else:
                    print(f"Content: '{part}'")
                    
        print_tokens(tokenizer, formatted, "Full token sequence")
    else:
        print("\nNo chat template found in tokenizer")

def get_template_prompt(input_prompt: str, mode: PromptMode) -> str:
    """Get the appropriate template prompt based on mode."""
    
    if mode == PromptMode.RAW:
        return input_prompt
        
    template = f"""{mode.title()} mode - Video Recaption Task:

You are a large language model specialized in rewriting video descriptions. Your task is to modify the input description.

0. Preserve ALL information, including style words and technical terms.

1. If the input is in Chinese, translate the entire description to English.

2. If the input is just one or two words describing an object or person, provide a brief, simple description focusing on basic visual characteristics. Limit the description to 1-2 short sentences.

3. If the input does not include style, lighting, atmosphere, you can make reasonable associations.

4. Output ALL must be in English.

Given Input:
input: "{input_prompt}"
"""
    return template

def main():
    parser = argparse.ArgumentParser(description="Generate text using mlx_lm")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model folder"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Shrek playing a violin, rendered as The Legend of Zelda N64 videogame",
        help="Prompt for text generation"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[m.value for m in PromptMode],
        default="normal",
        help="Prompt formatting mode"
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Use the model's chat template"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top p for sampling"
    )
    parser.add_argument(
        "--debug-all",
        action="store_true",
        help="Print all debug information"
    )
    parser.add_argument(
        "--debug-prompt",
        action="store_true",
        help="Debug only prompt formatting"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading tokenizer"
    )

    args = parser.parse_args()

    # Set logging level to debug
    logging.basicConfig(level=logging.DEBUG)

    # Load the model config with trust_remote_code
    config = AutoConfig.from_pretrained(
        args.model, 
        trust_remote_code=args.trust_remote_code
    )

    # Load model
    print(f"\nLoading model from {args.model}")
    model, tokenizer = load(
        args.model,
        model_config=config,
        tokenizer_config={
            "trust_remote_code": args.trust_remote_code,
            "use_fast": False,
            "eos_token": None,  # Remove the eos_token
        }
    )
    # After loading model and tokenizer
    debug_tokenizer_type(tokenizer)

    # Log tokenizer loading for debugging
    logging.debug(f"Tokenizer loaded with config: {tokenizer.__dict__}")

    # Convert mode string to enum
    mode = PromptMode(args.mode.lower())
    
    if args.debug_all or args.debug_prompt:
        print("\nModel Configuration:")
        print(f"Mode: {mode.value}")
        print(f"Using chat template: {args.use_chat_template}")
        print(f"Special tokens map: {tokenizer.special_tokens_map}")
        
        # Show raw prompt
        print("\nRaw input prompt:", args.prompt)
        print_tokens(tokenizer, args.prompt, "Raw Input Tokens")
        
        # Show template
        template = get_template_prompt(args.prompt, mode)
        print("\nTemplate:")
        print(template)
        print_tokens(tokenizer, template, "Template Tokens")
        
        if args.use_chat_template:
            messages = [{"role": "user", "content": template}]
            debug_chat_format(tokenizer, messages)
    
    # Format prompt for generation
    prompt = format_for_model(tokenizer, args.prompt, mode, args.use_chat_template)
    
    if args.debug_prompt:
        print("\nFinal prompt for generation:")
        print(prompt)
        print_tokens(tokenizer, prompt, "Final Prompt Tokens")
        return  # Stop here if only debugging prompt

    # Generate parameters
    generate_kwargs = {
        "max_tokens": args.max_tokens,
        "temp": args.temp,
        "top_p": args.top_p,
        "verbose": args.debug_all
    }
    
    # Get valid parameters for generate from its signature
    valid_generate_kwargs = {
        k: v for k, v in generate_kwargs.items() if k in get_type_hints(generate)
    }

    # Generate
    print("\nGenerating response...")
    response = generate(model, tokenizer, prompt=prompt, **valid_generate_kwargs)

    # Output
    if args.debug_all:
        print_tokens(tokenizer, response, "Generated Response")
        print("\nFinal Response:")
    print(response)

if __name__ == "__main__":
    main()