import argparse
from mlx_lm import load, generate, stream_generate
import mlx.core as mx
from typing import get_type_hints, List, Union 
from pathlib import Path
import logging
from transformers import AutoConfig
import tiktoken
from enum import Enum

# # Define the prompt template for video description
# PROMPT_TEMPLATE_ENCODE_VIDEO = (
#     "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
#     "1. The main content and theme of the video."
#     "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
#     "3. Actions, events, behaviors, temporal relationships, physical movement changes of the objects."
#     "4. background environment, light, style and atmosphere."
#     "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
#     "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
# )
PROMPT_TEMPLATE_ENCODE_VIDEO=("{}")

def debug_prompt(tokenizer, prompt: str, use_chat_template: bool):
    """Show exactly what's being sent to the model"""
    print("\nPrompt Debug:")
    print("Raw input:", repr(prompt))
    
    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("\nAfter chat template:")
        print(repr(formatted))
        tokens = tokenizer.encode(formatted)
    else:
        tokens = tokenizer.encode(prompt)
        
    print("\nFinal tokens:")
    print([tokenizer.decode([t]) for t in tokens])
    return tokens

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
    test_text = "Hello world"  # Test both English and Chinese
    print(f"\nTest tokenization of '{test_text}':")
    try:
        tokens = tokenizer.encode(test_text)
        print(f"Tokens: {tokens}")
        print(f"Token bytes: {[tokenizer.decode([t]) for t in tokens]}")
    except Exception as e:
        print(f"Tokenization error: {e}")

def print_tokens(tokenizer, text: Union[str, List[int]], label: str = ""):
    """Helper to show both token IDs and special tokens"""
    # Check if the input is a string (needs encoding) or a list of token IDs
    if isinstance(text, str):
        tokens = tokenizer.encode(text)
    else:
        tokens = text

    decoded = []
    for t in tokens:
        # Decode each token ID to its string representation
        token_str = tokenizer.decode([t])

        # Check for specific special tokens and format accordingly
        if t == tokenizer.eos_token_id:
            decoded_token = f"[{tokenizer.eos_token}]"
        elif t == tokenizer.bos_token_id:
            decoded_token = f"[{tokenizer.bos_token}]"
        elif t == tokenizer.pad_token_id:
            decoded_token = f"[{tokenizer.pad_token}]"
        elif token_str in tokenizer.additional_special_tokens:  # Check against the list of special tokens
            decoded_token = f"[{token_str}]"
        else:
            decoded_token = token_str  # Regular token

        decoded.append(decoded_token)

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

def format_for_model(tokenizer, messages, use_chat_template: bool = False, add_generation_prompt: bool = False) -> str:
    """Formats messages using the tokenizer's chat template."""
    if use_chat_template:
        # Apply template without generation prompt first
        formatted = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=add_generation_prompt
        )
        # Debug output before any modifications
        print("\nInitial formatting:")
        print("Raw formatted:", repr(formatted))
        print("Initial tokens:", [tokenizer.decode([t]) for t in tokenizer.encode(formatted)])

        # # Make sure user messages end with extra_0
        # if messages[-1]["role"] == "user":
        #     if not formatted.endswith("<|extra_0|>"):
        #         formatted += "<|extra_0|>"

        # Debug final format
        print("\nFinal formatting:")
        print("Final tokens:", [tokenizer.decode([t]) for t in tokenizer.encode(formatted)])
        
        return formatted
    else:
        return PROMPT_TEMPLATE_ENCODE_VIDEO.format(messages)

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
                    
        print_tokens(tokenizer, tokenizer.encode(formatted), "Full token sequence")
    else:
        print("\nNo chat template found in tokenizer")

def main():
    parser = argparse.ArgumentParser(description="Generate text using mlx_lm")
    parser.add_argument(
        "--model",
        type=str,
        default="~/Storage/hyvideo-promptrewrite-mlx-2bit-32g",
        help="Path to the model folder"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Shrek playing a violin, rendered as The Legend of Zelda N64 videogame",
        help="Prompt for text generation"
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
        default=0,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1,
        help="Top p for sampling"
    )
    parser.add_argument(
        "--debug-all",
        action="store_true",
        default=False,
        help="Print all debug information"
    )
    parser.add_argument(
        "--debug-prompt",
        action="store_true",
        default=False,
        help="Debug only prompt formatting"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading tokenizer"
    )
    parser.add_argument(
        "--generate-type",
        choices=["stream", "regular"],
        default="regular",
        help="Type of generation to use: 'stream' for stream_generate, 'regular' for generate"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print verbose output (default: False)",
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
        model_config=config,  # Pass the pre-loaded config here
        tokenizer_config={
            "trust_remote_code": args.trust_remote_code,
            "use_fast": False,
            # "eos_token": None,  # Remove the eos_token
        }
    )
    # After loading model and tokenizer
    debug_tokenizer_type(tokenizer)

    # Log tokenizer loading for debugging
    logging.debug(f"Tokenizer loaded with config: {tokenizer.__dict__}")

    if args.debug_prompt:
        debug_prompt(tokenizer, args.prompt, args.use_chat_template)

    if args.debug_all:
        print("\nModel Configuration:")
        print(f"Using chat template: {args.use_chat_template}")
        print(f"Special tokens map: {tokenizer.special_tokens_map}")
        
        # Show raw prompt
        print("\nRaw input prompt:", args.prompt)
        print_tokens(tokenizer, args.prompt, "Raw Input Tokens")
        
        # Show template
        template = args.prompt
        print("\nTemplate:")
        print(template)
        print_tokens(tokenizer, template, "Template Tokens")
        
        if args.use_chat_template:
            messages = [{"role": "user", "content": template}]
            debug_chat_format(tokenizer, messages)
            
    # Select the prompt based on the arguments
    if args.use_chat_template:
        messages = [{"role": "user", "content": args.prompt}]
        prompt = format_for_model(tokenizer, messages, use_chat_template=True, add_generation_prompt=False)
    else:
        prompt = tokenizer.encode(args.prompt)

    # Generate parameters
    generate_kwargs = {
        "max_tokens": args.max_tokens,
        "temp": args.temp,
        "top_p": args.top_p,
        "verbose": args.debug_all
    }
    
    print(f"\nTesting generation with prompt: {args.prompt}:")

    # Generate text
    if args.generate_type == "regular":
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            **generate_kwargs
        )
        if args.verbose:
            print_tokens(tokenizer, response, "Generated Response")
        else:
            print(response)

    elif args.generate_type == "stream":
        print("=" * 10)
        response_text = ""
        generated_tokens = []  # Keep track of generated tokens
        
        # Only include valid keyword arguments for stream_generate
        stream_generate_kwargs = {
            "max_tokens": args.max_tokens,
        }
        
        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            **stream_generate_kwargs
        ):
            generated_tokens.append(response.token) # Append each new token
            print(response.text, end="", flush=True)
            response_text += response.text  # append text to full response
            
            # Check if the last two tokens match the two-token EOS sequence
            if generated_tokens[-2:] == [127960, 127957]:
                # Remove the two EOS tokens from the response
                response_text = response_text[:-len(tokenizer.decode([127960, 127957]))]
                print_tokens(tokenizer, generated_tokens[:-2], "Generated Response (Without EOS)")
                break # Stop generation

            # Print tokens as they are generated in stream mode
            if args.debug_all:
                print_tokens(tokenizer, [response.token], f"Stream Step {response.generation_tokens}")
        if args.verbose:
            print()
            print_tokens(tokenizer, generated_tokens, "Generated Response") # Show all tokens

        print("=" * 10)
        if len(response_text) == 0:
            print("No text generated for this prompt")
        else:
            print(
                f"Prompt: {response.prompt_tokens} tokens, "
                f"{response.prompt_tps:.3f} tokens-per-sec"
            )
            print(
                f"Generation: {response.generation_tokens} tokens, "
                f"{response.generation_tps:.3f} tokens-per-sec"
            )
            print(f"Peak memory: {response.peak_memory:.3f} GB")

        print("-" * 80)
        print(f"\nFinal Response:\n{response_text}")

if __name__ == "__main__":
    main()