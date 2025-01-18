import argparse
from mlx_lm import load, generate
import mlx.core as mx

DEFAULT_MAX_TOKENS = 512
DEFAULT_SEED = 0
DEFAULT_MODEL = "/Users/fredbliss/Storage/hyvideo-mlx-2bit"


def generate_rewritten_prompt(
    model,
    tokenizer,
    ori_prompt,
    mode="Normal",
    max_tokens=DEFAULT_MAX_TOKENS,
    use_default_chat_template=False,
    system_prompt=None,
):
    # Define templates as before...
    normal_mode_prompt = """Normal mode - Video Recaption Task:

You are a large language model specialized in rewriting video descriptions. Your task is to modify the input description.

0. Preserve ALL information, including style words and technical terms.

1. If the input is in Chinese, translate the entire description to English. 

2. If the input is just one or two words describing an object or person, provide a brief, simple description focusing on basic visual characteristics. Limit the description to 1-2 short sentences.

3. If the input does not include style, lighting, atmosphere, you can make reasonable associations.

4. Output ALL must be in English.

Given Input:
input: "{input}"
"""

    master_mode_prompt = """Master mode - Video Recaption Task:

You are a large language model specialized in rewriting video descriptions. Your task is to modify the input description.

0. Preserve ALL information, including style words and technical terms.

1. If the input is in Chinese, translate the entire description to English. 

2. To generate high-quality visual scenes with aesthetic appeal, it is necessary to carefully depict each visual element to create a unique aesthetic.

3. If the input does not include style, lighting, atmosphere, you can make reasonable associations.

4. Output ALL must be in English.

Given Input:
input: "{input}"
"""

    encode_video_system_prompt = (
        "Describe the video by detailing the following aspects: "
        "1. The main content and theme of the video."
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
        "4. background environment, light, style and atmosphere."
        "5. camera angles, movements, and transitions used in the video."
    )

    if mode == "Normal":
        prompt = normal_mode_prompt.format(input=ori_prompt)
    elif mode == "Master":
        prompt = master_mode_prompt.format(input=ori_prompt)
    else:
        raise Exception("Only supports Normal and Master", mode)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Use default chat template if requested
    if use_default_chat_template:
        if tokenizer.chat_template is not None:
            tokenizer.chat_template = tokenizer.chat_template
        else:
            raise ValueError(
                "Requested default chat template, but tokenizer does not have a default chat template."
            )

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, add_special_tokens=True
    )

    # Generate the response
    generate(model, tokenizer, prompt, max_tokens=max_tokens, verbose=True)

    return ""


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Rewrite video descriptions using a large language model."
    )

    # Model and path arguments
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )

    # Prompt and mode arguments
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The original video description prompt. Use '-' to read from stdin.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["Normal", "Master"],
        default="Normal",
        help="The rewriting mode (Normal or Master).",
    )

    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="The PRNG seed")

    # Tokenizer options
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        help="System prompt to use",
    )
    # Debugging
    parser.add_argument(
        "--debug-tokens",
        action="store_true",
        help="Print the raw tokenized prompt, including special tokens and system messages.",
    )

    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    mx.random.seed(args.seed)

    tokenizer_config = {"trust_remote_code": args.trust_remote_code}

    model, tokenizer = load(
        args.model, adapter_path=args.adapter_path, tokenizer_config=tokenizer_config
    )

    # Debug tokens
    if args.debug_tokens:
        messages = [{"role": "user", "content": args.prompt}]
        if args.system_prompt is not None:
            messages.insert(0, {"role": "system", "content": args.system_prompt})
        if tokenizer.chat_template is not None:
            # tokenize each message individually to show special tokens within each message
            for message in messages:
                tokenized_message = tokenizer.encode(
                    tokenizer.apply_chat_template(
                        [message], tokenize=False, add_generation_prompt=True
                    ),
                    add_special_tokens=True,
                )
                print(f"  {message['role']}:")
                for token_id in tokenized_message:
                    token_text = tokenizer.decode([token_id])
                    print(f"    {token_id} -> '{token_text}'")

            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            tokenized_prompt = tokenizer.encode(prompt, add_special_tokens=False)
            print(f"Full prompt:")
            for token_id in tokenized_prompt:
                token_text = tokenizer.decode([token_id])
                print(f"    {token_id} -> '{token_text}'")
        else:
            tokenized_prompt = tokenizer.encode(args.prompt, add_special_tokens=True)
            print(f"  user:")
            for token_id in tokenized_prompt:
                token_text = tokenizer.decode([token_id])
                print(f"    {token_id} -> '{token_text}'")

            print(f"Full prompt:")
            for token_id in tokenized_prompt:
                token_text = tokenizer.decode([token_id])
                print(f"    {token_id} -> '{token_text}'")

    # Generate the rewritten prompt
    generate_rewritten_prompt(
        model,
        tokenizer,
        args.prompt,
        mode=args.mode,
        max_tokens=args.max_tokens,
        use_default_chat_template=args.use_default_chat_template,
        system_prompt=args.system_prompt,
    )


if __name__ == "__main__":
    main()
