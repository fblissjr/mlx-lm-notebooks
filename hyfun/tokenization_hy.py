import json
import os
import base64
import logging
import tiktoken
import unicodedata
from transformers import PreTrainedTokenizer, AddedToken
from typing import Collection, Dict, List, Set, Tuple, Union, Optional

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "hy.tiktoken", "vocab_file_json": "vocab.json"}  # Add vocab.json
PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|""" \
          r"""[^\r\n\p{L}\p{N}]?\p{L}+|""" \
          r"""\p{N}|""" \
          r""" ?[^\s\p{L}\p{N}]+[\r\n]*|""" \
          r"""\s*[\r\n]+|""" \
          r"""\s+(?!\S)|""" \
          r"""\s+"""
# default eod_token and bod_token of our base model
ENDOFTEXT = "<|endoftext|>"
STARTOFTEXT = "<|startoftext|>"

# extra flag token for other training
BOSTOKEN = "<|bos|>"
EOSTOKEN = "<|eos|>"

PADTOKEN = "<|pad|>"

# extra special tokens for the tokenizer
# as the default behavior is changed to allow special tokens in
# regular texts, the surface forms of special tokens need to be
# as different as possible to minimize the impact
EXTRAS = tuple((f"<|extra_{i}|>" for i in range(204)))

SPECIAL_START_ID = 127957

def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
    dic = {}
    rank = 0
    for i, line in enumerate(open(tiktoken_bpe_file, "rb")):
        if line:
            token, _ = line.split()
            # skip duplicated tokens, this should not happen
            if base64.b64decode(token) in dic:
                raise ValueError(f"!ERROR: duplicated token {token} in your vocab file")
            dic[base64.b64decode(token)] = int(rank)
            rank += 1
    return dic

# special tokens for pretrain and finetune models
SPECIAL_TOKENS = tuple(
    enumerate(
        (
            (
                ENDOFTEXT,
                STARTOFTEXT,
                BOSTOKEN,
                EOSTOKEN,
                PADTOKEN,
            )
            + EXTRAS
        ),
        start=SPECIAL_START_ID,
    )
)

SPECIAL_TOKENS_SET = set(t for i, t in SPECIAL_TOKENS)

class HYTokenizer(PreTrainedTokenizer):
    """
    HunYuan Tokenizer Initialization. We extend `tiktoken` vocab and
        the default EOD & BOD special tokens are used for base model.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.

        errors (`str`):
            How to handle errors in decoding UTF-8 byte sequences.
            use ignore if you are in streaming inference

        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `""<|startoftext|>""`):
            The beginning of document token that was used for training. can be modified by your task.
            default to be `<|startoftext|>` for released base model.

        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `""<|endoftext|>""`):
            The end of document token that was used for training. can be modified by your task.
            default to be `<|endoftext|>` for released base model.

        pad_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.

        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add an `bos_token` at the start of documents.
            This will effect `build_inputs_with_special_tokens` method

        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of documents.
            This will effect `build_inputs_with_special_tokens` method

    """    
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        errors="replace",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        add_bos_token=True,
        add_eos_token=False,
        **kwargs,
    ):
        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            **kwargs,
        )
        self.errors = errors
        self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)
        self.special_tokens = {token: index for index, token in SPECIAL_TOKENS}

        enc = tiktoken.Encoding(
            "HunYuan",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        assert (
            len(self.mergeable_ranks) + len(self.special_tokens) == enc.n_vocab
        ), f"{len(self.mergeable_ranks)} + {len(self.special_tokens)} != {enc.n_vocab} in encoding"

        self.decoder = {v: k for k, v in self.mergeable_ranks.items()}
        self.decoder.update({v: k for k, v in self.special_tokens.items()})
        self.tokenizer = enc

        self.eod_token, self.eod_id = self.eos_token, self.eos_token_id

        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["tokenizer"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        enc = tiktoken.Encoding(
            "HunYuan",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.tokenizer = enc

    def __len__(self) -> int:
        return self.tokenizer.n_vocab

    def get_vocab(self) -> Dict[bytes, int]:
        return self.mergeable_ranks

    def convert_tokens_to_ids(self, tokens: Union[bytes, str, List[Union[bytes, str]]]) -> List[int]:
        ids = []
        if isinstance(tokens, (str, bytes)):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.mergeable_ranks.get(tokens)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.mergeable_ranks.get(token))
        return ids

    def _add_tokens(
        self,
        new_tokens: Union[List[str], List[AddedToken]],
        special_tokens: bool = False,
    ) -> List[Tuple[int, str]]:
        """do not support adding tokens"""
        if not special_tokens and new_tokens:
            raise ValueError("Adding regular tokens is not supported")
        for token in new_tokens:
            surface_form = token.content if isinstance(token, AddedToken) else token
            if surface_form not in SPECIAL_TOKENS_SET:
                raise ValueError("Adding unknown special tokens is not supported")
        return 0

    def save_vocabulary(self, save_directory: str, **kwargs) -> Tuple[str]:
        """
        Save the vocabulary to a tiktoken file and a JSON file.
        """
        # Save tiktoken file
        file_path = os.path.join(save_directory, "hy.tiktoken")
        with open(file_path, "w", encoding="utf8") as w:
            for k, v in self.mergeable_ranks.items():
                line = base64.b64encode(k).decode("utf8") + " " + str(v) + "\n"
                w.write(line)

        # Save JSON file
        json_file_path = os.path.join(save_directory, "vocab.json")
        with open(json_file_path, "w", encoding="utf8") as json_w:
            vocab_dict = {}

            # Add regular tokens (decoded from bytes)
            for token_bytes, token_id in self.mergeable_ranks.items():
                vocab_dict[token_id] = token_bytes.decode("utf-8", errors="ignore")

            # Add special tokens (already strings)
            for token, token_id in self.special_tokens.items():
              vocab_dict[token_id] = token

            # Sort by token ID for easier readability
            vocab_dict = dict(sorted(vocab_dict.items()))

            json.dump(vocab_dict, json_w, ensure_ascii=False, indent=4)

        return file_path, json_file_path

    def tokenize(self, text, allowed_special="all", disallowed_special=(), **kwargs):
        logger.debug(f"tokenize called with text: {repr(text)}")
        logger.debug(f"allowed_special: {allowed_special}, disallowed_special: {disallowed_special}")
        tokens = []
        text = unicodedata.normalize("NFC", text)

        # Log the direct tiktoken encoding
        raw_tokens = self.tokenizer.encode(
            text, allowed_special=allowed_special, disallowed_special=disallowed_special
        )
        logger.debug(f"tiktoken raw tokens: {raw_tokens}")

        for t in raw_tokens:
            tokens.append(self.decoder[t])
        logger.debug(f"tokenize output: {tokens}")
        return tokens

    def convert_tokens_to_string(self, tokens: List[Union[bytes, str]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors=self.errors)
                    temp = b""
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type types or str")
        if temp:
            text += temp.decode("utf-8", errors=self.errors)
        return text

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    def _convert_id_to_token(self, index: int) -> Union[bytes, str]:
        if index in self.decoder:
            return self.decoder[index]
        raise ValueError("unknown ids")

    def _convert_token_to_id(self, token: Union[bytes, str]) -> int:
        if token in self.special_tokens:
            return self.special_tokens[token]
        if token in self.mergeable_ranks:
            return self.mergeable_ranks[token]
        raise ValueError("unknown token")

    def _tokenize(self, text: str, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).
        Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def _decode(self, token_ids, skip_special_tokens=False, errors=None, **kwargs):
        logger.debug(f"_decode called with tokens: {token_ids}")
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            token_ids = [i for i in token_ids if i < self.eod_id]
        result = self.tokenizer.decode(token_ids, errors=errors or self.errors)
        logger.debug(f"_decode result: {result}")
        return result

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for question answering tasks by concatenating and
        adding special tokens. A sequence has the following format:
        - single sequence: `bos_token_id` + `X`
        - pair of sequences: `bos_token_id` + `A` + `eos_token_id` + `bos_token_id` + `B` + `eos_token_id`
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        bos = [self.bos_token_id] if self.add_bos_token else []
        eos = [self.eos_token_id] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos + token_ids_0 + eos
        return bos + token_ids_0 + eos + bos + token_ids_1 + eos