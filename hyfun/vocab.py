import os
import base64
import logging
import tiktoken
import unicodedata
import json
from transformers import PreTrainedTokenizer, AddedToken
from typing import Collection, Dict, List, Set, Tuple, Union

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "hy.tiktoken"}
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

        bod_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `""<|startoftext|>""`):
            The beginning of document token that was used for training. can be modified by your task.
            default to be `<|startoftext|>` for released base model.

        eod_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `""<|endoftext|>""`):
            The end of document token that was used for training. can be modified by your task.
            default to be `<|endoftext|>` for released base model.

        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `None`):
            The start or sep special token that was used for some training tasks.
            default to be `<|startoftext|>` for released base model.
            It can be set to `<|bos|>` when you training for some other task

        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `None`):
            The end or sep special token that was used for some training tasks.
            default to be `<|endoftext|>` for released base model.
            It can be set to `<|eos|>` when you training for some other task

        pad_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.

        special_vocab_file (str, *optional*):
            Customed special extra vocab file, same format with hy.tiktoken.
            **Be careful** to use the extra special vocab, it will may cause the model loading collapse.
            The data line be like:
                `PHxhYmN8Pg== 0`
            the id followed `base64.encode(str)` is unused, we will reset them in case of collision

        add_bod_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add an `bos_token` at the start of documents.
            This will effect `build_inputs_with_special_tokens` method

        add_eod_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of documents.
            This will effect `build_inputs_with_special_tokens` method

    """
    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        errors="replace",
        bod_token="<|startoftext|>",
        eod_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        add_bod_token=True,
        add_eod_token=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.errors = errors
        self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)  # type: Dict[bytes, int]
        self.special_tokens = {
            token: index
            for index, token in SPECIAL_TOKENS
        }

        enc = tiktoken.Encoding(
            "HunYuan",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        assert (
            len(self.mergeable_ranks) + len(self.special_tokens) == enc.n_vocab
        ), f"{len(self.mergeable_ranks)} + {len(self.special_tokens)} != {enc.n_vocab} in encoding"

        self.decoder = {
            v: k for k, v in self.mergeable_ranks.items()
        }  # type: dict[int, bytes|str]
        self.decoder.update({v: k for k, v in self.special_tokens.items()})

        self.tokenizer = enc

        self.bod_token, self.bod_id = bod_token, self.special_tokens[bod_token]
        self.eod_token, self.eod_id = eod_token, self.special_tokens[eod_token]
        self.bos_token, self.bos_id = bos_token, self.special_tokens[bos_token]
        self.eos_token, self.eos_id = eos_token, self.special_tokens[eos_token]
        self.pad_token, self.pad_id = pad_token, self.special_tokens[pad_token]

        self._num_special_token = len(self.special_tokens)

        self.add_bod_token = add_bod_token
        self.add_eod_token = add_eod_token

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
        """Return the vocabulary as a dictionary, without special tokens."""
        return self.mergeable_ranks

    def convert_tokens_to_ids(
        self, tokens: Union[bytes, str, List[Union[bytes, str]]]
    ) -> List[int]:
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

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bod_token_id = [self.bod_id] if self.add_bod_token else []
        eod_token_id = [self.eod_id] if self.add_eod_token else []
        output = bod_token_id + token_ids_0 + eod_token_id
        if token_ids_1 is not None:
            output = output + bod_token_id + token_ids_1 + eod_token_id
        return output

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
    
    def save_vocabulary(self, save_directory: str, filename_prefix: str = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # Save the tiktoken vocabulary
        vocab_file_path = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + "hy.tiktoken"
        )
        with open(vocab_file_path, "w", encoding="utf8") as w:
            for k, v in self.mergeable_ranks.items():
                line = base64.b64encode(k).decode("utf8") + " " + str(v) + "\n"
                w.write(line)

        # Save the vocabulary in JSON format
        vocab_json_path = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        vocab_dict = {base64.b64encode(k).decode("utf-8"): v for k, v in self.mergeable_ranks.items()}
        with open(vocab_json_path, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=4)

        # Correctly save special tokens to special_tokens_map.json
        special_tokens_dict = {}
        for token, index in self.special_tokens.items():
            special_tokens_dict[token] = index

        with open(special_tokens_path, "w", encoding="utf-8") as f:
            json.dump(special_tokens_dict, f, ensure_ascii=False, indent=4)

        return (vocab_file_path, vocab_json_path, special_tokens_path)

    def tokenize(
        self,
        text: str,
        allowed_special: Union[Set, str] = "all",
        disallowed_special: Union[Collection, str] = (),
        **kwargs,
    ) -> List[Union[bytes, str]]:
        """
        Converts a string in a sequence of tokens.
        Args:
            text (`str`):
                The sequence to be encoded.
            allowed_special (`Literal["all"]` or `set`):
                The surface forms of the tokens to be encoded as special tokens in regular texts.
                Default to "all".
            disallowed_special (`Literal["all"]` or `Collection`):
                The surface forms of the tokens that should not be in regular texts and trigger errors.
                Default to an empty tuple.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific encode method.
        Returns:
            `List[bytes|str]`: The list of tokens.
        """
        tokens = []
        text = unicodedata.normalize("NFC", text)

        # this implementation takes a detour: text -> token id -> token surface forms
        for t in self.tokenizer.encode(
            text, allowed_special=allowed_special, disallowed_special=disallowed_special
        ):
            tokens.append(self.decoder[t])
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
        """Converts an id to a token, special tokens included"""
        if index in self.decoder:
            return self.decoder[index]
        raise ValueError("unknown ids")

    def _convert_token_to_id(self, token: Union[bytes, str]) -> int:
        """Converts a token to an id using the vocab, special tokens included"""
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

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        errors: str = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            token_ids = [i for i in token_ids if i < self.eod_id]
        return self.tokenizer.decode(token_ids, errors=errors or self.errors)
    
tokenizer = HYTokenizer(vocab_file="hy.tiktoken")
tokenizer.save_pretrained("saved_tokenizer")