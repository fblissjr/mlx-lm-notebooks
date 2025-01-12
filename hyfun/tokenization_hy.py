# coding=utf-8
# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# Licensed under the Hunyuan License.

import os
import base64
import logging
import tiktoken
import unicodedata
from transformers import PreTrainedTokenizer
from typing import Collection, Dict, List, Set, Tuple, Union, Optional
from transformers.tokenization_utils_base import AddedToken

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "hy.tiktoken"}
PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|""" \
          r"""[^\r\n\p{L}\p{N}]?\p{L}+|""" \
          r"""\p{N}|""" \
          r""" ?[^\s\p{L}\p{N}]+[\r\n]*|""" \
          r"""\s*[\r\n]+|""" \
          r"""\s+(?!\S)|""" \
          r"""\s+"""

# Special tokens
ENDOFTEXT = "<|endoftext|>"
STARTOFTEXT = "<|startoftext|>"
BOSTOKEN = "<|bos|>"
EOSTOKEN = "<|eos|>"
PADTOKEN = "<|pad|>"

# Special token IDs (starting from base)
SPECIAL_START_ID = 127957

# Extra tokens for fine-tuning tasks
EXTRA_SPECIAL_TOKENS = tuple(f"<|extra_{i}|>" for i in range(204))


def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
    """Load tiktoken BPE vocabulary."""
    vocab = {}
    try:
        # Load vocab file
        with open(tiktoken_bpe_file, "rb") as f:
            for line in f:
                if line:
                    token, _ = line.split()
                    vocab[base64.b64decode(token)] = len(vocab)  # Assign ranks sequentially
                    
        if not vocab:
            raise ValueError("Empty vocabulary file")
            
        return vocab
        
    except Exception as e:
        raise ValueError(f"Error loading tiktoken vocabulary: {e}")


class HYTokenizer(PreTrainedTokenizer):
    """HunYuan tokenizer using tiktoken BPE."""
    
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        errors = "replace",
        bod_token = STARTOFTEXT,
        eod_token = ENDOFTEXT,
        bos_token = STARTOFTEXT, 
        eos_token = ENDOFTEXT,
        pad_token = PADTOKEN,
        add_bod_token = True,
        add_eod_token = True,
        **kwargs
    ):
        # First load the vocabulary
        self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)
        
        # Setup special tokens dictionary
        special_tokens = {
            STARTOFTEXT: SPECIAL_START_ID,
            ENDOFTEXT: SPECIAL_START_ID + 1,
            BOSTOKEN: SPECIAL_START_ID + 2,
            EOSTOKEN: SPECIAL_START_ID + 3,
            PADTOKEN: SPECIAL_START_ID + 4,
        }
        
        # Initialize internal state early
        self.special_tokens = special_tokens
        self.errors = errors
        self.vocab_file = vocab_file
        self.add_bod_token = add_bod_token
        self.add_eod_token = add_eod_token
        
        # Create encoder/decoder maps
        self._decoder = {v: k for k, v in self.mergeable_ranks.items()}
        self._decoder.update({v: k for k, v in special_tokens.items()})
        
        # Initialize the tokenizer before parent class
        self.tokenizer = tiktoken.Encoding(
            "Hunyuan",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=special_tokens
        )
        
        # Now initialize the parent class
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            errors=errors,
            **kwargs
        )
        
        # Set special token properties after parent init
        self.bod_token = bod_token  
        self.eod_token = eod_token
        self.bod_id = self.encode(bod_token, add_special_tokens=False)[0]
        self.eod_id = self.encode(eod_token, add_special_tokens=False)[0]

    @property
    def vocab_size(self) -> int:
        return len(self.get_vocab())

    def get_vocab(self) -> Dict[str, int]:
        vocab = {}

        # Add regular tokens        
        for token_bytes, idx in self.mergeable_ranks.items():
            try:
                token_str = token_bytes.decode('utf-8', errors=self.errors)
                vocab[token_str] = idx
            except:
                continue

        # Add special tokens
        for token, idx in self.special_tokens.items():
            vocab[token] = idx

        return vocab

    def _tokenize(self, text: str) -> List[str]:
        text = unicodedata.normalize('NFC', text)        
        tokens = []
        
        for token_id in self.tokenizer.encode(text, allowed_special="all"):
            value = self._decoder.get(token_id)
            if value is not None:
                if isinstance(value, bytes):
                    try:
                        tokens.append(value.decode('utf-8', errors=self.errors))
                    except:
                        continue
                else:
                    tokens.append(value)
                    
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token string to an integer id."""        
        # Handle special tokens
        if token in self.special_tokens:
            return self.special_tokens[token]
            
        # Handle regular tokens
        token_bytes = token.encode('utf-8')
        return self.mergeable_ranks.get(token_bytes, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        """Convert an integer id to a token string."""
        decoded = self._decoder.get(index)
        if decoded is None:
            return self.unk_token
            
        if isinstance(decoded, bytes):
            try:
                return decoded.decode('utf-8', errors=self.errors)
            except:
                return self.unk_token
        return decoded

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a sequence of tokens to a single string."""
        text = ""
        temp_bytes = b""
        
        for token in tokens:
            if token in self.special_tokens:
                if temp_bytes:
                    text += temp_bytes.decode('utf-8', errors=self.errors)
                    temp_bytes = b""
                text += token
            else:
                temp_bytes += token.encode('utf-8')
                
        if temp_bytes:
            text += temp_bytes.decode('utf-8', errors=self.errors)
            
        return text

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Build model inputs from a sequence by adding special tokens."""
        
        # Single sequence case
        if token_ids_1 is None:
            if self.add_bod_token:
                token_ids_0 = [self.bod_id] + token_ids_0
            if self.add_eod_token:
                token_ids_0 = token_ids_0 + [self.eod_id]
            return token_ids_0
            
        # Sequence pair case
        result = []
        if self.add_bod_token:
            result += [self.bod_id]
        result += token_ids_0
        if self.add_eod_token:
            result += [self.eod_id]
            
        if self.add_bod_token:
            result += [self.bod_id]
        result += token_ids_1
        if self.add_eod_token:
            result += [self.eod_id]
            
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False
    ) -> List[int]:
        """Get list where non-special tokens are 0s and special tokens are 1s."""
        
        if already_has_special_tokens:
            return [1 if token in self.all_special_ids else 0 for token in token_ids_0]

        # Single sequence case            
        if token_ids_1 is None:
            return [1 if token in self.all_special_ids else 0 for token in self.build_inputs_with_special_tokens(token_ids_0)]
            
        # Sequence pair case
        return [1 if token in self.all_special_ids else 0 for token in self.build_inputs_with_special_tokens(token_ids_0, token_ids_1)]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Save the vocabulary files and special tokens file."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            for token_bytes, rank in sorted(self.mergeable_ranks.items(), key=lambda x: x[1]):
                encoded = base64.b64encode(token_bytes).decode('utf-8')
                f.write(f"{encoded} {rank}\n")

        return (vocab_file,)