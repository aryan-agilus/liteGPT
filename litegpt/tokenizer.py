"""
Tokenizer wrapper around tiktoken (GPT-2 BPE encoding).

Adds 5 special tokens for the chat format:
    <|bos|>             — beginning of sequence
    <|user_start|>      — user turn starts
    <|user_end|>        — user turn ends
    <|assistant_start|> — assistant turn starts
    <|assistant_end|>   — assistant turn ends

Vocab is padded to the next multiple of 64 for efficient matmul tiling.

Chat format (byte-stream):
    <|bos|>
    <|user_start|> {user text} <|user_end|>
    <|assistant_start|> {assistant text} <|assistant_end|>
    ...
"""

import tiktoken

_BASE_VOCAB = 50257   # GPT-2 base vocab size

SPECIAL_TOKENS: dict[str, int] = {
    "<|bos|>":             _BASE_VOCAB + 0,
    "<|user_start|>":      _BASE_VOCAB + 1,
    "<|user_end|>":        _BASE_VOCAB + 2,
    "<|assistant_start|>": _BASE_VOCAB + 3,
    "<|assistant_end|>":   _BASE_VOCAB + 4,
}

# Pad to next multiple of 64 for efficient GPU matmul tiling
_EXTENDED = _BASE_VOCAB + len(SPECIAL_TOKENS)
VOCAB_SIZE = ((_EXTENDED + 63) // 64) * 64   # 50304


class Tokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")

        self.bos_id   = SPECIAL_TOKENS["<|bos|>"]
        self.usr_s    = SPECIAL_TOKENS["<|user_start|>"]
        self.usr_e    = SPECIAL_TOKENS["<|user_end|>"]
        self.asst_s   = SPECIAL_TOKENS["<|assistant_start|>"]
        self.asst_e   = SPECIAL_TOKENS["<|assistant_end|>"]

        self._special_ids = set(SPECIAL_TOKENS.values())

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    # ------------------------------------------------------------------
    # Core encode / decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        # disallowed_special=() treats any special-token strings (e.g. <|endoftext|>)
        # found in the raw text as normal text rather than raising an error.
        return self.enc.encode(text, disallowed_special=())

    def decode(self, ids: list[int]) -> str:
        # Strip special tokens before decoding
        clean = [i for i in ids if i not in self._special_ids and i < _BASE_VOCAB]
        return self.enc.decode(clean)

    # ------------------------------------------------------------------
    # Chat formatting
    # ------------------------------------------------------------------

    def encode_chat(self, messages: list[dict[str, str]]) -> list[int]:
        """
        Encode a conversation to a flat token sequence.

        Args:
            messages: list of {"role": "user"|"assistant", "content": str}

        Returns:
            token ids starting with <|bos|>
        """
        tokens = [self.bos_id]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                tokens += [self.usr_s] + self.encode(content) + [self.usr_e]
            elif role == "assistant":
                tokens += [self.asst_s] + self.encode(content) + [self.asst_e]
            else:
                raise ValueError(f"Unknown role: {role!r}")
        return tokens

    def decode_response(self, ids: list[int]) -> str:
        """
        Decode assistant tokens, stopping at <|assistant_end|>.
        Use this when sampling a response.
        """
        if self.asst_e in ids:
            ids = ids[: ids.index(self.asst_e)]
        return self.decode(ids)
