from collections import defaultdict
from copy import deepcopy
from typing import Iterator

from tqdm import tqdm
from .tokenizer_utils import pretokenize, initialize_pair_frequencies, apply_merge, PAT
import regex as re


class Tokenizer:
    def __init__(
        self, vocabulary: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]
    ) -> None:
        if special_tokens is None:
            special_tokens = []

        self.vocabulary = vocabulary

        self.inverse_vocabulary: dict[bytes, int] = {token: id for id, token in vocabulary.items()}

        for i in range(len(special_tokens)):
            encoded_special_token = special_tokens[i].encode("utf-8")
            if encoded_special_token in self.inverse_vocabulary:
                continue
            new_id = 1 + len(vocabulary)
            self.vocabulary[new_id] = encoded_special_token
            self.inverse_vocabulary[encoded_special_token] = new_id

        self.merges = merges
        self.special_tokens = special_tokens

    def _encode(self, pretoken_conversion: dict[tuple[bytes, ...], tuple[bytes, ...]], text: str) -> list[int]:
        tokens: list[int] = []
        segments = re.split("|".join(re.escape(token) for token in sorted(self.special_tokens,reverse=True)), text) if self.special_tokens else [text]
        current_index = 0
        for segment_index, segment in enumerate(segments):
            if segment_index > 0 and self.special_tokens:
                special_token_found = max([
                    special_token
                    for special_token in self.special_tokens
                    if text[current_index:].find(special_token) == 0
                ])
                special_token_end_index: int = current_index + len(special_token_found)
                tokens.append(self.inverse_vocabulary[special_token_found.encode("utf-8")])
                current_index += special_token_end_index - current_index

            if segment == "":
                continue

            current_index += len(segment)
            for match_ in re.finditer(PAT, segment):
                word = match_.group(0)
                encoded_word = tuple([c.to_bytes() for c in word.encode()])

                for sub_pretoken in pretoken_conversion[encoded_word]:
                    tokens.append(self.inverse_vocabulary[sub_pretoken])

        return tokens

    def encode(self, text: str) -> list[int]:
        pretoken_counts = pretokenize(text=text, special_tokens=self.special_tokens)
        pretokens_list = [(pretoken, count) for pretoken, count in pretoken_counts.items()]
        original_pretokens_list = deepcopy(pretokens_list)
        pairs_to_pretoken_indices: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
        pair_frequencies = initialize_pair_frequencies(pretokens_list, pairs_to_pretoken_indices)

        for merge in self.merges:
            apply_merge(
                pairs_to_pretoken_indices=pairs_to_pretoken_indices,
                candidate_pair=merge,
                pair_frequencies=pair_frequencies,
                pretokens=pretokens_list,
            )
        pretoken_conversion = {
            original_pretoken: bpe_applied_pretoken
            for (original_pretoken, _), (bpe_applied_pretoken, _) in zip(original_pretokens_list, pretokens_list)
        }

        return self._encode(pretoken_conversion, text)

    def encode_iterable(self, iterable: list[str]) -> Iterator[int]:
        accumulated_string = ""
        for item in iterable:
            accumulated_string += item
            if len(accumulated_string) > 10000:
                yield from self.encode(accumulated_string)
                accumulated_string = ""

        yield from self.encode(accumulated_string) 


    def decode(self, tokens: list[int]) -> str:
        text_list: list[bytes] = []

        for token in tokens:
            text_list.append(self.vocabulary[token])

        text_bytes: bytes = b""
        
        for byte in text_list:
            text_bytes += byte

        return text_bytes.decode("utf-8", errors="replace")
