"""Implement a BPE based subword tokenizer."""

from collections import defaultdict
import pickle
import sys
import time

import tqdm
from .tokenizer_utils import pretokenize_corpus_parallel, TVocabulary, TBPEMerge, initialize_pair_frequencies, apply_merge

def update_pair_frequency(
    pair_frequencies: dict[tuple[bytes, bytes], int],
    pairs_to_pretoken_indices: dict[tuple[bytes, bytes], set[int]],
    pretoken: tuple[bytes, ...],
    count: int,
    index: int,
):
    for i in range(len(pretoken) - 1):
        pair_frequencies[(pretoken[i], pretoken[i + 1])] = (
            pair_frequencies.get((pretoken[i], pretoken[i + 1]), 0) + count
        )
        pairs_to_pretoken_indices[(pretoken[i], pretoken[i + 1])].add(index)




def initialize_vocabulary(special_tokens: list[str]) -> TVocabulary:
    vocab = {i: i.to_bytes() for i in range(256)}
    for i in range(len(special_tokens)):
        vocab[i + 256] = special_tokens[i].encode("utf-8")

    return vocab


def get_merge_candidate(pair_frequencies: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes] | None:
    max_pair_frequency = max(pair_frequencies.values())
    if max_pair_frequency == 0:
        return None
    # return the pair with max pair frequency that is lexicographically the greatest
    return sorted([pair for pair, frequency in pair_frequencies.items() if frequency == max_pair_frequency])[-1]


def run_bpe(input_path: str, vocabulary_size: int, special_tokens: list[str]) -> tuple[TVocabulary, TBPEMerge]:
    pair_frequencies: dict[tuple[bytes, bytes], int] = {}
    vocabulary = initialize_vocabulary(special_tokens)
    with open(input_path, "rb") as f:
        pretokens = pretokenize_corpus_parallel(f, num_processes=7, special_tokens=special_tokens)

    merges: list[tuple[bytes, bytes]] = []
    pairs_to_pretoken_indices: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    pretokens_list = [(pretoken, count) for pretoken, count in pretokens.items()]
    pair_frequencies = initialize_pair_frequencies(pretokens_list=pretokens_list, pairs_to_pretoken_indices=pairs_to_pretoken_indices)
    progress = tqdm.tqdm(total=vocabulary_size - len(vocabulary), desc="BPE merges")
    while len(vocabulary) < vocabulary_size:
        progress.update(1)
        candidate = get_merge_candidate(pair_frequencies)
        if candidate is None:
            break
        merges.append(candidate)
        max_token_id = max(list(vocabulary.keys()))

        new_token_id = max_token_id + 1
        vocabulary[new_token_id] = candidate[0] + candidate[1]

        apply_merge(pretokens_list, pairs_to_pretoken_indices, pair_frequencies, candidate)

    return vocabulary, merges

if __name__ == "__main__":

    print("starting")
    start_time = time.perf_counter()
    vocabulary, merges = run_bpe(
        input_path=sys.argv[1], vocabulary_size=int(sys.argv[2]), special_tokens=["<|endoftext|>"]
    )
    print(f"Took {time.perf_counter() - start_time}s.")
    with open(sys.argv[1] + "vocabulary.pkl", "wb") as f:
        pickle.dump(vocabulary, f)

    with open(sys.argv[1] + "merges.pkl", "wb") as f:
        pickle.dump(merges, f)

