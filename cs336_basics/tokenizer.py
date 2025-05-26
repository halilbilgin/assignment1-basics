"""Implement a BPE based subword tokenizer."""

import cProfile
from collections import defaultdict
import concurrent.futures

from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
import io
import json
import os
import pickle
from pstats import SortKey
import pstats
import sys
import time
import regex as re
from typing import BinaryIO

import tqdm

TPretokenCounts = dict[tuple[bytes, ...], int]
TPretokenCountsList = list[tuple[tuple[bytes, ...], int]]
TVocabulary = dict[int, bytes]
TBPEMerge = list[tuple[bytes, bytes]]

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize(text: str, special_tokens: list[str]) -> TPretokenCounts:
    segments = re.split("|".join(re.escape(token) for token in special_tokens), text)
    result: TPretokenCounts = {}
    for segment in segments:
        for match_ in re.finditer(PAT, segment):
            word = match_.group(0)
            encoded_word = tuple([c.to_bytes() for c in word.encode()])

            result[encoded_word] = result.get(encoded_word, 0) + 1

    return result


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_corpus_parallel(f: BinaryIO, num_processes: int, special_tokens: list[str]):
    result: TPretokenCounts = {}
    with ProcessPoolExecutor(max_workers=num_processes) as pool:
        futures: list[Future[TPretokenCounts]] = []
        boundaries = find_chunk_boundaries(f, num_processes * 1000, b"<|endoftext|>")
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for i, (start, end) in tqdm.tqdm(enumerate(zip(boundaries[:-1], boundaries[1:])), total=len(boundaries) - 1):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="strict")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            futures.append(pool.submit(pretokenize, text=chunk, special_tokens=special_tokens))

            if i % num_processes * 3 > 0:
                continue
            for future in concurrent.futures.as_completed(futures):
                for pretoken, count in future.result().items():
                    result[pretoken] = result.get(pretoken, 0) + count
            futures.clear()

    for future in concurrent.futures.as_completed(futures):
        for pretoken, count in future.result().items():
            result[pretoken] = result.get(pretoken, 0) + count

    print("pretokenization is over. number of pretokens:", len(result))
    return result


def update_pair_frequency(pair_frequencies: dict[tuple[bytes, bytes], int], pairs_to_pretoken_indices: dict[tuple[bytes, bytes], set[int]], pretoken: tuple[bytes, ...], count: int, index: int):
    for i in range(len(pretoken) - 1):
        pair_frequencies[(pretoken[i], pretoken[i + 1])] = (
            pair_frequencies.get((pretoken[i], pretoken[i + 1]), 0) + count
        )
        pairs_to_pretoken_indices[(pretoken[i], pretoken[i + 1])].add(index)


def update_pair_frequencies_with_merge(
    pretoken: tuple[bytes, ...], count: int, pair: tuple[bytes, bytes], pair_frequencies: dict[tuple[bytes, bytes], int]
):
    new_pretoken: list[bytes] = []
    i = 0
    # candidate: t h
    # pretoken k,a,t,h,t,h,e,r,i,n,a,t,h,a
    # new_pretoken k,a,th,th,e,r,i,n,a,th,a

    while i < len(pretoken):
        if i < len(pretoken) - 1 and (pretoken[i], pretoken[i + 1]) == pair:
            merge = pair[0] + pair[1]
            pair_frequencies[pair] -= count
            if len(new_pretoken):
                pair_frequencies[(new_pretoken[-1], merge)] = pair_frequencies.get((new_pretoken[-1], merge), 0) + count
                pair_frequencies[(new_pretoken[-1], pair[0])] = (
                    pair_frequencies.get((new_pretoken[-1], pair[0]), 0) - count
                )
            if i + 2 < len(pretoken):
                pair_frequencies[(merge, pretoken[i + 2])] = pair_frequencies.get((merge, pretoken[i + 2]), 0) + count
                pair_frequencies[(pair[1], pretoken[i + 2])] = (
                    pair_frequencies.get((pair[1], pretoken[i + 2]), 0) - count
                )

            new_pretoken.append(merge)
            i += 2
        else:
            new_pretoken.append(pretoken[i])
            i += 1

    return tuple(new_pretoken)


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
    for index, (pretoken, count) in enumerate(pretokens_list):
        update_pair_frequency(pair_frequencies, pairs_to_pretoken_indices, pretoken, count, index)

    progress = tqdm.tqdm(total=vocabulary_size - len(vocabulary), desc="BPE merges")
    while len(vocabulary) < vocabulary_size:
        progress.update(1)
        candidate = get_merge_candidate(pair_frequencies)
        if candidate is None:
            break
        merges.append(candidate)
        apply_merge(vocabulary, pretokens_list, pairs_to_pretoken_indices, pair_frequencies, candidate)

    return vocabulary, merges


def apply_merge(
    vocabulary: TVocabulary,
    pretokens: TPretokenCountsList,
    pairs_to_pretoken_indices: dict[tuple[bytes, bytes], set[int]],
    pair_frequencies: dict[tuple[bytes, bytes], int],
    candidate_pair: tuple[bytes, bytes],
) -> None:
    max_token_id = max(list(vocabulary.keys()))

    new_token_id = max_token_id + 1
    vocabulary[new_token_id] = candidate_pair[0] + candidate_pair[1]

    for index in list(pairs_to_pretoken_indices.get(candidate_pair, [])):
        pretoken, count = pretokens[index]
        for i in range(len(pretoken) - 1):
            if (pretoken[i], pretoken[i + 1]) == candidate_pair:
                new_pretoken = update_pair_frequencies_with_merge(pretoken, count, candidate_pair, pair_frequencies)
                for pair in zip(pretoken[:-1], pretoken[1:]):
                    pairs_to_pretoken_indices[pair].discard(index)
                for pair in zip(new_pretoken[:-1], new_pretoken[1:]):
                    pairs_to_pretoken_indices[pair].add(index)

                pretokens[index] = (new_pretoken, count)
                break


if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()

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
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

