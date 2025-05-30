import regex as re

import concurrent.futures

from concurrent.futures import Future, ProcessPoolExecutor
import os
from typing import BinaryIO

import tqdm

TPretokenCounts = dict[tuple[bytes, ...], int]
TPretokenCountsList = list[tuple[tuple[bytes, ...], int]]
TVocabulary = dict[int, bytes]
TBPEMerge = list[tuple[bytes, bytes]]

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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


def initialize_pair_frequencies(pretokens_list: TPretokenCountsList, pairs_to_pretoken_indices: dict[tuple[bytes, bytes], set[int]]) -> dict[tuple[bytes, bytes], int]:
    pair_frequencies: dict[tuple[bytes, bytes], int] = {}
    for index, (pretoken, count) in enumerate(pretokens_list):
        update_pair_frequency(pair_frequencies, pairs_to_pretoken_indices, pretoken, count, index)

    return pair_frequencies

def pretokenize(text: str, special_tokens: list[str]) -> TPretokenCounts:
    segments = re.split("|".join(re.escape(token) for token in sorted(special_tokens, reverse=True)), text) if special_tokens else [text]
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

            if i % num_processes * 4 > 0:
                continue
            for future in concurrent.futures.as_completed(futures[: -num_processes * 2]):
                for pretoken, count in future.result().items():
                    result[pretoken] = result.get(pretoken, 0) + count
            futures = futures[-num_processes * 2 :]

    for future in concurrent.futures.as_completed(futures):
        for pretoken, count in future.result().items():
            result[pretoken] = result.get(pretoken, 0) + count

    print("pretokenization is over. number of pretokens:", len(result))
    return result



def apply_merge(
    pretokens: TPretokenCountsList,
    pairs_to_pretoken_indices: dict[tuple[bytes, bytes], set[int]],
    pair_frequencies: dict[tuple[bytes, bytes], int],
    candidate_pair: tuple[bytes, bytes],
) -> None:
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