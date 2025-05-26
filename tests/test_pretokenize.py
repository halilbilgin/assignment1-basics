import tempfile
from unittest.mock import patch
from cs336_basics.tokenizer import pretokenize_corpus_parallel


def _convert_to_str(s: tuple[bytes, ...]) -> str:
    return "".join([c.decode() for c in s])


def test_pretokenize_corpus_parallel():
    _, filename = tempfile.mkstemp()

    with open(filename, "w") as f:
        f.write(
            """low low low low low lower lower widest widest<|endoftext|>widest newest newest newest<|endoftext|>newest newest newest"""
        )
    with open(filename, "rb") as f:
        assert {
            _convert_to_str(x): v
            for x, v in pretokenize_corpus_parallel(f=f, num_processes=3, special_tokens=["<|endoftext|>"]).items()
        } == {"newest": 1, " newest": 5, "widest": 1, "low": 1, " low": 4, " lower": 2, " widest": 2}
