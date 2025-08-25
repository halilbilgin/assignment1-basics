import pytest
from scipy import stats
import torch
import numpy as np

from unittest.mock import MagicMock
from cs336_basics.inference import LLMInference

def test_encode_no_tricks():
    sample_count = 100
    mock_tokenizer, mock_llm = MagicMock(), MagicMock(return_value=torch.from_numpy(np.asarray([[[1, 2, 3, 4]]])))
    inference = LLMInference(mock_tokenizer, mock_llm, max_token_generated=sample_count, temperature=1, top_p=1)
    expected_probs = np.array([0.45, 0.2, 0.3, 0.05])
    mock_llm.softmax.return_value = torch.from_numpy(np.asarray([[expected_probs]]))
    mock_tokenizer.decode.side_effect = lambda x: str(x[0])
    observed = np.bincount([int(v) for v in inference.encode(text="hehe")])
    chi_squared_stat = (((observed - expected_probs*sample_count)**2)/expected_probs*sample_count).sum()

    assert stats.chi2.cdf(chi_squared_stat, df=3) > 0.1
