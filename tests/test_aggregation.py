import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ptfidf.train import aggregation


def test_token_statistics():
    """check aggregation on small example."""
    X = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [1, 0],
    ])

    labels = pd.Series([0, 1, 1, 2])

    expected = {
        0: {(1, 1, 2), (2, 1, 1)},
        1: {(1, 0, 2), (2, 2, 1)},}

    res = aggregation.get_token_statistics(csr_matrix(X), labels)
    actual = res.groupby('token').apply(lambda df: set(df[['n', 'k', 'weight']].apply(tuple, axis=1))).to_dict()

    assert set(expected.keys()) == set(actual.keys()), 'token set does not match'
    for k in expected.keys():
        assert actual[k] == expected[k], 'statistics do not match'
