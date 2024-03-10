from collections import Counter
import math
from pathlib import Path

import pytest
import nyt_expt_24.experiment as expt


def test_iter_month_params():
    actual = list(expt._iter_month_params(2020, 6, 30))
    assert actual[0] == (2020, 6)
    assert actual[6] == (2020, 12)
    assert actual[7] == (2021, 1)


@pytest.fixture
def data_root() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture
def returns(data_root):
    expt.load_yfinance_spy(data_root)["Close"]


def test_ret_scores(data_root):
    idx = expt.load_yfinance_spy(data_root)["Close"]
    rets = idx.pct_change()
    ret_scores = expt.zsbucket(rets, -0.3, 0.3)
    # assert ret_scores['2019-01-04'] == 1


def test_dataset(data_root):
    ds = expt.make_nytimes_dataset(
        start_year=2020, start_month=2, n_months=2, k1=-1, k2=1, data_root=data_root
    )
    elems = list(iter(ds))
    labels = [elem["labels"].item() for elem in elems]
    labels = ["nan" if math.isnan(label) else label for label in labels]
    counts = dict(Counter(labels))
    assert counts[0] > 10
    assert counts[1] > 10
    assert counts[2] > 10


def test_make_nyt_test_data(data_root):
    real_data_root = "/Users/dcuddihy/data"
    expt.make_nyt_test_data(
        source_data_root=real_data_root,
        target_data_root=data_root,
        start_year=2020,
        start_month=2,
        n_periods=2,
        max_articles_per_day=3,
    )
