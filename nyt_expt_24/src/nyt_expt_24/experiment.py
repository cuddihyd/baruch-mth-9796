import datetime
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Union, Sequence, Generator


import pandas as pd
import requests
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AdamW,
)
from torch.utils.data import DataLoader
import yfinance as yf


def train_model(dataset):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3
    )
    model.to(device)
    model.train()  # Put the model into "training-mode."
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    optim = AdamW(model.parameters(), lr=5e-5)
    for epoch in range(5):
        print("\nEpoch", epoch)
        for batch in train_loader:
            print(".", end="")
            optim.zero_grad()  # Zero-out gradients.
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()  # calculate gradient: dloss/dweights
            optim.step()
    print("Done")
    return model


class GenericDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict, labels: Sequence) -> None:
        """
        :param encodings: A dict like {'input_ids':[[token11, token12, ...], [token_n1, token_n2]],  'attention_mask': [[...],...].
        :param labels: A sequence of labels e.g. 0,1,2, etc.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration()
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def iter_nytimes_articles(start_year, start_month, n_months, data_root):
    for year, month in _iter_month_params(start_year, start_month, n_months):
        text_file = load_nyt_file(year, month, data_root)
        print(year, month, len(text_file["response"]["docs"]), "articles.")
        for article in text_file["response"]["docs"]:
            yield article


@dataclass
class DatedText:
    pub_date: datetime.date
    text: str


def make_dated_text_dated_index_dataset(
    dated_text_generator: Generator[DatedText, None, None],
    market_index: pd.Series,
    tokenizer,
) -> GenericDataset:
    texts = []
    labels = []
    for dated_text in dated_text_generator:
        texts.append(dated_text.text)
        labels.append(market_index.asof(dated_text.pub_date))
    encodings = tokenizer(texts, truncation=True, padding=True)
    return GenericDataset(encodings=encodings, labels=labels)


def nyt_article_to_dated_text(article):
    headline = article["headline"]["main"]
    pub_date = article["pub_date"]
    dt_pub_date = pd.Timestamp(pub_date.split("+")[0])
    return DatedText(text=headline, pub_date=dt_pub_date)


def make_nytimes_dataset(
    start_year: int,
    start_month: int,
    n_months: int,
    k1: float,
    k2: float,
    data_root: Union[Path, str],
) -> torch.utils.data.Dataset:
    data_root = Path(data_root)
    spy = load_yfinance_spy(data_root)["Close"]
    rets = spy.pct_change()
    zs_cats = zsbucket(rets, k1, k2)
    article_generator = map(
        nyt_article_to_dated_text,
        iter_nytimes_articles(
            start_year=start_year,
            start_month=start_month,
            n_months=n_months,
            data_root=data_root,
        ),
    )
    return make_dated_text_dated_index_dataset(
        dated_text_generator=article_generator,
        market_index=zs_cats,
        tokenizer=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    )

def _make_nytimes_dataset(
    start_year: int,
    start_month: int,
    n_months: int,
    k1: float,
    k2: float,
    data_root: Union[Path, str],
) -> torch.utils.data.Dataset:
    data_root = Path(data_root)
    idx = load_yfinance_spy(data_root)["Close"]
    rets = idx.pct_change()
    ret_scores = zsbucket(rets, k1, k2)
    for year, month in _iter_month_params(start_year, start_month, n_months):
        text_file = load_nyt_file(year, month, data_root)
        print(year, month, len(text_file["response"]["docs"]), "articles.")
        texts = []
        labels = []
        for article in text_file["response"]["docs"]:
            headline = article["headline"]["main"]
            pub_date = pd.Timestamp(
                article["pub_date"].split("+")[0]
            )  # Split off TZ-info so we get a tz-naive timestamp.
            texts.append(headline)
            label = ret_scores.asof(pub_date)
            labels.append(
                label
            )  # we expect ret_score index to be all tz-naive timestamps.

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    encodings = tokenizer(texts, truncation=True, padding=True)
    return GenericDataset(encodings=encodings, labels=labels)


def zsbucket(rets: pd.Series, k1: float, k2: float) -> pd.Series:
    """A series of bucket-categories calculated from current and all prior returns.
    :param k1: The zscore-boundary between buckets 0 and 1.
    :param k2: The zscore-boundary between buckets 1 and 2.
    """
    assert k1 < k2
    xzs = (rets - rets.expanding().mean()) / rets.expanding().std()

    def bucket(val: float):
        if val < k1:
            return 0
        elif k1 <= val < k2:
            return 1
        else:
            return 2

    return xzs.apply(bucket)


def setup_dataroot(data_root: Union[Path, str]) -> None:
    """Prep directory structure for our experiment."""
    data_root = Path(data_root)
    for subdir in ["nyt-archive-data/", "yfinance"]:
        subpath = data_root / subdir
        if subpath.is_dir():
            print(subpath, "OK.")
        else:
            os.mkdir(subpath)
            print("mkdir", subpath)


def load_yfinance_spy(data_root: Union[Path, str]) -> pd.DataFrame:
    target_path = Path(data_root) / "yfinance/spy.parquet"
    return pd.read_parquet(target_path)


def download_yfinance_spy(
    start: datetime.date, end: datetime.date, data_root: Union[Path, str]
) -> None:
    target_path = Path(data_root) / "yfinance/spy.parquet"
    spy = yf.download(["SPY"], start=start, end=end)
    spy.to_parquet(target_path)


def load_nyt_file(year: int, month: int, data_root: Union[Path, str]) -> dict:
    """Retrieve a NYT monthly archive record."""
    data_root = Path(data_root) / "nyt-archive-data/"
    with _get_nyt_fetch_path(year, month, data_root).open() as fp:
        return json.load(fp)


def download_nyt_monthly_archive_records(
    start_year: int,
    start_month: int,
    n_periods: int,
    apikey: str,
    data_root: Union[Path, str],
    sleep_sec=12,
):
    """Dump a sequence of NYT monthly archive records."""
    data_root = Path(data_root) / "nyt-archive-data/"
    for year, month in _iter_month_params(start_year, start_month, n_periods):
        print(year, month)
        try:
            resp = _fetch_nytimes_archive_data(year, month, apikey)
            _dump_json_to_file(resp.json(), year, month, data_root)
            print("..saved", len(resp.json()))
        except Exception as ex:
            print(ex)
        time.sleep(sleep_sec)


def _iter_month_params(year, month, n_periods):
    while n_periods > 0:
        yield (year, month)
        year = year + (month) // 12
        month = 1 + month % 12
        n_periods -= 1


def _fetch_nytimes_archive_data(
    year: int, month_num: int, apikey: str, data_root: Path
):
    archive_url_tmpl = "https://api.nytimes.com/svc/archive/v1/{year}/{month_num}.json?api-key={apikey}"
    archive_url = archive_url_tmpl.format(year=year, month_num=month_num, apikey=apikey)
    print(archive_url)
    return requests.get(archive_url)


def _get_nyt_fetch_path(year: int, month: int, data_root: Path):
    return data_root / "fetch-{year}-{month}.json".format(year=year, month=month)


def _dump_json_to_file(resp_json: object, year: int, month: int, data_root: Path):
    with _get_nyt_fetch_path(year, month, data_root).open("w") as fp:
        json.dump(resp_json, fp)


def make_nyt_test_data(
    source_data_root: str,
    target_data_root: str,
    start_year,
    start_month,
    n_periods,
    max_articles_per_day,
):
    target_data_root = Path(target_data_root) / "nyt-archive-data/"
    for year, month in _iter_month_params(start_year, start_month, n_periods):
        nyt_json = load_nyt_file(year, month, source_data_root)
        new_json = slim_nyt_json(nyt_json, max_articles_per_day=max_articles_per_day)
        _dump_json_to_file(
            resp_json=new_json, year=year, month=month, data_root=target_data_root
        )


def slim_nyt_json(nyt_json: dict, max_articles_per_day: int) -> dict:
    """Given a dict loaded from a NYTimes JSON file, return a slimmed-down version."""
    new_json = {"copyright": nyt_json["copyright"]}
    new_docs = _slim_docs(nyt_json["response"]["docs"], max_articles_per_day)
    new_json["response"] = {"docs": new_docs, "meta": {"hits": len(new_docs)}}
    return new_json


def _slim_docs(docs: Sequence[dict], max_articles_per_day: int) -> Sequence[dict]:
    perday = defaultdict(list)
    for doc in docs:
        dt = pd.Timestamp(doc["pub_date"]).normalize()
        if len(perday[dt]) < max_articles_per_day:
            perday[dt].append(doc)
    return reduce(lambda l, e: l + e, perday.values(), [])
