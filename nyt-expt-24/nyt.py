import datetime
import json
import os
from pathlib import Path
from typing import Union, Sequence


import pandas as pd
import requests
import yfinance as yf


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
    with _get_nyt_fetch_path(year, month).open("w") as fp:
        json.dump(resp_json, fp)
