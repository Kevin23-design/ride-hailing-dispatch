"""数据准备：Foursquare NYC 签到 -> 模拟订单（与 notebook 口径一致）。"""
import math
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

LAT_MIN, LAT_MAX = 40.60, 40.88
LON_MIN, LON_MAX = -74.05, -73.75
KM_PER_LAT = 111.0
KM_PER_LON = 111.0 * math.cos(math.radians(40.74))
W_KM = (LON_MAX - LON_MIN) * KM_PER_LON
H_KM = (LAT_MAX - LAT_MIN) * KM_PER_LAT

BASE_FARE = 5.0
PER_KM = 2.0
PICKUP_RADIUS = 3.0

ROOT = Path(__file__).resolve().parents[2]


def _find_csv():
    env = os.environ.get("NYC_CSV")
    cands = [Path(env)] if env else []
    cands += [ROOT / "data" / "dataset_TSMC2014_NYC.csv",
              ROOT.parent / "实验2_2" / "foursquare" / "dataset_TSMC2014_NYC.csv",
              ROOT.parent / "实验2_1" / "foursquare" / "dataset_TSMC2014_NYC.csv"]
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError("找不到 dataset_TSMC2014_NYC.csv")


def to_xy(lat, lon):
    return (lon - LON_MIN) * KM_PER_LON, (lat - LAT_MIN) * KM_PER_LAT


def load_checkins(path):
    df = pd.read_csv(path, encoding="latin-1")
    utc = pd.to_datetime(df["utcTimestamp"], format="%a %b %d %H:%M:%S +0000 %Y")
    df["local"] = utc + pd.to_timedelta(df["timezoneOffset"], unit="m")
    inside = df.latitude.between(LAT_MIN, LAT_MAX) & df.longitude.between(LON_MIN, LON_MAX)
    df = df[inside].copy()
    df["x"], df["y"] = to_xy(df.latitude.values, df.longitude.values)
    df["minute"] = df["local"].dt.hour * 60 + df["local"].dt.minute
    return df.sort_values(["userId", "local"]).reset_index(drop=True)


def build_trips(ck, min_gap=10, max_gap=180, min_km=1.0):
    prev = ck.groupby("userId")[["x", "y", "local", "minute"]].shift()
    gap = (ck["local"] - prev["local"]).dt.total_seconds() / 60
    km = (ck["x"] - prev["x"]).abs() + (ck["y"] - prev["y"]).abs()
    ok = (ck["local"].dt.date == prev["local"].dt.date) & gap.between(min_gap, max_gap) & (km >= min_km)
    trips = pd.DataFrame({
        "date": prev.loc[ok, "local"].dt.date.values,
        "minute": prev.loc[ok, "minute"].values.astype(int),
        "ox": prev.loc[ok, "x"].values, "oy": prev.loc[ok, "y"].values,
        "dx": ck.loc[ok, "x"].values, "dy": ck.loc[ok, "y"].values,
    })
    trips["trip_km"] = (trips.ox - trips.dx).abs() + (trips.oy - trips.dy).abs()
    trips["fare"] = BASE_FARE + PER_KM * trips["trip_km"]
    return trips.reset_index(drop=True)


def load_all():
    """返回 dict(ck, trips, trips_train, trips_test, ck_train)，带磁盘缓存。"""
    cache = ROOT / "results" / "_data_cache.pkl"
    if cache.is_file():
        with open(cache, "rb") as f:
            return pickle.load(f)
    ck = load_checkins(_find_csv())
    trips = build_trips(ck)
    dates = np.sort(trips["date"].unique())
    split = dates[int(len(dates) * 0.7)]
    out = dict(
        ck=ck[["x", "y", "minute", "local"]].copy(),
        trips=trips,
        trips_train=trips[trips.date < split].reset_index(drop=True),
        trips_test=trips[trips.date >= split].reset_index(drop=True),
        ck_train=ck.loc[ck["local"].dt.date < split, ["x", "y", "minute"]].reset_index(drop=True),
        split=split, n_days=len(dates),
    )
    cache.parent.mkdir(exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    return out
