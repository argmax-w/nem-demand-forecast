"""AEMO regional demand acquisition and parsing.

The target is NSW1 demand from AEMO's aggregated price-and-demand archive
(the ``TOTALDEMAND`` column of ``DISPATCHREGIONSUM``, the regional demand met
by scheduled, semi-scheduled and significant generation), published as
monthly ``PRICE_AND_DEMAND_{YYYYMM}_{REGION}.csv`` files at five-minute
dispatch resolution. This series is chosen for its depth: the archive reaches
back several years, which is what lets the evaluation split cover every
season in both validation and test without any look-ahead. The half-hourly
``OPERATIONAL_DEMAND.ACTUAL`` reports on NEMWeb (parsed by the functions at
the foot of this module) retain only about thirteen months and so cannot
support an all-season split; they remain available as an alternative source.

AEMO stamps dispatch intervals with their ending time in market time (AEST,
UTC+10, no daylight saving). The project stores everything in UTC with
period-start timestamps: five-minute readings are averaged into the half hour
whose ending boundary they fall under, the ending stamp is shifted back by
thirty minutes to a period start and the fixed-offset market time is then
converted to UTC. Plots convert back to AEST at the display layer.
"""

from __future__ import annotations

import io
import re
import zipfile
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import requests

MARKET_TZ = "Australia/Brisbane"
PRICE_DEMAND_URL = (
    "https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{ym}_{region}.csv"
)


def download_price_demand(start: date, end: date, region: str, raw_dir: Path) -> list[Path]:
    """Download the monthly price-and-demand CSVs spanning the window.

    Files already present are left untouched, so reruns are cheap. The
    window is widened by one month on each side so the half-hourly grid has
    no edge effects after timezone conversion.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    months = []
    cursor = date(start.year, start.month, 1) - timedelta(days=1)
    cursor = date(cursor.year, cursor.month, 1)
    last = date(end.year, end.month, 1) + timedelta(days=32)
    last = date(last.year, last.month, 1)
    while cursor <= last:
        months.append(f"{cursor.year}{cursor.month:02d}")
        nxt = cursor + timedelta(days=32)
        cursor = date(nxt.year, nxt.month, 1)

    paths = []
    for ym in months:
        path = raw_dir / f"PRICE_AND_DEMAND_{ym}_{region}.csv"
        if not path.exists() or path.stat().st_size == 0:
            response = requests.get(PRICE_DEMAND_URL.format(ym=ym, region=region), timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
        paths.append(path)
    return paths


def load_demand_csv(start: date, end: date, region: str, raw_dir: Path) -> pl.DataFrame:
    """Half-hourly NSW1 demand from the price-and-demand CSVs, UTC period start.

    Parameters
    ----------
    start, end
        Inclusive period-start window in market time.
    region
        NEM region identifier, for example ``NSW1``.
    raw_dir
        Directory holding (or receiving) the monthly CSVs.

    Returns
    -------
    polars.DataFrame
        Columns ``ts`` (UTC period start, strictly increasing) and
        ``demand_mw``, the mean of the five-minute ``TOTALDEMAND`` readings
        in each half hour.
    """
    paths = download_price_demand(start, end, region, raw_dir)
    lower = pl.datetime(start.year, start.month, start.day)
    upper = pl.datetime(end.year, end.month, end.day) + pl.duration(days=1)
    frames = [pl.read_csv(path, schema_overrides={"SETTLEMENTDATE": pl.String}) for path in paths]
    return (
        pl.concat(frames)
        .filter(pl.col("REGION") == region)
        .with_columns(settlement=pl.col("SETTLEMENTDATE").str.to_datetime("%Y/%m/%d %H:%M:%S"))
        .with_columns(
            # Five-minute stamps are period ending; the containing half hour
            # ends at the next 30-minute boundary (a stamp already on a
            # boundary stays put).
            interval_end=pl.col("settlement").dt.offset_by("-1s").dt.truncate("30m")
            + pl.duration(minutes=30)
        )
        .group_by("interval_end")
        .agg(demand_mw=pl.col("TOTALDEMAND").mean())
        .with_columns(ts=pl.col("interval_end") - pl.duration(minutes=30))
        .filter((pl.col("ts") >= lower) & (pl.col("ts") < upper))
        .with_columns(ts=pl.col("ts").dt.replace_time_zone(MARKET_TZ).dt.convert_time_zone("UTC"))
        .select("ts", "demand_mw")
        .sort("ts")
    )


ARCHIVE_URL = "https://nemweb.com.au/Reports/ARCHIVE/Operational_Demand/ACTUAL_HH/"
_HEADERS = {"User-Agent": "nem-demand-forecast (github.com/argmax-w/nem-demand-forecast)"}
_ZIP_NAME = re.compile(r"PUBLIC_ACTUAL_OPERATIONAL_DEMAND_HH_(\d{8})\.zip", re.IGNORECASE)
_DATA_ROW_PREFIX = b"D,OPERATIONAL_DEMAND,ACTUAL"
_RAW_COLUMNS = [
    "row_type",
    "table",
    "subtable",
    "version",
    "regionid",
    "interval_datetime",
    "operational_demand",
    "operational_demand_adjustment",
    "wdr_estimate",
    "lastchanged",
]


def list_archive_weeks() -> list[date]:
    """List the week-start dates of the weekly zips currently on NEMWeb."""
    response = requests.get(ARCHIVE_URL, headers=_HEADERS, timeout=60)
    response.raise_for_status()
    stamps = _ZIP_NAME.findall(response.text)
    weeks = sorted({date(int(s[:4]), int(s[4:6]), int(s[6:8])) for s in stamps})
    return weeks


def required_weeks(start: date, end: date) -> list[date]:
    """Week-start dates whose zips cover the inclusive period-start window.

    Weekly zips are named by their first market day (a Sunday) and hold seven
    days of interval-ending stamps from ``00:00`` on that day. A period-start
    window ``[start, end]`` needs stamps from ``start 00:30`` through
    ``end + 1 day 00:00``, so the last half hour of a window ending on a
    Saturday lives in the following week's zip.
    """
    first = start - timedelta(days=(start.weekday() + 1) % 7)
    last_stamp_day = end + timedelta(days=1)
    last = last_stamp_day - timedelta(days=(last_stamp_day.weekday() + 1) % 7)
    weeks = []
    week = first
    while week <= last:
        weeks.append(week)
        week += timedelta(days=7)
    return weeks


def download_archive(start: date, end: date, dest: Path) -> list[Path]:
    """Download the weekly zips covering the window, skipping existing files.

    Parameters
    ----------
    start, end
        Inclusive period-start window in market time.
    dest
        Directory for the raw zips, typically ``data/raw/aemo``.

    Returns
    -------
    list of Path
        Paths of the zips covering the window, in chronological order.

    Raises
    ------
    RuntimeError
        If a needed week has dropped out of the rolling NEMWeb archive. The
        remedy is to move the window start forward in ``config/default.yaml``.
    """
    dest.mkdir(parents=True, exist_ok=True)
    available = set(list_archive_weeks())
    paths = []
    for week in required_weeks(start, end):
        name = f"PUBLIC_ACTUAL_OPERATIONAL_DEMAND_HH_{week:%Y%m%d}.zip"
        path = dest / name
        if not path.exists():
            if week not in available:
                raise RuntimeError(
                    f"Week {week} is outside the rolling NEMWeb archive; "
                    "move window.start forward in the configuration."
                )
            response = requests.get(ARCHIVE_URL + name, headers=_HEADERS, timeout=300)
            response.raise_for_status()
            path.write_bytes(response.content)
        paths.append(path)
    return paths


def parse_archive_zip(path: Path, region: str) -> pl.DataFrame:
    """Parse one weekly zip into a polars frame of half-hourly demand.

    Every inner zip holds a small AEMO CSV whose ``D`` rows carry one record
    per region. The rows are filtered as raw bytes and parsed in a single
    vectorised ``read_csv`` call rather than file by file.
    """
    rows: list[bytes] = []
    with zipfile.ZipFile(path) as outer:
        for inner_name in outer.namelist():
            with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
                for member in inner.namelist():
                    for line in inner.read(member).splitlines():
                        if line.startswith(_DATA_ROW_PREFIX):
                            rows.append(line)
    frame = pl.read_csv(
        io.BytesIO(b"\n".join(rows)),
        has_header=False,
        new_columns=_RAW_COLUMNS,
        schema_overrides={
            "operational_demand": pl.Float64,
            "operational_demand_adjustment": pl.Float64,
            "wdr_estimate": pl.Float64,
        },
    )
    return (
        frame.filter(pl.col("regionid") == region)
        .with_columns(
            pl.col("interval_datetime").str.to_datetime("%Y/%m/%d %H:%M:%S"),
            pl.col("lastchanged").str.to_datetime("%Y/%m/%d %H:%M:%S"),
        )
        .select("regionid", "interval_datetime", "operational_demand", "lastchanged")
    )


def load_demand(
    start: date,
    end: date,
    region: str,
    raw_dir: Path,
) -> pl.DataFrame:
    """Load half-hourly operational demand for the window, downloading if needed.

    Parameters
    ----------
    start, end
        Inclusive period-start window in market time.
    region
        NEM region identifier, for example ``NSW1``.
    raw_dir
        Directory holding (or receiving) the raw weekly zips.

    Returns
    -------
    polars.DataFrame
        Columns ``ts`` (UTC period start, strictly increasing) and
        ``demand_mw``. The window bounds are market-time days, matching the
        archive layout; duplicate publications are resolved by keeping the
        latest ``LASTCHANGED``.
    """
    paths = download_archive(start, end, raw_dir)
    weekly = [parse_archive_zip(path, region) for path in paths]
    lower = pl.datetime(start.year, start.month, start.day)
    upper = pl.datetime(end.year, end.month, end.day) + pl.duration(days=1)
    return (
        pl.concat(weekly)
        .sort("lastchanged")
        .unique(subset="interval_datetime", keep="last")
        .with_columns(ts=pl.col("interval_datetime") - pl.duration(minutes=30))
        .filter((pl.col("ts") >= lower) & (pl.col("ts") < upper))
        .with_columns(
            # Market time is a fixed +10:00 offset, so this conversion is
            # unambiguous year-round.
            ts=pl.col("ts").dt.replace_time_zone(MARKET_TZ).dt.convert_time_zone("UTC")
        )
        .select("ts", demand_mw=pl.col("operational_demand"))
        .sort("ts")
    )
