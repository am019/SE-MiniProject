"""Dataset schema detection and validation (FR-01, FR-02, FR-03, SR-03, NFR-01).

Vectorised with pandas so a 50,000 x 20 file validates in well under 10 s.
Cell values are treated strictly as data: they are parsed, never evaluated, and
never echoed back in reports (SR-03, SR-06) - only row numbers and counts are.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import subdir
from .errors import AppError

MAX_ROWS_PREVIEW = 20
PRICE_REQUIRED = ["date", "close"]
OPTION_REQUIRED = ["date", "strike", "expiry", "option_type", "underlying_price", "option_price"]
MIN_PRICE_ROWS = 120
MIN_OPTION_ROWS = 200
MIN_OPTION_DATES = 60

_ALIASES = {
    "date": "date", "timestamp": "date", "trade_date": "date", "quote_date": "date",
    "close": "close", "adj_close": "close", "close_price": "close", "price": "close",
    "strike": "strike", "strike_price": "strike", "k": "strike",
    "expiry": "expiry", "expiration": "expiry", "expiry_date": "expiry", "maturity": "expiry",
    "expiration_date": "expiry",
    "option_type": "option_type", "type": "option_type", "cp": "option_type", "cp_flag": "option_type",
    "call_put": "option_type", "right": "option_type",
    "underlying_price": "underlying_price", "underlying": "underlying_price", "spot": "underlying_price",
    "s": "underlying_price", "underlying_close": "underlying_price",
    "option_price": "option_price", "mid": "option_price", "premium": "option_price",
    "option_close": "option_price", "last": "option_price",
    "rate": "rate", "risk_free_rate": "rate", "r": "rate",
}


def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping, seen = {}, set()
    for c in df.columns:
        n = _norm(c)
        n = _ALIASES.get(n, n)
        if n in seen:  # keep the first column for each canonical name
            n = f"{n}__dup{len(seen)}"
        seen.add(n)
        mapping[c] = n
    return df.rename(columns=mapping)


def _issue(code, severity, field, message, rows=None, count=None):
    d = {"code": code, "severity": severity, "field": field, "message": message}
    if rows is not None:
        d["rows"] = [int(r) for r in rows[:MAX_ROWS_PREVIEW]]
    d["count"] = int(count if count is not None else (len(rows) if rows is not None else 1))
    return d


def parse_dates(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce", format="ISO8601")
    bad = out.isna() & s.notna() & (s.astype(str).str.strip() != "")
    if bad.any():  # fall back to flexible parsing only for the rows ISO could not read
        try:
            alt = pd.to_datetime(s[bad], errors="coerce", format="mixed", dayfirst=False)
            out.loc[bad] = alt
        except (ValueError, TypeError):
            pass
    return out


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    if b"\x00" in raw[:65536]:
        raise AppError("FILE_TYPE_UNSUPPORTED", "The file is not a plain-text CSV.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise AppError("SCHEMA_INVALID", "The file is not valid UTF-8 text.")
    if not text.strip():
        raise AppError("SCHEMA_INVALID", "The file is empty.")
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, skipinitialspace=True,
                         on_bad_lines="skip", engine="c")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        raise AppError("SCHEMA_INVALID", "The file could not be parsed as CSV.")
    df = df.replace(r"^\s*$", np.nan, regex=True)
    return df


def detect_type(cols: set) -> str | None:
    if set(OPTION_REQUIRED) <= cols:
        return "OPTION_CHAIN"
    if set(PRICE_REQUIRED) <= cols:
        return "PRICE_SERIES"
    return None


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def validate(raw: bytes) -> dict:
    """Return {type, issues, clean(DataFrame|None), stats}. Raises AppError(SCHEMA_INVALID) when
    required columns are missing."""
    df = normalize_columns(read_csv_bytes(raw))
    cols = set(df.columns)
    dtype = detect_type(cols)
    if dtype is None:
        # decide which family the user most likely meant, to name the missing fields
        target = OPTION_REQUIRED if len(cols & set(OPTION_REQUIRED)) > 2 and "close" not in cols else PRICE_REQUIRED
        missing = [c for c in target if c not in cols]
        raise AppError("SCHEMA_INVALID",
                       "Required column(s) missing: " + ", ".join(f"'{m}'" for m in missing) + ".",
                       fields=missing)
    issues: list = []
    n = len(df)
    rownum = lambda mask: (np.flatnonzero(mask.to_numpy()) + 2)  # +2: header line + 1-based  # noqa: E731

    required = PRICE_REQUIRED if dtype == "PRICE_SERIES" else OPTION_REQUIRED
    clean = pd.DataFrame(index=df.index)

    # --- missing values in required columns
    for c in required:
        m = df[c].isna()
        if m.any():
            issues.append(_issue("MISSING_VALUE", "BLOCKING", c,
                                 f"Column '{c}' has missing values.", rownum(m)))
    # --- dates
    d = parse_dates(df["date"])
    badd = d.isna() & df["date"].notna()
    if badd.any():
        issues.append(_issue("DATE_UNPARSEABLE", "BLOCKING", "date",
                             "Some dates could not be parsed.", rownum(badd)))
    clean["date"] = d
    if dtype == "PRICE_SERIES":
        px = _num(df, "close")
        nn = px.isna() & df["close"].notna()
        if nn.any():
            issues.append(_issue("NOT_NUMERIC", "BLOCKING", "close", "Column 'close' has non-numeric values.",
                                 rownum(nn)))
        nonpos = px <= 0
        if nonpos.any():
            issues.append(_issue("NON_POSITIVE_PRICE", "BLOCKING", "close",
                                 "Column 'close' has zero or negative prices.", rownum(nonpos)))
        clean["close"] = px
        dup = d.duplicated(keep="first") & d.notna()
        if dup.any():
            issues.append(_issue("DUPLICATE_DATE", "BLOCKING", "date", "Duplicate dates found.", rownum(dup)))
        for opt in ("open", "high", "low", "volume"):
            if opt in df.columns:
                clean[opt] = _num(df, opt)
        if n < MIN_PRICE_ROWS:
            issues.append(_issue("INSUFFICIENT_DATA", "BLOCKING", None,
                                 f"At least {MIN_PRICE_ROWS} rows are required; found {n}."))
    else:
        for c in ("strike", "underlying_price", "option_price"):
            v = _num(df, c)
            nn = v.isna() & df[c].notna()
            if nn.any():
                issues.append(_issue("NOT_NUMERIC", "BLOCKING", c, f"Column '{c}' has non-numeric values.",
                                     rownum(nn)))
            bad = (v < 0) if c == "option_price" else (v <= 0)
            if bad.any():
                issues.append(_issue("NON_POSITIVE_PRICE", "BLOCKING", c,
                                     f"Column '{c}' has {'negative' if c == 'option_price' else 'zero or negative'} values.",
                                     rownum(bad)))
            if c == "option_price" and (v == 0).any():
                issues.append(_issue("ZERO_OPTION_PRICE", "WARNING", c,
                                     "Zero-priced options are ignored when building features.", rownum(v == 0)))
            clean[c] = v
        ex = parse_dates(df["expiry"])
        bade = ex.isna() & df["expiry"].notna()
        if bade.any():
            issues.append(_issue("DATE_UNPARSEABLE", "BLOCKING", "expiry", "Some expiry dates could not be parsed.",
                                 rownum(bade)))
        early = (ex <= d) & ex.notna() & d.notna()
        if early.any():
            issues.append(_issue("EXPIRY_NOT_AFTER_DATE", "BLOCKING", "expiry",
                                 "Expiry must be after the quote date.", rownum(early)))
        clean["expiry"] = ex
        t = df["option_type"].astype(str).str.strip().str.lower()
        typ = t.map({"c": "call", "call": "call", "p": "put", "put": "put"})
        badt = typ.isna() & df["option_type"].notna()
        if badt.any():
            issues.append(_issue("OPTION_TYPE_INVALID", "BLOCKING", "option_type",
                                 "option_type must be call/put (or C/P).", rownum(badt)))
        clean["option_type"] = typ
        if "rate" in df.columns:
            clean["rate"] = _num(df, "rate")
        dup = clean.duplicated(subset=["date", "strike", "expiry", "option_type"], keep="first")
        if dup.any():
            issues.append(_issue("DUPLICATE_CONTRACT", "BLOCKING", "date",
                                 "Duplicate (date, strike, expiry, option type) rows found.", rownum(dup)))
        nd = int(d.nunique())
        if n < MIN_OPTION_ROWS or nd < MIN_OPTION_DATES:
            issues.append(_issue("INSUFFICIENT_DATA", "BLOCKING", None,
                                 f"At least {MIN_OPTION_ROWS} rows over {MIN_OPTION_DATES} dates are required; "
                                 f"found {n} rows over {nd} dates."))
    if d.notna().any() and not d.dropna().is_monotonic_increasing:
        issues.append(_issue("NOT_SORTED", "WARNING", "date",
                             "Rows are not in chronological order; they will be sorted."))
    extra = [c[:64] for c in df.columns if c not in required and c not in clean.columns]
    stats = {"rows": int(n), "columns": int(len(df.columns)),
             "dateMin": d.min().strftime("%Y-%m-%d") if d.notna().any() else None,
             "dateMax": d.max().strftime("%Y-%m-%d") if d.notna().any() else None,
             "ignoredColumns": extra[:30]}
    blocking = any(i["severity"] == "BLOCKING" for i in issues)
    return {"type": dtype, "issues": issues, "clean": clean, "stats": stats, "blocking": blocking}


# -------------------------------------------------------------- persistence
def ds_dir() -> Path:
    return subdir("datasets")


def valid_id(ds_id: str) -> str:
    if not re.fullmatch(r"ds-[0-9a-f]{10,16}", ds_id or ""):
        raise AppError("DATASET_NOT_FOUND", "Dataset not found.")
    return ds_id


def load_meta(ds_id: str) -> dict:
    p = ds_dir() / f"{valid_id(ds_id)}.json"
    if not p.exists():
        raise AppError("DATASET_NOT_FOUND", "Dataset not found.")
    return json.loads(p.read_text())


def load_clean(ds_id: str) -> tuple[dict, pd.DataFrame]:
    """Re-validate the stored CSV and return (meta, cleaned frame sorted by date).
    Refuses datasets with blocking issues (FR-03)."""
    meta = load_meta(ds_id)
    if meta["status"] != "VALID":
        raise AppError("VALIDATION_FAILED",
                       "The dataset has blocking validation issues; feature building and model "
                       "execution are blocked.", fields=[i.get("field") for i in meta["issues"]
                                                          if i["severity"] == "BLOCKING" and i.get("field")])
    raw = (ds_dir() / f"{ds_id}.csv").read_bytes()
    res = validate(raw)
    clean = res["clean"].sort_values("date", kind="mergesort").reset_index(drop=True)
    return meta, clean
