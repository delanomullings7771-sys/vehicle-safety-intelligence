"""Build the final crash models' features from one CRSS crash record.

A crash record is the crash's rows from the CRSS tables, in CRSS codes, nested by table:
    {"accident": [{...one row...}], "vehicle": [{...}, ...], "person": [{...}, ...], ...}
Only the fields in crash_spec.json are read; any other CRSS fields are ignored, so a full CRSS release works.

The rules are the ones src/build_crss_features.py used to build the training data:
  * accident fields are kept as they are; a code outside the training-year code list becomes -1;
  * each other table is summarised per crash: a row count, a count of rows per code, and the min/max of
    numeric fields, after unknown-value sentinels (e.g. speed 998) are set to missing.
src/export_artifacts.py checks that this reproduces the training features exactly for real 2024 crashes.
"""
import math

ACCIDENT_PREFIX = "acc"


def parse_feature(name: str) -> tuple[str, str | None, str, str | None]:
    """'person__AIR_BAG=8' -> ('person', 'AIR_BAG', 'count', '8'); 'vehicle__TRAV_SP__max' -> (..., 'max', None)."""
    prefix, rest = name.split("__", 1)
    table = "accident" if prefix == ACCIDENT_PREFIX else prefix
    if rest == "n_rows":
        return table, None, "n_rows", None
    if "=" in rest:
        field, code = rest.split("=", 1)
        return table, field, "count", code
    if rest.endswith(("__min", "__max")):
        field, agg = rest.rsplit("__", 1)
        return table, field, agg, None
    return table, rest, "value", None


def _num(v):
    if v is None or (isinstance(v, str) and not v.strip()):
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _same_code(v, code: str) -> bool:
    x, c = _num(v), _num(code)
    if not math.isnan(c):
        return not math.isnan(x) and x == c
    return str(v) == code


def _clean(v, field: str, sentinels: dict) -> float:
    x = _num(v)
    return x if x < sentinels.get(field, math.inf) else math.nan


def normalise(record: dict) -> dict:
    """Lower-case table names and upper-case field names, as in the training code."""
    return {t.lower(): [{str(k).upper(): v for k, v in row.items()} for row in rows] for t, rows in record.items()}


def missing_fields(record: dict, spec: dict) -> list[str]:
    """Required 'table.field' entries the record does not supply.

    A table that is absent is missing. A table given as an empty list means the crash has no such rows
    (for example no pedestrians), which is a real value, as in training.
    """
    out = []
    for key in spec["fields"]:
        table, field = key.split(".")
        rows = record.get(table)
        if rows is None:
            out.append(key)
        elif field != "n_rows" and rows and not any(field in r for r in rows):
            out.append(key)
    return out


def build_features(record: dict, features: list[str], spec: dict) -> dict:
    record = normalise(record)
    gone = set(missing_fields(record, spec))
    sentinels, vocab = spec["numeric_sentinels"], spec["accident_codes"]
    out = {}
    for name in features:
        table, field, kind, code = parse_feature(name)
        if f"{table}.{field or 'n_rows'}" in gone:
            out[name] = math.nan  # flagged to the user, never filled with a guessed value here
            continue
        rows = record.get(table, [])
        if kind == "value":  # accident level: one row per crash
            v = rows[0].get(field) if rows else None
            if field in sentinels:
                out[name] = _clean(v, field, sentinels)
            else:
                x = _num(v)
                out[name] = x if x in vocab[field] else -1.0
        elif kind == "n_rows":
            out[name] = float(len(rows))
        elif kind == "count":
            out[name] = float(sum(_same_code(r.get(field), code) for r in rows))
        else:
            vals = [x for x in (_clean(r.get(field), field, sentinels) for r in rows) if not math.isnan(x)]
            out[name] = (min(vals) if kind == "min" else max(vals)) if vals else math.nan
    return out
