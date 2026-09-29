"""
Maps either intake payload (NewCustomerApplicationPayload or
ReturningCustomerApplicationPayload) into a single-row dict shaped like
a Kaggle application_train.csv row, so it can be fed unchanged into
features/application.py's engineer_application_features(fit=False, ...).

Only covers the application-level block. History features (PREV_*,
INST_*, BUREAU_*) are handled separately by history_mapper.py for
returning customers, or left as "no history" for new customers.

NULL-SAFETY CONTRACT (matches api/schemas.py's Optional fields)
-----------------------------------------------------------------
Every FieldWithConfidence-wrapped input this module reads can now be
None — the whole wrapper, or its .value — because api/schemas.py marks
them Optional. Two different, deliberate strategies handle that,
depending on how application.py's engineer_application_features()
touches the resulting column:

  1. NUMERIC columns application.py accesses by direct bracket indexing
     (df["AMT_CREDIT"], df["DAYS_BIRTH"], etc. — see its
     _core_ratios_and_age / _secondary_features) MUST always be present
     as a key in the returned dict, or a single-row DataFrame built from
     it won't have that column and application.py will KeyError
     immediately. So for these, a missing input becomes an EXPLICIT
     float('nan') value — the key stays, the value signals "unknown" —
     and pandas/XGBoost/LightGBM's native NaN handling takes over from
     there (see api/schemas.py's module docstring for the trade-off).

  2. CATEGORICAL columns (CODE_GENDER, NAME_*, FLAG_OWN_*) are never
     accessed by direct indexing inside application.py — they just ride
     along as extra DataFrame columns. For these, a missing input means
     the key is OMITTED from the returned dict entirely. Downstream,
     build_features.transform_serve_row() already does
     row.get(col, "Missing") for every categorical column in
     fit_artifacts.categorical_cols, so an absent key resolves to the
     same "Missing" category application.py itself uses for genuinely
     missing training-data categoricals. No special-casing needed there.

Columns with NO source in either payload (OCCUPATION_TYPE,
ORGANIZATION_TYPE, REGION_RATING_CLIENT, FLAG_DOCUMENT_*, etc.) are still
deliberately absent from the returned dict, same as before this change —
that's strategy 2, they just never had a source key to begin with.
"""
import math
from typing import Optional, Union

from src import config

# Kaggle NAME_FAMILY_STATUS categories the OneHotEncoder was fit on.
_FAMILY_STATUS_MAP = {
    "single": "Single / not married",
    "married": "Married",
    "civil_marriage": "Civil marriage",
    "widow": "Widow",
    "separated": "Separated",
}

_HOUSING_TYPE_MAP = {
    "owned": "House / apartment",
    "rented": "Rented apartment",
    "with_parents": "With parents",
    "municipal": "Municipal apartment",
    "office": "Office apartment",
    "co_op": "Co-op apartment",
}

_EDUCATION_TYPE_MAP = {
    "secondary": "Secondary / secondary special",
    "higher_education": "Higher education",
    "incomplete_higher": "Incomplete higher",
    "lower_secondary": "Lower secondary",
    "academic_degree": "Academic degree",
}

# form_data.loan_purpose -> Kaggle's NAME_CONTRACT_TYPE. Extend this map as
# new loan_purpose values show up; anything unmapped defaults to "Cash loans".
_CONTRACT_TYPE_MAP = {
    "personal_cash": "Cash loans",
    "revolving": "Revolving loans",
}

_NAN = float("nan")


def _fwc_value(field: Optional[dict]) -> object:
    """Safely pull .value out of an Optional[FieldWithConfidence]-shaped
    dict. Returns None if the wrapper itself is missing/null OR if it's
    present but its value is null — both mean "we don't have this"."""
    if not field:
        return None
    return field.get("value")


def _to_float_or_nan(x) -> float:
    """Numeric strategy (see module docstring, case 1): None or a
    non-numeric value both become NaN rather than raising or silently
    becoming 0, which would misrepresent a real zero-value applicant."""
    if x is None:
        return _NAN
    try:
        return float(x)
    except (TypeError, ValueError):
        return _NAN


def _normalize_iscore(credit_score: float) -> float:
    """ASSUMPTION 2 in config.py — see that docstring before trusting this."""
    clipped = max(config.ISCORE_MIN, min(config.ISCORE_MAX, credit_score))
    return (clipped - config.ISCORE_MIN) / (config.ISCORE_MAX - config.ISCORE_MIN)


def map_payload_to_application_row(payload: Union[dict, object]) -> dict:
    """
    Parameters
    ----------
    payload : a NewCustomerApplicationPayload or ReturningCustomerApplicationPayload
        instance (or its .model_dump() dict) — both have identical shape for
        every field this function reads.

    Returns
    -------
    dict of {kaggle_column_name: value}, ready to become a one-row DataFrame
    and passed into engineer_application_features(df, fit=False, fit_artifacts=...).
    Numeric core columns are always present (NaN when unknown); categorical
    columns are omitted when unknown. See module docstring.
    """
    p = payload if isinstance(payload, dict) else payload.model_dump()

    national_id = p.get("national_id_fields") or {}
    salary = p.get("salary_certificate_fields") or {}
    form = p.get("form_data") or {}
    iscore = p.get("iscore_report_fields") or {}

    age_years = _fwc_value(national_id.get("age_years"))
    employed_years = _fwc_value(salary.get("employment_tenure_years"))
    monthly_salary = _fwc_value(salary.get("declared_net_salary"))
    gender = _fwc_value(national_id.get("gender"))

    # ---- numeric core columns: ALWAYS present, NaN when unknown (case 1) ----
    row = {
        # Back-derived so engineer_application_features's existing
        # DAYS_BIRTH / DAYS_EMPLOYED math runs completely unchanged. NaN
        # propagates naturally through -x*365.25 (NaN * anything = NaN).
        "DAYS_BIRTH": -round(age_years * 365.25) if age_years is not None else _NAN,
        "DAYS_EMPLOYED": -round(employed_years * 365.25) if employed_years is not None else _NAN,

        # ASSUMPTION 1 in config.py — monthly salary annualized to match
        # Kaggle's AMT_INCOME_TOTAL scale.
        "AMT_INCOME_TOTAL": (monthly_salary * config.INCOME_ANNUALIZATION_MULTIPLIER)
                            if monthly_salary is not None else _NAN,
        "AMT_CREDIT": _to_float_or_nan(form.get("requested_amount")),
        "AMT_ANNUITY": _to_float_or_nan(form.get("requested_annuity")),
        "CNT_CHILDREN": _to_float_or_nan(form.get("children_count")),
        "CNT_FAM_MEMBERS": _to_float_or_nan(form.get("family_members_count")),
    }

    # AMT_GOODS_PRICE: previously `goods_price or requested_amount` (a real
    # 0 or missing goods_price falls back to the requested amount, common
    # for non-goods cash loans). Now null-safe: only fall back when
    # goods_price is genuinely absent (None), not when it's a real 0.
    goods_price = form.get("goods_price")
    if goods_price is not None:
        row["AMT_GOODS_PRICE"] = _to_float_or_nan(goods_price) if goods_price != 0 else _to_float_or_nan(form.get("requested_amount"))
    else:
        row["AMT_GOODS_PRICE"] = _to_float_or_nan(form.get("requested_amount"))

    # ---- categorical columns: OMITTED when unknown (case 2) ----
    if gender is not None:
        row["CODE_GENDER"] = gender  # already 'M'/'F', matches Kaggle directly

    family_status = form.get("family_status")
    if family_status is not None:
        row["NAME_FAMILY_STATUS"] = _FAMILY_STATUS_MAP.get(family_status, "Unknown")

    housing_type = form.get("housing_type")
    if housing_type is not None:
        row["NAME_HOUSING_TYPE"] = _HOUSING_TYPE_MAP.get(housing_type, "House / apartment")

    education_type = form.get("education_type")
    if education_type is not None:
        row["NAME_EDUCATION_TYPE"] = _EDUCATION_TYPE_MAP.get(education_type, "Secondary / secondary special")

    loan_purpose = form.get("loan_purpose")
    if loan_purpose is not None:
        row["NAME_CONTRACT_TYPE"] = _CONTRACT_TYPE_MAP.get(loan_purpose, "Cash loans")

    owns_car = form.get("owns_car")
    if owns_car is not None:
        row["FLAG_OWN_CAR"] = "Y" if owns_car else "N"

    owns_realty = form.get("owns_realty")
    if owns_realty is not None:
        row["FLAG_OWN_REALTY"] = "Y" if owns_realty else "N"

    # EXT_SOURCE_1/2/3: proxy from I-Score if available and numeric, else NaN.
    # IMPORTANT: these keys must always be present (even as NaN) — application.py's
    # _secondary_features() does df[[EXT_SOURCE_1, _2, _3]].mean(axis=1), which
    # raises KeyError on missing COLUMNS (as opposed to missing VALUES).
    # "Unknown" I-Score (report unavailable, field omitted/null, {"value":
    # null}, or a non-numeric value) all become NaN, so the fit-time median
    # imputation handles it. A real score of 0 is NOT "unknown" — it is
    # clipped to ISCORE_MIN and maps to EXT_SOURCE 0.0 (worst possible).
    raw_score = _fwc_value(iscore.get("credit_score"))
    try:
        score_value = float(raw_score) if raw_score is not None else None
    except (TypeError, ValueError):
        score_value = None

    if iscore.get("is_available") and score_value is not None:
        proxied = _normalize_iscore(score_value)
    else:
        proxied = _NAN
    row["EXT_SOURCE_1"] = proxied
    row["EXT_SOURCE_2"] = proxied
    row["EXT_SOURCE_3"] = proxied

    # OBS_30/60_CNT_SOCIAL_CIRCLE, DEF_30/60_CNT_SOCIAL_CIRCLE: deliberately
    # NOT set. application.py's _secondary_features() already does
    # df.get("OBS_30_CNT_SOCIAL_CIRCLE", 0), so omitting them here correctly
    # yields 0 without any special-case code.

    return row