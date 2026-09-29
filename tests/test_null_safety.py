"""
Regression tests for the null-safety pass across api/schemas.py,
application_mapper.py, and history_mapper.py. Covers the 4 concrete bug
reports plus the general "every OCR field can be null" contract.

No trained artifacts or pandas needed for most of these — run:
    python -m pytest tests/test_null_safety.py -v
"""
import copy
import math

import pytest

from api.schemas import NewCustomerApplicationPayload, ReturningCustomerApplicationPayload
from src.adapters.application_mapper import map_payload_to_application_row
from src.adapters.history_mapper import build_history_features_for_customer


def _minimal_returning_base() -> dict:
    """The smallest ReturningCustomerApplicationPayload that satisfies the
    still-required fields (application_id, documents, form_data as a
    Union[...]-safe FormData block isn't required to be populated at all
    since every field is Optional now)."""
    return {
        "application_id": "T-001",
        "applicant_type": "individual",
        "submission_timestamp": "2026-09-28T00:00:00Z",
        "is_returning_customer": True,
        "documents": [{"document_type": "national_id", "overall_quality_score": 0.9,
                       "is_tampered_suspected": False, "extracted_fields": {}}],
        "national_id_fields": {},
        "salary_certificate_fields": {},
        "bank_statement_fields": {},
        "iscore_report_fields": {"is_available": True, "bureau_facilities": []},
        "form_data": {},
        "internal_history": {},
        "consistency_checks": {},
    }


# ------------------------------------------------------------------
# The 4 concrete bug reports
# ------------------------------------------------------------------
def test_bug1_bare_int_statement_period_months_no_longer_422s():
    p = _minimal_returning_base()
    p["bank_statement_fields"]["statement_period_months"] = 5  # bare int, not {value, confidence}
    ReturningCustomerApplicationPayload(**p)  # must not raise


def test_bug1_wrapped_statement_period_months_still_works():
    p = _minimal_returning_base()
    p["bank_statement_fields"]["statement_period_months"] = {"value": 6, "confidence": 0.9}
    ReturningCustomerApplicationPayload(**p)


@pytest.mark.parametrize("field,null_value", [
    ("lender_name", None),
    ("overdue_days", None),
    ("overdue_amount", None),
])
def test_bugs_2_3_4_null_bureau_facility_fields_no_longer_422(field, null_value):
    p = _minimal_returning_base()
    p["iscore_report_fields"]["bureau_facilities"] = [{
        "granted_amount": 20000.0, "outstanding_amount": 5000.0,
        "installment_amount": 500.0, "status": "active",
        "start_date": "2022-01-01", field: null_value,
    }]
    m = ReturningCustomerApplicationPayload(**p)  # must not raise
    # and history_mapper must not crash on the resulting None either
    dump = m.model_dump(mode="json")
    hist = build_history_features_for_customer(
        dump["internal_history"], dump["iscore_report_fields"]["bureau_facilities"]
    )
    assert hist["BUREAU_FACILITY_COUNT"] == 1


def test_clean_performing_facility_with_all_overdue_fields_omitted():
    """The real-world case the bug report describes: a performing facility
    that simply never populates overdue_days/overdue_amount at all."""
    p = _minimal_returning_base()
    p["iscore_report_fields"]["bureau_facilities"] = [{
        "lender_name": "QNB", "status": "active",
        "granted_amount": 10000.0, "outstanding_amount": 1000.0,
    }]
    m = ReturningCustomerApplicationPayload(**p)
    dump = m.model_dump(mode="json")
    hist = build_history_features_for_customer(
        dump["internal_history"], dump["iscore_report_fields"]["bureau_facilities"]
    )
    assert hist["BUREAU_TOTAL_OVERDUE"] == 0.0
    assert hist["BUREAU_MAX_OVERDUE_DAYS"] == 0


# ------------------------------------------------------------------
# General null-safety contract: numeric -> NaN (key present),
# categorical -> omitted key
# ------------------------------------------------------------------
def test_fully_null_new_customer_payload_is_schema_valid_and_maps_safely():
    p = {
        "application_id": "NULL-STORM",
        "applicant_type": "individual",
        "submission_timestamp": "2026-09-28T00:00:00Z",
        "documents": [{"document_type": "national_id", "overall_quality_score": 0.5,
                       "is_tampered_suspected": False, "extracted_fields": {}}],
        "national_id_fields": {},
        "salary_certificate_fields": {},
        "bank_statement_fields": {},
        "iscore_report_fields": {"is_available": False},
        "form_data": {},
        "internal_history": {},
        "consistency_checks": {},
    }
    m = NewCustomerApplicationPayload(**p)  # must not raise
    row = map_payload_to_application_row(m.model_dump(mode="json"))

    # numeric core columns: key present, value NaN
    for col in ["DAYS_BIRTH", "DAYS_EMPLOYED", "AMT_CREDIT", "AMT_ANNUITY",
                "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE", "CNT_CHILDREN",
                "CNT_FAM_MEMBERS", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        assert col in row, f"{col} must always be present"
        assert math.isnan(row[col]), f"{col} should be NaN when unknown"

    # categorical columns: key omitted entirely
    for col in ["CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
                "NAME_EDUCATION_TYPE", "NAME_CONTRACT_TYPE",
                "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        assert col not in row, f"{col} should be omitted, not null, when unknown"


def test_partial_payload_mixes_known_and_unknown_correctly():
    p = {
        "application_id": "PARTIAL-001",
        "applicant_type": "individual",
        "submission_timestamp": "2026-09-28T00:00:00Z",
        "documents": [{"document_type": "national_id", "overall_quality_score": 0.9,
                       "is_tampered_suspected": False, "extracted_fields": {}}],
        "national_id_fields": {"gender": {"value": "F", "confidence": 0.9}},  # age_years still missing
        "salary_certificate_fields": {"declared_net_salary": {"value": 15000.0, "confidence": 0.9}},
        "bank_statement_fields": {},
        "iscore_report_fields": {"is_available": False},
        "form_data": {"requested_amount": 50000.0, "owns_car": False},
        "internal_history": {},
        "consistency_checks": {},
    }
    m = NewCustomerApplicationPayload(**p)
    row = map_payload_to_application_row(m.model_dump(mode="json"))

    assert math.isnan(row["DAYS_BIRTH"])          # age_years was missing
    assert row["AMT_INCOME_TOTAL"] == 15000.0 * 12  # salary was present
    assert row["AMT_CREDIT"] == 50000.0             # requested_amount was present
    assert row["CODE_GENDER"] == "F"                # gender was present -> key exists
    assert row["FLAG_OWN_CAR"] == "N"                # owns_car False is NOT "missing"
    assert "NAME_FAMILY_STATUS" not in row           # family_status was missing -> omitted


def test_goods_price_real_zero_is_not_confused_with_missing():
    """goods_price=0 (a real cash loan with no goods) must NOT trigger the
    requested_amount fallback the way goods_price=None does."""
    p = {
        "application_id": "GOODS-001",
        "applicant_type": "individual",
        "submission_timestamp": "2026-09-28T00:00:00Z",
        "documents": [{"document_type": "national_id", "overall_quality_score": 0.9,
                       "is_tampered_suspected": False, "extracted_fields": {}}],
        "national_id_fields": {}, "salary_certificate_fields": {}, "bank_statement_fields": {},
        "iscore_report_fields": {"is_available": False},
        "form_data": {"requested_amount": 90000.0, "goods_price": 0.0},
        "internal_history": {}, "consistency_checks": {},
    }
    m = NewCustomerApplicationPayload(**p)
    row = map_payload_to_application_row(m.model_dump(mode="json"))
    # goods_price=0 falls back to requested_amount, same as the pre-existing
    # (pre-null-safety) behavior for a non-goods cash loan.
    assert row["AMT_GOODS_PRICE"] == 90000.0