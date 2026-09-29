"""
Pydantic request schemas for the two intake payloads.

DESIGN RULE (applies throughout this file): any field whose value comes
from OCR/document extraction, or from an external bureau report, is
Optional with a null-safe default. "We couldn't read/obtain this" is a
normal, expected state for this pipeline — not a validation failure. The
only fields that stay strictly required are structural/identity fields
the request literally cannot be processed without (application_id,
applicant_type, submission_timestamp, the presence of each top-level
section).

A null at the schema layer does NOT silently vanish. Every consumer this
value reaches (src/adapters/application_mapper.py,
src/adapters/history_mapper.py) has been updated alongside this file to
turn "missing" into an explicit NaN (numeric) or an omitted dict key
(categorical) rather than crashing or silently treating null as 0/False.
See those files' docstrings for the None-handling contract. NaN then
flows into build_features.transform_serve_row() and on into
XGBoost/LightGBM, both of which handle missing feature values natively.

TRADE-OFF WORTH KNOWING: nulling a field the model leans on heavily
(age, income, requested amount, I-Score) is not equivalent to the
"missing" the model saw during training. Kaggle's application_train.csv
is essentially never missing DAYS_BIRTH/AMT_INCOME_TOTAL/AMT_CREDIT, so
XGBoost/LightGBM's learned "default direction" for a missing value on
those specific features is underdetermined — the model will still
produce *a* score, but confidence in it should be lower than for a
field (like EXT_SOURCE) that was genuinely missing at training time too.
This schema change does not fix that; it only stops a missing OCR field
from crashing the request. Treat a decision built on several nulled
core fields as lower-confidence, not equivalent to a fully-populated one.
"""
from datetime import datetime
from typing import Optional, Union

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Shared building blocks
# ------------------------------------------------------------------
class FieldWithConfidence(BaseModel):
    """value is `object` (not a narrower type) specifically so None passes
    validation here — "OCR ran but the field came back empty" — for the
    case where the extractor still emits the wrapper. The more common case
    ("OCR never produced this field at all") is handled by making the
    *containing* attribute Optional[FieldWithConfidence] = None throughout
    this file, not by anything in this class."""
    value: object
    confidence: float


class DocumentEntry(BaseModel):
    document_type: str
    overall_quality_score: float
    is_tampered_suspected: bool
    extracted_fields: dict


class NationalIDFields(BaseModel):
    full_name: Optional[FieldWithConfidence] = None
    national_id: Optional[FieldWithConfidence] = None
    date_of_birth: Optional[FieldWithConfidence] = None
    age_years: Optional[FieldWithConfidence] = None
    gender: Optional[FieldWithConfidence] = None
    governorate: Optional[FieldWithConfidence] = None
    address: Optional[FieldWithConfidence] = None
    profession_on_id: Optional[FieldWithConfidence] = None
    marital_status_on_id: Optional[FieldWithConfidence] = None
    id_expiry_date: Optional[FieldWithConfidence] = None


class SalaryCertificateFields(BaseModel):
    employer_name: Optional[FieldWithConfidence] = None
    employer_sector: Optional[FieldWithConfidence] = None
    job_title: Optional[FieldWithConfidence] = None
    declared_net_salary: Optional[FieldWithConfidence] = None
    declared_gross_salary: Optional[FieldWithConfidence] = None
    employment_date: Optional[FieldWithConfidence] = None
    employment_tenure_years: Optional[FieldWithConfidence] = None
    issue_date: Optional[FieldWithConfidence] = None


class BankStatementFields(BaseModel):
    """Not consumed by application_mapper.py today (see docs) — every
    field here is optional with zero downstream model impact either way."""
    bank_name: Optional[FieldWithConfidence] = None
    account_number: Optional[FieldWithConfidence] = None
    # BUG FIX: real OCR payloads have sent a bare int here (e.g. 5) instead
    # of the {value, confidence} wrapper, which 422'd against the old
    # FieldWithConfidence-only type. Accept either shape now.
    statement_period_months: Optional[Union[int, FieldWithConfidence]] = None
    avg_monthly_net_inflow: Optional[FieldWithConfidence] = None
    avg_monthly_balance: Optional[FieldWithConfidence] = None
    min_monthly_balance: Optional[FieldWithConfidence] = None
    max_monthly_balance: Optional[FieldWithConfidence] = None
    balance_volatility_std: Optional[FieldWithConfidence] = None
    overdraft_frequency: Optional[FieldWithConfidence] = None
    returned_cheques_count: Optional[FieldWithConfidence] = None
    income_regularity_score: Optional[FieldWithConfidence] = None


class BureauFacility(BaseModel):
    """One entry in iscore_report_fields.bureau_facilities[]. Every field
    optional: a real bureau report on a clean/performing facility commonly
    omits or nulls fields that only apply to delinquent ones (overdue_days,
    overdue_amount), and lender_name has been observed null on some records.
    Numeric fields default to 0/0.0 — semantically correct for a performing
    facility with no overdue history — rather than None, so
    history_mapper.py's arithmetic never has to special-case a null here.
    """
    lender_name: Optional[str] = None
    facility_type: Optional[str] = None
    granted_amount: Optional[float] = 0.0
    outstanding_amount: Optional[float] = 0.0
    installment_amount: Optional[float] = 0.0
    status: Optional[str] = None
    overdue_days: Optional[int] = 0
    overdue_amount: Optional[float] = 0.0
    start_date: Optional[str] = None
    legal_action_flag: Optional[bool] = False


class IScoreReportFields(BaseModel):
    is_available: bool
    credit_score: Optional[FieldWithConfidence] = None
    score_tier: Optional[FieldWithConfidence] = None
    score_date: Optional[FieldWithConfidence] = None
    total_active_loans_limit: Optional[FieldWithConfidence] = None
    total_outstanding_balance: Optional[FieldWithConfidence] = None
    total_overdue_amount: Optional[FieldWithConfidence] = None
    max_days_past_due: Optional[FieldWithConfidence] = None
    active_credit_cards_count: Optional[FieldWithConfidence] = None
    total_credit_card_utilization: Optional[FieldWithConfidence] = None
    bureau_facilities: list[BureauFacility] = Field(default_factory=list)


class FormData(BaseModel):
    """Unlike the OCR-sourced sections above, this is normally typed
    directly by the applicant/branch officer — but a partially-submitted
    form (draft, multi-step wizard, etc.) is just as real a "missing data"
    case, so the same null-safe treatment applies. requested_amount is the
    single most model-critical field in this entire payload (-> AMT_CREDIT)
    — see the module docstring's trade-off note before relying on a score
    computed with this null."""
    requested_amount: Optional[float] = None
    tenure_months: Optional[int] = None
    requested_annuity: Optional[float] = None
    loan_purpose: Optional[str] = None
    goods_price: Optional[float] = None
    family_status: Optional[str] = None
    children_count: Optional[int] = None
    family_members_count: Optional[int] = None
    housing_type: Optional[str] = None
    education_type: Optional[str] = None
    owns_car: Optional[bool] = None
    owns_realty: Optional[bool] = None
    branch_id: Optional[int] = None


class ConsistencyChecks(BaseModel):
    """Not consumed by the scoring model today (see docs) — every field
    optional with zero downstream model impact."""
    income_mismatch_ratio: Optional[float] = None
    employer_name_match: Optional[bool] = None
    employer_match_similarity_score: Optional[float] = None
    national_id_match_across_documents: Optional[bool] = None
    iscore_report_age_days: Optional[int] = None
    document_tampering_flag: Optional[bool] = None
    running_balance_math_valid: Optional[bool] = None
    fraud_risk_level: Optional[str] = None
    triggered_fraud_rules: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------
# New-to-bank: internal_history is present but structurally empty
# ------------------------------------------------------------------
class NewCustomerAggregatedMetrics(BaseModel):
    prev_app_count: int = 0
    prev_approved_count: int = 0
    prev_refused_count: int = 0
    prev_approved_ratio: float = 0.0
    prev_avg_credit: float = 0.0
    inst_payment_count: int = 0
    inst_late_count: int = 0
    inst_severe_late_count: int = 0
    inst_late_ratio: float = 0.0
    inst_avg_days_late: float = 0.0
    inst_max_days_late: int = 0
    pos_record_count: int = 0
    pos_avg_dpd: float = 0.0
    cc_avg_balance: float = 0.0
    cc_balance_to_limit_ratio: float = 0.0


class NewCustomerInternalHistory(BaseModel):
    internal_history_missing: int = 1
    customer_id: Optional[str] = None
    kyc_status: str = "not_applicable"
    customer_since_date: Optional[str] = None
    relationship_tenure_months: int = 0
    bank_accounts: list = Field(default_factory=list)
    previous_bank_loans: list = Field(default_factory=list)
    aggregated_metrics: NewCustomerAggregatedMetrics = Field(default_factory=NewCustomerAggregatedMetrics)


class NewCustomerApplicationPayload(BaseModel):
    application_id: str
    applicant_type: str
    submission_timestamp: datetime
    is_returning_customer: bool = False
    documents: list[DocumentEntry]
    national_id_fields: NationalIDFields
    salary_certificate_fields: SalaryCertificateFields
    bank_statement_fields: BankStatementFields
    iscore_report_fields: IScoreReportFields
    form_data: FormData
    internal_history: NewCustomerInternalHistory
    consistency_checks: ConsistencyChecks
    excluded_features_note: Optional[str] = None
    # Hook for a future fraud-engine income haircut (see README's Phase 2
    # notes). NOT read by src/adapters/application_mapper.py or the
    # current model today — accepted and validated, otherwise ignored.
    risk_adjusted_salary: Optional[float] = None


# ------------------------------------------------------------------
# Returning customer: internal_history carries real per-account/per-loan
# records that history_mapper.py aggregates into PREV_*/INST_*/BUREAU_* features.
# ------------------------------------------------------------------
class BankAccount(BaseModel):
    account_id: str
    account_type: str
    currency: str
    current_balance: float
    open_date: str
    status: str


class PreviousBankLoan(BaseModel):
    """One of the applicant's own past loans with this bank. Left required
    for now — out of scope for this pass (only FieldWithConfidence,
    SalaryCertificateFields, NationalIDFields, IScoreReportFields,
    BankStatementFields, BureauFacility, FormData, InternalHistory, and
    ConsistencyChecks were addressed). If real internal loan-history
    records turn out to have gaps the way bureau_facilities did, apply the
    same null-safe pattern here and update history_mapper.py's
    _prev_and_inst_features() to match."""
    loan_id: str
    loan_type: str
    principal_amount: float
    interest_rate: float
    tenure_months: int
    status: str
    start_date: str
    installments_count: int
    paid_on_time_count: int
    late_payments_count: int
    severe_late_count: int
    max_days_past_due: int


class ReturningCustomerAggregatedMetrics(NewCustomerAggregatedMetrics):
    pass


class ReturningCustomerInternalHistory(BaseModel):
    """customer_id/kyc_status/customer_since_date/relationship_tenure_months
    made optional per the same null-safe rule — none of them are actually
    read by history_mapper.py (only previous_bank_loans and
    aggregated_metrics are), so relaxing them here has zero effect on
    scoring and only prevents an avoidable 422."""
    internal_history_missing: int = 0
    customer_id: Optional[str] = None
    kyc_status: Optional[str] = None
    customer_since_date: Optional[str] = None
    relationship_tenure_months: Optional[int] = None
    bank_accounts: list[BankAccount] = Field(default_factory=list)
    previous_bank_loans: list[PreviousBankLoan] = Field(default_factory=list)
    aggregated_metrics: ReturningCustomerAggregatedMetrics = Field(
        default_factory=ReturningCustomerAggregatedMetrics
    )


class ReturningCustomerApplicationPayload(BaseModel):
    application_id: str
    applicant_type: str
    submission_timestamp: datetime
    is_returning_customer: bool = True
    documents: list[DocumentEntry]
    national_id_fields: NationalIDFields
    salary_certificate_fields: SalaryCertificateFields
    bank_statement_fields: BankStatementFields
    iscore_report_fields: IScoreReportFields
    form_data: FormData
    internal_history: ReturningCustomerInternalHistory
    consistency_checks: ConsistencyChecks
    excluded_features_note: Optional[str] = None
    risk_adjusted_salary: Optional[float] = None


# ------------------------------------------------------------------
# Shared response schema (both endpoints return the same shape)
# ------------------------------------------------------------------
class CreditDecisionResponse(BaseModel):
    application_id: str
    credit_score: int
    default_probability: float
    decision: str            # AUTO-APPROVE / MANUAL REVIEW / AUTO-REJECT
    risk_tier: str
    reason_codes: list[str]
    model_version: str