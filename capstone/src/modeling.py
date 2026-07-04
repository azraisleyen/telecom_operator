from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocessing import ym_to_month_index, save_table

TARGET = "no_billable_activity_next_2m"
CATEGORICAL = ["dominant_problem_category", "dominant_problem_subcategory", "lifecycle_status_v2"]
NUMERIC = [
    "complaint_count", "unique_problem_category_count", "max_friction_score_v3", "max_business_risk_score", "max_priority_rank",
    "any_has_churn_risk", "any_has_billing_dispute", "any_has_technical_outage", "any_has_legal_risk", "any_potential_agent_mismatch",
    "any_bill_shock_flag", "any_repeat_complaint_next_2m", "high_value_customer", "total_revenue", "avg_bill", "avg_data_usage",
    "months_active", "active_month_ratio", "months_since_last_spending", "inactive_gap_count", "reactivation_count",
    "previous_bill_amount", "rolling_3m_avg_bill", "bill_change_abs", "bill_change_pct",
]
FEATURES = NUMERIC + CATEGORICAL


def _mode(s):
    m = s.dropna().astype(str).mode()
    return m.iloc[0] if len(m) else "unknown"


def build_customer_month_table(complaints: pd.DataFrame) -> pd.DataFrame:
    c = complaints.copy()
    c["month_index"] = c["billing_year_month"].map(ym_to_month_index)
    priority_map = {"P3 Low": 1, "P2 Medium": 2, "P1 High": 3, "P0 Critical": 4}
    c["priority_rank"] = c.get("priority_level", "P3 Low").map(priority_map).fillna(0)
    agg = c.groupby(["customer_id", "billing_year_month"], dropna=False).agg(
        month_index=("month_index", "max"),
        complaint_count=("customer_id", "size"),
        unique_problem_category_count=("problem_category", "nunique"),
        dominant_problem_category=("problem_category", _mode),
        dominant_problem_subcategory=("problem_subcategory", _mode),
        max_friction_score_v3=("friction_score_v3", "max"),
        max_business_risk_score=("business_risk_score", "max"),
        max_priority_rank=("priority_rank", "max"),
        any_has_churn_risk=("has_churn_risk", "max"),
        any_has_billing_dispute=("has_billing_dispute", "max"),
        any_has_technical_outage=("has_technical_outage", "max"),
        any_has_legal_risk=("has_legal_risk", "max"),
        any_potential_agent_mismatch=("potential_agent_mismatch", "max"),
        any_bill_shock_flag=("bill_shock_flag", "max"),
        any_repeat_complaint_next_2m=("repeat_complaint_next_2m", "max"),
        high_value_customer=("high_value_customer", "max"),
        total_revenue=("total_revenue", "max"),
        avg_bill=("avg_bill", "max"),
        avg_data_usage=("avg_data_usage", "max"),
        lifecycle_status_v2=("lifecycle_status_v2", _mode),
        months_active=("months_active", "max"),
        active_month_ratio=("active_month_ratio", "max"),
        months_since_last_spending=("months_since_last_spending", "max"),
        inactive_gap_count=("inactive_gap_count", "max"),
        reactivation_count=("reactivation_count", "max"),
        previous_bill_amount=("previous_bill_amount", "max"),
        rolling_3m_avg_bill=("rolling_3m_avg_bill", "max"),
        bill_change_abs=("bill_change_abs", "max"),
        bill_change_pct=("bill_change_pct", "max"),
        no_billable_activity_next_2m=(TARGET, "max"),
    ).reset_index()
    agg[TARGET] = agg[TARGET].fillna(0).astype(int)
    return agg


def valid_time_split(df: pd.DataFrame):
    d = df.copy().sort_values("month_index")
    max_m = d.month_index.max()
    d = d[d.month_index <= max_m - 2]
    months = sorted(d.month_index.unique())
    if len(months) < 3:
        raise ValueError("Not enough valid months for time-aware train/validation/test split after excluding final two months.")
    n = len(months); train_m = set(months[:max(1, int(n * 0.6))]); val_m = set(months[max(1, int(n * 0.6)):max(2, int(n * 0.8))]); test_m = set(months[max(2, int(n * 0.8)):])
    return d[d.month_index.isin(train_m)], d[d.month_index.isin(val_m)], d[d.month_index.isin(test_m)]


def _preprocessor(scale=False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])


def build_models():
    return {
        "logistic_regression": Pipeline([("prep", _preprocessor(scale=True)), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "hist_gradient_boosting": Pipeline([("prep", _preprocessor(scale=False)), ("model", HistGradientBoostingClassifier(class_weight="balanced", random_state=42))]),
        "random_forest": Pipeline([("prep", _preprocessor(scale=False)), ("model", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1))]),
    }


def score_model(model, X):
    if isinstance(model, str) and model == "business_risk_baseline":
        s = X["max_business_risk_score"].fillna(0).astype(float)
        return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else np.zeros(len(s))
    return model.predict_proba(X)[:, 1]


def topk_metrics(y, score, revenue=None, ks=(0.01, 0.05, 0.10)):
    y = np.asarray(y).astype(int); score = np.asarray(score); revenue = np.asarray(revenue if revenue is not None else np.ones(len(y)), dtype=float)
    out = {}; base = y.mean() if len(y) else 0
    order = np.argsort(-score)
    for k in ks:
        n = max(1, int(np.ceil(len(y) * k))) if len(y) else 0
        idx = order[:n]
        prec = y[idx].mean() if n else 0
        recall = y[idx].sum() / y.sum() if y.sum() else 0
        revcap = revenue[idx][y[idx] == 1].sum() / revenue[y == 1].sum() if revenue[y == 1].sum() else 0
        pct = int(k * 100)
        out[f"precision_top_{pct}pct"] = prec
        out[f"recall_top_{pct}pct"] = recall
        out[f"lift_top_{pct}pct"] = prec / base if base else 0
        out[f"revenue_capture_top_{pct}pct"] = revcap
    return out


def evaluate_scores(y, score, split, revenue=None):
    m = topk_metrics(y, score, revenue)
    return {
        "split": split,
        "roc_auc": roc_auc_score(y, score) if len(set(y)) > 1 else np.nan,
        "average_precision_pr_auc": average_precision_score(y, score) if len(set(y)) > 1 else np.nan,
        "brier_score": brier_score_loss(y, score) if len(y) else np.nan,
        **m,
    }


def decile_table(df, score, split):
    d = df[["customer_id", "billing_year_month", TARGET, "total_revenue"]].copy(); d["score"] = score
    d["decile"] = pd.qcut(d["score"].rank(method="first", ascending=False), 10, labels=False, duplicates="drop") + 1
    return d.groupby("decile").agg(split=(TARGET, lambda _: split), customer_months=(TARGET, "size"), target_rate=(TARGET, "mean"), captured_targets=(TARGET, "sum"), captured_revenue=("total_revenue", "sum"), avg_score=("score", "mean")).reset_index()


def calibration_table(df, score, split):
    d = df[[TARGET]].copy(); d["score"] = score
    d["bucket"] = pd.qcut(d["score"].rank(method="first"), 10, labels=False, duplicates="drop") + 1
    return d.groupby("bucket").agg(split=(TARGET, lambda _: split), rows=(TARGET, "size"), avg_predicted_risk=("score", "mean"), observed_rate=(TARGET, "mean")).reset_index()


def predict_retention_risk(input_df, model_path="capstone/models/retention_risk_model.joblib"):
    model = load(model_path)
    probs = score_model(model, input_df[FEATURES])
    pct = pd.Series(probs).rank(pct=True).to_numpy()
    return pd.DataFrame({"risk_probability": probs, "risk_percentile": pct})


def save_model(model, path):
    dump(model, path)
