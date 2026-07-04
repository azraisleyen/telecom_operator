import json
from pathlib import Path

import numpy as np
import pandas as pd

from .retrieval import build_complaint_query
from .response_generation import build_template_response
from .preprocessing import save_table


def _json_safe(value):
    """Convert pandas/numpy scalar values into JSON-serializable Python types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    return value


def _make_json_safe(obj):
    """Recursively convert nested structures to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_make_json_safe(v) for v in obj]
    return _json_safe(obj)


def _pick(df: pd.DataFrame, mask: pd.Series, sort: str = "business_risk_score") -> pd.DataFrame:
    """
    Pick the highest-risk row matching the mask.

    The mask may come from the original full DataFrame, while df may be a filtered
    subset. Reindexing prevents the 'Boolean Series key will be reindexed' warning.
    """
    if isinstance(mask, pd.Series):
        mask = mask.reindex(df.index, fill_value=False)
    x = df.loc[mask].sort_values(sort, ascending=False)
    return x.head(1)


def build_demo_cases(df, index, outdir):
    picks = []

    specs = [
        (
            "Billing escalation / legal or highest billing risk",
            (df.problem_category == "billing")
            & (df.has_billing_dispute == 1)
            & (df.priority_level.isin(["P0 Critical", "P1 High"])),
        ),
        (
            "Technical outage + churn risk",
            (df.problem_category.isin(["data", "coverage", "connectivity", "speed"]))
            & (df.has_technical_outage == 1)
            & (df.has_churn_risk == 1),
        ),
        (
            "Misunderstood customer / high friction",
            (df.friction_level_v3 == "high")
            & (
                (df.customer_repetition_count_v3 > 0)
                | (df.unresolved_problem_count_v3 > 0)
                | (df.customer_frustration_count_v3 > 0)
                | (df.agent_clarification_count_v3 > 0)
                | (df.potential_agent_mismatch == 1)
            ),
        ),
        (
            "High-value at-risk customer",
            (df.high_value_customer == 1)
            & (
                (df.priority_level.isin(["P0 Critical", "P1 High"]))
                | (df.business_risk_level == "high")
            ),
        ),
        (
            "KB-grounded standard support / self-service",
            df.priority_level.isin(["P2 Medium", "P3 Low"]),
        ),
    ]

    used = set()

    for name, mask in specs:
        available = df.loc[~df.index.isin(used)]
        row = _pick(available, mask)

        # Fallback: if the exact case criteria cannot be found, select the highest-risk unused real row.
        if row.empty:
            row = available.sort_values("business_risk_score", ascending=False).head(1)

        if row.empty:
            continue

        used.add(row.index[0])
        r = row.iloc[0]

        docs = index.retrieve(build_complaint_query(r), 3)
        resp = build_template_response(r, docs)

        signals = [
            signal_name
            for signal_name, condition in [
                ("legal", r.get("has_legal_risk", 0)),
                ("churn_candidate", r.get("has_churn_risk", 0)),
                ("billing_dispute", r.get("has_billing_dispute", 0)),
                ("technical_outage", r.get("has_technical_outage", 0)),
                ("potential_agent_mismatch", r.get("potential_agent_mismatch", 0)),
                ("bill_shock", r.get("bill_shock_flag", 0)),
                ("high_value", r.get("high_value_customer", 0)),
            ]
            if int(condition) == 1
        ]

        case = {
            "case_name": name,
            "customer_id": r.get("customer_id"),
            "billing_year_month": r.get("billing_year_month"),
            "lifecycle_status_v2": r.get("lifecycle_status_v2"),
            "customer_segment": r.get("customer_segment"),
            "detected_category": r.get("problem_category"),
            "detected_subcategory": r.get("problem_subcategory"),
            "probable_intent": r.get("problem_subcategory"),
            "intent_confidence": 1.0,
            "friction_score_v3": r.get("friction_score_v3"),
            "friction_level_v3": r.get("friction_level_v3"),
            "business_risk_score": r.get("business_risk_score"),
            "business_risk_level": r.get("business_risk_level"),
            "priority_level": r.get("priority_level"),
            "risk_signals": signals,
            "recommended_queue": r.get("recommended_queue"),
            "retrieved_documents": docs,
            "why_this_case_matters_for_retention": (
                "Resmi churn etiketi yoktur; bu vaka yaşam döngüsü, değer ve deneyim "
                "sinyalleriyle proaktif elde tutma açısından izlenmelidir."
            ),
            **resp,
        }

        picks.append(_make_json_safe(case))

    demo_dir = Path(outdir) / "demo_cases"
    final_dir = Path(outdir) / "final"
    demo_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    (demo_dir / "final_demo_cases.json").write_text(
        json.dumps(picks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_table(pd.DataFrame(picks), demo_dir / "final_demo_cases.csv")

    md = ["# Director Demo Cases\n"]

    for c in picks:
        source_ids = ", ".join(d["document_id"] for d in c.get("retrieved_documents", []))
        md += [
            f"\n## {c['case_name']}",
            f"- Customer: {c['customer_id']} / {c['billing_year_month']}",
            f"- Priority: {c['priority_level']} | Queue: {c['recommended_queue']}",
            f"- KB Sources: {source_ids}",
            f"- Suggested reply: {c['suggested_agent_reply']}",
            f"- Retention relevance: {c['why_this_case_matters_for_retention']}",
        ]

    (final_dir / "director_demo_cases.md").write_text("\n".join(md), encoding="utf-8")

    return picks