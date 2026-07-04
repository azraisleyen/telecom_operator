import pandas as pd
from .preprocessing import save_table


def add_risk_priority(df):
    o = df.copy()
    o["repeat_complaint"] = o.get("repeat_complaint_next_2m", o.get("repeat_complaint", 0)).fillna(0).astype(int)
    o["risk_component_legal"] = 40 * o.has_legal_risk
    o["risk_component_churn"] = 30 * o.has_churn_risk
    o["risk_component_high_frustration"] = 20 * o.high_frustration_flag
    o["risk_component_high_friction"] = 15 * (o.friction_level_v3 == "high").astype(int)
    o["risk_component_repeat"] = 15 * o.repeat_complaint
    o["risk_component_billing"] = 15 * o.has_billing_dispute
    o["risk_component_technical"] = 10 * o.has_technical_outage
    o["risk_component_high_value"] = 10 * o.high_value_customer
    o["risk_component_bill_shock"] = 10 * o.bill_shock_flag
    component_cols = [c for c in o.columns if c.startswith("risk_component_")]
    o["business_risk_score"] = o[component_cols].sum(axis=1).astype(int)
    o["business_risk_level"] = pd.cut(o.business_risk_score, [-1, 19, 49, 999], labels=["low", "medium", "high"]).astype(str)
    p0 = (o.has_legal_risk == 1) | (o.business_risk_score >= 90) | ((o.has_churn_risk == 1) & (o.friction_level_v3 == "high") & (o.high_value_customer == 1))
    p1 = (o.has_churn_risk == 1) | (o.business_risk_level == "high")
    p2 = o.business_risk_level == "medium"
    o["priority_level"] = "P3 Low"
    o.loc[p2, "priority_level"] = "P2 Medium"
    o.loc[p1, "priority_level"] = "P1 High"
    o.loc[p0, "priority_level"] = "P0 Critical"
    o["risk_score_band"] = pd.cut(o.business_risk_score, [-1, 19, 39, 59, 79, 999], labels=["0-19", "20-39", "40-59", "60-79", "80+"]).astype(str)
    return o


def export_risk_tables(df, outdir):
    component_cols = [c for c in df.columns if c.startswith("risk_component_")]
    save_table(df.business_risk_level.value_counts().rename_axis("business_risk_level").reset_index(name="complaint_count"), f"{outdir}/tables/business_risk_level_summary.csv")
    save_table(df.priority_level.value_counts().rename_axis("priority_level").reset_index(name="complaint_count"), f"{outdir}/tables/priority_distribution.csv")
    save_table(df.risk_score_band.value_counts().rename_axis("risk_score_band").reset_index(name="complaint_count"), f"{outdir}/tables/risk_score_band_distribution.csv")
    save_table(pd.DataFrame([{"component": c, "total_points": df[c].sum(), "active_rows": int((df[c] > 0).sum()), "active_rate": float((df[c] > 0).mean()) if len(df) else 0.0} for c in component_cols]), f"{outdir}/tables/risk_component_summary.csv")
    save_table(pd.crosstab(df.problem_category, df.priority_level).reset_index(), f"{outdir}/tables/priority_by_category.csv")
    if "recommended_queue" in df:
        save_table(df.recommended_queue.value_counts().rename_axis("recommended_queue").reset_index(name="complaint_count"), f"{outdir}/tables/routing_distribution.csv")
    hv = df[(df.high_value_customer == 1) & (df.business_risk_level == "high")]
    save_table(hv.head(10000), f"{outdir}/tables/high_value_at_risk.csv")
    save_table(pd.DataFrame([{"high_value_at_risk_customers": hv.customer_id.nunique(), "historical_revenue": hv.total_revenue.sum()}]), f"{outdir}/tables/high_value_at_risk_summary.csv")
    save_table(pd.crosstab(df[df.high_value_customer == 1].priority_level, columns="complaint_count").reset_index(), f"{outdir}/tables/high_value_priority_distribution.csv")
    save_table(df[df.high_value_customer == 1].groupby("priority_level", dropna=False).agg(customer_count=("customer_id", "nunique"), historical_revenue=("total_revenue", "sum")).reset_index(), f"{outdir}/tables/high_value_revenue_by_priority.csv")
