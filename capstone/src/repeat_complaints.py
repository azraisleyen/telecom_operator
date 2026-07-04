from __future__ import annotations

import pandas as pd

from .preprocessing import save_table, ym_to_month_index


def add_repeat_complaint_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add customer-month repeat complaint flags without counting same-month duplicates."""
    out = df.copy()
    out["month_index"] = out["billing_year_month"].map(ym_to_month_index)
    months = (
        out[["customer_id", "billing_year_month", "month_index"]]
        .drop_duplicates()
        .sort_values(["customer_id", "month_index"])
    )
    lookup = set(zip(months["customer_id"], months["month_index"]))
    months["repeat_complaint_next_1m"] = [int((cid, mi + 1) in lookup) for cid, mi in zip(months.customer_id, months.month_index)]
    months["repeat_complaint_next_2m"] = [int(((cid, mi + 1) in lookup) or ((cid, mi + 2) in lookup)) for cid, mi in zip(months.customer_id, months.month_index)]
    out = out.merge(
        months[["customer_id", "billing_year_month", "repeat_complaint_next_1m", "repeat_complaint_next_2m"]],
        on=["customer_id", "billing_year_month"],
        how="left",
    )
    out[["repeat_complaint_next_1m", "repeat_complaint_next_2m"]] = out[["repeat_complaint_next_1m", "repeat_complaint_next_2m"]].fillna(0).astype(int)
    out["repeat_complaint"] = out["repeat_complaint_next_2m"]
    return out


def export_repeat_tables(df: pd.DataFrame, outdir: str) -> None:
    summary = pd.DataFrame([
        {
            "complaint_rows": len(df),
            "repeat_next_1m_rate": float(df["repeat_complaint_next_1m"].mean()) if len(df) else 0.0,
            "repeat_next_2m_rate": float(df["repeat_complaint_next_2m"].mean()) if len(df) else 0.0,
            "repeat_next_1m_rows": int(df["repeat_complaint_next_1m"].sum()) if len(df) else 0,
            "repeat_next_2m_rows": int(df["repeat_complaint_next_2m"].sum()) if len(df) else 0,
        }
    ])
    by_cat = (
        df.groupby("problem_category", dropna=False)
        .agg(
            complaint_count=("customer_id", "size"),
            repeat_next_1m_rate=("repeat_complaint_next_1m", "mean"),
            repeat_next_2m_rate=("repeat_complaint_next_2m", "mean"),
        )
        .reset_index()
    )
    save_table(summary, f"{outdir}/tables/repeat_complaint_summary.csv")
    save_table(by_cat, f"{outdir}/tables/repeat_complaint_by_category.csv")
