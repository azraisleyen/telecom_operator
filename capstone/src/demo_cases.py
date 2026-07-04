import json
from pathlib import Path
import numpy as np
import pandas as pd
from .retrieval import build_complaint_query
from .response_generation import build_template_response
from .preprocessing import save_table


def _json_safe(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False: return None
    return value


def _make_json_safe(obj):
    if isinstance(obj, dict): return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_make_json_safe(v) for v in obj]
    if isinstance(obj, tuple): return [_make_json_safe(v) for v in obj]
    return _json_safe(obj)


def _pick(df, mask, used, used_categories, prefer_low_priority=False, sort="business_risk_score"):
    available = df.loc[~df.index.isin(used)].copy()
    if mask is not None:
        mask = mask.reindex(available.index, fill_value=False)
        cand = available.loc[mask].copy()
        exact = True
    else:
        cand = available.copy(); exact = False
    if cand.empty:
        cand = available.copy(); exact = False
    if cand.empty: return cand, "no_available_row"
    cand["_category_seen"] = cand.problem_category.isin(used_categories).astype(int)
    if prefer_low_priority:
        order = ["_category_seen", "friction_score_v3", "business_risk_score"]
        cand = cand.sort_values(order, ascending=[True, True, True])
    else:
        cand = cand.sort_values(["_category_seen", sort], ascending=[True, False])
    row = cand.head(1).drop(columns=["_category_seen"], errors="ignore")
    return row, "exact_match" if exact else "transparent_fallback_highest_available_diverse_row"


def build_demo_cases(df, index, outdir):
    specs = [
        {"case_name":"Billing escalation / plan mismatch or refund / billing risk","business_use_case":"Risk-aware complaint triage + high-value protection","mask":(df.problem_category=="billing") & (df.has_billing_dispute==1)},
        {"case_name":"Technical outage + retention risk","business_use_case":"Retention risk prioritization + technical service recovery","mask":(df.problem_category.isin(["data","coverage","connectivity","speed","voice","equipment","installation"])) & (df.has_technical_outage==1) & (df.has_churn_risk==1)},
        {"case_name":"Misunderstood customer / high friction / potential mismatch","business_use_case":"Friction-aware agent copilot","mask":(df.friction_level_v3=="high") & ((df.potential_agent_mismatch==1) | (df.agent_clarification_count_v3>0) | (df.customer_repetition_count_v3>0))},
        {"case_name":"High-value at-risk customer","business_use_case":"High-value customer protection","mask":(df.high_value_customer==1) & (df.priority_level.isin(["P1 High","P2 Medium"]) | (df.business_risk_level=="high"))},
        {"case_name":"KB-grounded standard support / self-service","business_use_case":"LLM-powered RAG resolution copilot","mask":(df.priority_level.isin(["P2 Medium","P3 Low"])) & (df.friction_level_v3=="low"), "prefer_low_priority":True},
    ]
    picks=[]; used=set(); used_categories=set(); used_queues=set()
    for spec in specs:
        row, reason = _pick(df, spec["mask"], used, used_categories, spec.get("prefer_low_priority", False))
        if row.empty: continue
        r=row.iloc[0]
        used.add(row.index[0]); used_categories.add(r.get("problem_category")); used_queues.add(r.get("recommended_queue"))
        docs=index.retrieve(build_complaint_query(r),3); resp=build_template_response(r,docs)
        if spec.get("prefer_low_priority") and docs and docs[0].get("score",0) < 0.05: reason += "; low_rag_confidence"
        signals=[name for name, condition in [("legal",r.get("has_legal_risk",0)),("churn_candidate",r.get("has_churn_risk",0)),("billing_dispute",r.get("has_billing_dispute",0)),("technical_outage",r.get("has_technical_outage",0)),("potential_agent_mismatch",r.get("potential_agent_mismatch",0)),("bill_shock",r.get("bill_shock_flag",0)),("high_value",r.get("high_value_customer",0)),("repeat_next_2m",r.get("repeat_complaint_next_2m",0))] if int(condition)==1]
        case={"case_name":spec["case_name"],"example_name":spec["case_name"],"business_use_case":spec["business_use_case"],"fallback_reason":reason,"customer_id":r.get("customer_id"),"billing_year_month":r.get("billing_year_month"),"lifecycle_status_v2":r.get("lifecycle_status_v2"),"customer_segment":r.get("customer_segment"),"detected_category":r.get("problem_category"),"detected_subcategory":r.get("problem_subcategory"),"probable_intent":r.get("problem_subcategory"),"intent_confidence":None,"friction_score_v3":r.get("friction_score_v3"),"friction_level_v3":r.get("friction_level_v3"),"business_risk_score":r.get("business_risk_score"),"business_risk_level":r.get("business_risk_level"),"priority_level":r.get("priority_level"),"risk_signals":signals,"recommended_queue":r.get("recommended_queue"),"recommended_business_action":r.get("recommended_business_action"),"retrieved_documents":docs,"why_this_case_matters_for_retention":"Bu temsilî örnek, kesin churn iddiası üretmeden yaşam döngüsü, değer, risk ve deneyim sinyalleriyle proaktif elde tutma önceliklendirmesini gösterir.",**resp}
        picks.append(_make_json_safe(case))
    demo_dir=Path(outdir)/"demo_cases"; final_dir=Path(outdir)/"final"; demo_dir.mkdir(parents=True,exist_ok=True); final_dir.mkdir(parents=True,exist_ok=True)
    (demo_dir/"final_demo_cases.json").write_text(json.dumps(picks,ensure_ascii=False,indent=2),encoding="utf-8")
    save_table(pd.DataFrame(picks),demo_dir/"final_demo_cases.csv")
    md=["# Director Demo Examples\n", "Bu kayıtlar bireysel müşteri örneğidir; ana ürün iş kullanım senaryolarını temsil eder, sistem yalnızca bu beş kayıt için tasarlanmamıştır.\n"]
    for c in picks:
        source_ids=", ".join(d["document_id"] for d in c.get("retrieved_documents",[]))
        md += [f"\n## {c['case_name']}",f"- Business use case: {c['business_use_case']}",f"- Customer: {c['customer_id']} / {c['billing_year_month']}",f"- Priority: {c['priority_level']} | Queue: {c['recommended_queue']}",f"- Action: {c['recommended_business_action']}",f"- Fallback status: {c['fallback_reason']}",f"- KB Sources: {source_ids}",f"- Suggested reply: {c['suggested_agent_reply']}",f"- Retention relevance: {c['why_this_case_matters_for_retention']}"]
    (final_dir/"director_demo_cases.md").write_text("\n".join(md),encoding="utf-8")
    return picks
