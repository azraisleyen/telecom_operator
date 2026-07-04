import json, pandas as pd
from pathlib import Path
from .retrieval import build_complaint_query
from .response_generation import build_template_response
from .preprocessing import save_table
def _pick(df,mask,sort='business_risk_score'):
    x=df[mask].sort_values(sort,ascending=False); return x.head(1)
def build_demo_cases(df,index,outdir):
    picks=[]
    specs=[('Billing escalation / legal or highest billing risk',(df.problem_category=='billing')&(df.has_billing_dispute==1)&(df.priority_level.isin(['P0 Critical','P1 High']))),('Technical outage + churn risk',(df.problem_category.isin(['data','coverage','connectivity','speed']))&(df.has_technical_outage==1)&(df.has_churn_risk==1)),('Misunderstood customer / high friction',(df.friction_level_v3=='high')&((df.customer_repetition_count_v3>0)|(df.unresolved_problem_count_v3>0)|(df.customer_frustration_count_v3>0)|(df.agent_clarification_count_v3>0)|(df.potential_agent_mismatch==1))),('High-value at-risk customer',(df.high_value_customer==1)&((df.priority_level.isin(['P0 Critical','P1 High']))|(df.business_risk_level=='high'))),('KB-grounded standard support / self-service',df.priority_level.isin(['P2 Medium','P3 Low']))]
    used=set()
    for name,mask in specs:
        row=_pick(df[~df.index.isin(used)],mask)
        if row.empty: row=df[~df.index.isin(used)].sort_values('business_risk_score',ascending=False).head(1)
        if row.empty: continue
        used.add(row.index[0]); r=row.iloc[0]; docs=index.retrieve(build_complaint_query(r),3); resp=build_template_response(r,docs)
        signals=[s for s,c in [('legal',r.has_legal_risk),('churn_candidate',r.has_churn_risk),('billing_dispute',r.has_billing_dispute),('technical_outage',r.has_technical_outage),('potential_agent_mismatch',r.potential_agent_mismatch),('bill_shock',r.bill_shock_flag),('high_value',r.high_value_customer)] if int(c)==1]
        picks.append({'case_name':name,'customer_id':r.customer_id,'billing_year_month':r.billing_year_month,'lifecycle_status_v2':r.lifecycle_status_v2,'customer_segment':r.customer_segment,'detected_category':r.problem_category,'detected_subcategory':r.problem_subcategory,'probable_intent':r.problem_subcategory,'intent_confidence':1.0,'friction_score_v3':int(r.friction_score_v3),'friction_level_v3':r.friction_level_v3,'business_risk_score':int(r.business_risk_score),'business_risk_level':r.business_risk_level,'priority_level':r.priority_level,'risk_signals':signals,'recommended_queue':r.recommended_queue,'retrieved_documents':docs,'why_this_case_matters_for_retention':'Resmi churn etiketi yoktur; bu vaka yaşam döngüsü, değer ve deneyim sinyalleriyle proaktif elde tutma açısından izlenmelidir.',**resp})
    Path(f'{outdir}/demo_cases').mkdir(parents=True,exist_ok=True); Path(f'{outdir}/final').mkdir(parents=True,exist_ok=True)
    Path(f'{outdir}/demo_cases/final_demo_cases.json').write_text(json.dumps(picks,ensure_ascii=False,indent=2),encoding='utf-8')
    save_table(pd.DataFrame(picks),f'{outdir}/demo_cases/final_demo_cases.csv')
    md=['# Director Demo Cases\n']
    for c in picks: md += [f"\n## {c['case_name']}",f"- Customer: {c['customer_id']} / {c['billing_year_month']}",f"- Priority: {c['priority_level']} | Queue: {c['recommended_queue']}",f"- KB Sources: {', '.join(d['document_id'] for d in c['retrieved_documents'])}",f"- Suggested reply: {c['suggested_agent_reply']}"]
    Path(f'{outdir}/final/director_demo_cases.md').write_text('\n'.join(md),encoding='utf-8')
    return picks
