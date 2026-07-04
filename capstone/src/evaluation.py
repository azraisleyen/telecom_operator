import pandas as pd
from pathlib import Path
from .preprocessing import save_table
def export_final_evaluation(customer,spending,complaints,profile,bill,rag_summary,outdir):
    tables=Path(outdir)/'tables'; final=Path(outdir)/'final'; tables.mkdir(parents=True,exist_ok=True); final.mkdir(parents=True,exist_ok=True)
    save_table(pd.DataFrame([{'customer_rows':len(customer),'spending_rows':len(spending),'complaint_rows':len(complaints)}]),tables/'data_scale_summary.csv')
    save_table(profile.lifecycle_status_v2.value_counts().rename_axis('lifecycle_status_v2').reset_index(name='customer_count'),tables/'lifecycle_status_v2_summary.csv')
    save_table(pd.DataFrame([{'bill_shock_rate':bill.bill_shock_flag.mean(),'same_month_billing_complaint_lift':'context feature; prior analysis ~1.00x'}]),tables/'bill_shock_summary.csv')
    save_table(complaints.friction_level_v3.value_counts().rename_axis('friction_level_v3').reset_index(name='complaint_count'),tables/'friction_v3_summary.csv')
    save_table(complaints.business_risk_level.value_counts().rename_axis('business_risk_level').reset_index(name='complaint_count'),tables/'business_risk_summary.csv')
    if rag_summary is not None: save_table(rag_summary,tables/'rag_retrieval_summary.csv')
    rows=[{'topic':'positioning','finding':'Bundlekom is an explainable customer risk and resolution copilot, not an uncontrolled chatbot.'},{'topic':'billing','finding':'Billing is the largest complaint area and same-category repeat is operationally important.'},{'topic':'bill_shock','finding':'Bill shock exists but is treated as context; prior analysis did not show meaningful same-month billing complaint lift.'},{'topic':'friction','finding':'Friction v3 supports conversation quality, escalation, and agent-copilot routing; it is not a standalone churn model.'},{'topic':'risk','finding':'Business risk score combines legal, retention, billing, technical, value, shock, repeat, and friction signals for triage.'},{'topic':'rag','finding':'TF-IDF retrieval is an inspectable V1 baseline and returns document_id sources for grounded responses.'}]
    save_table(pd.DataFrame(rows),final/'final_executive_summary.csv')
    md='# Final Executive Summary\n\n'+'\n'.join(f"- **{r['topic']}**: {r['finding']}" for r in rows)+'\n\nNo official churn label is claimed; outputs use lifecycle risk and no observed billable activity language.'
    (final/'final_executive_summary.md').write_text(md,encoding='utf-8')
