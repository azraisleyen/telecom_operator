import pandas as pd
from pathlib import Path
from .preprocessing import save_table


def _pct(x): return f"{100*float(x):.1f}%"


def export_final_evaluation(customer, spending, complaints, profile, bill, rag_summary, outdir):
    tables=Path(outdir)/'tables'; final=Path(outdir)/'final'; tables.mkdir(parents=True,exist_ok=True); final.mkdir(parents=True,exist_ok=True)
    save_table(pd.DataFrame([{'customer_rows':len(customer),'spending_rows':len(spending),'complaint_rows':len(complaints)}]),tables/'data_scale_summary.csv')
    save_table(profile.lifecycle_status_v2.value_counts().rename_axis('lifecycle_status_v2').reset_index(name='customer_count'),tables/'lifecycle_status_v2_summary.csv')
    bill_shock_rate=float(bill.bill_shock_flag.mean()) if len(bill) else 0.0
    save_table(pd.DataFrame([{'bill_shock_rate':bill_shock_rate,'same_month_billing_complaint_lift':'context feature; causal claim is not made'}]),tables/'bill_shock_summary.csv')
    save_table(complaints.friction_level_v3.value_counts().rename_axis('friction_level_v3').reset_index(name='complaint_count'),tables/'friction_v3_summary.csv')
    save_table(complaints.business_risk_level.value_counts().rename_axis('business_risk_level').reset_index(name='complaint_count'),tables/'business_risk_summary.csv')
    if rag_summary is not None: save_table(rag_summary,tables/'rag_retrieval_summary.csv')
    largest_cat = complaints.problem_category.value_counts().index[0] if len(complaints) else 'N/A'
    high_friction_rate = float((complaints.friction_level_v3=='high').mean()) if len(complaints) else 0.0
    high_risk_rate = float((complaints.business_risk_level=='high').mean()) if len(complaints) else 0.0
    priority = complaints.priority_level.value_counts().to_dict() if 'priority_level' in complaints else {}
    risk = complaints.business_risk_level.value_counts().to_dict() if 'business_risk_level' in complaints else {}
    hv = complaints[(complaints.high_value_customer==1)&(complaints.business_risk_level=='high')]
    rag_hit3 = None
    if rag_summary is not None and len(rag_summary):
        for col in ['hit_at_3','category_hit_at_3']:
            if col in rag_summary.columns: rag_hit3=float(rag_summary[col].iloc[0]); break
    rows=[
        {'topic':'Konumlandırma','finding':'Bundlekom, tekil demo kayıtlarından ibaret değildir; retention risk, kaynaklı çözüm, risk triage, friction azaltma ve high-value koruma kullanım senaryolarını birleştiren modüler bir operatör platformudur.'},
        {'topic':'Veri ölçeği','finding':f'{len(customer):,} müşteri, {len(spending):,} harcama satırı ve {len(complaints):,} şikâyet satırı işlendi.'},
        {'topic':'Şikâyet odağı','finding':f'En büyük şikâyet kategorisi: {largest_cat}.'},
        {'topic':'Friction','finding':f'Yüksek friction oranı {_pct(high_friction_rate)}; bu sinyal no-repeat handoff ve agent copilot için kullanılır.'},
        {'topic':'Risk ve öncelik','finding':f'Yüksek iş riski oranı {_pct(high_risk_rate)}. Priority dağılımı: {priority}. Risk dağılımı: {risk}.'},
        {'topic':'Bill shock','finding':f'Bill shock oranı {_pct(bill_shock_rate)}. Bu sinyal bağlamsal risk göstergesidir; nedensellik iddiası yapılmaz.'},
        {'topic':'High-value koruma','finding':f'Yüksek değerli ve yüksek riskli müşteri sayısı {hv.customer_id.nunique():,}; tarihsel gelir toplamı {float(hv.total_revenue.sum()):,.2f}.'},
        {'topic':'RAG','finding':f'TF-IDF/BM25 uyumlu retrieval V1 hazır. Hit@3 durumu: {(_pct(rag_hit3) if rag_hit3 is not None else "ölçüm yok")}.'},
        {'topic':'Sınırlılık','finding':'Veri setinde resmi churn etiketi yoktur; çıktılar kesin churn tahmini değil, no observed billable activity / retention risk önceliklendirmesidir.'},
    ]
    save_table(pd.DataFrame(rows),final/'final_executive_summary.csv')
    md='# Final Executive Summary\n\n'+'\n'.join(f"- **{r['topic']}**: {r['finding']}" for r in rows)+'\n'
    (final/'final_executive_summary.md').write_text(md,encoding='utf-8')
