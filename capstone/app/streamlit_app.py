from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
import streamlit as st
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from capstone.src.data_loader import load_or_build_processed_data
from capstone.src.retrieval import build_reference_index
from capstone.src.llm_rag import generate_rag_response, llm_configured
from capstone.src.response_generation import build_template_response

DATA_DIR=Path('capstone/data'); OUT=Path('capstone/outputs'); TABLES=OUT/'tables'
st.set_page_config(page_title='Bundlekom Operator AI Console', layout='wide')
st.title('Bundlekom Operator AI Console')
st.caption('Modüler telekom müşteri deneyimi, retention risk ve kaynaklı çözüm copilot platformu. Kesin churn tahmini yapmaz.')

def load_csv(name):
    p=TABLES/name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def metric(label, value): st.metric(label, value if value not in [None,''] else 'N/A')

def show_df(df, title):
    st.subheader(title)
    if df.empty: st.info(f'{title} henüz üretilmemiş. İlgili pipeline/script çalıştırılmalı.')
    else: st.dataframe(df, use_container_width=True)

tabs=st.tabs(['Executive Dashboard','Retention Risk ML Model','LLM-Powered RAG Resolution Copilot','Risk & Friction Triage','High-Value Customer Protection','Demo Examples','Evaluation Center'])
with tabs[0]:
    scale=load_csv('data_scale_summary.csv'); cat=load_csv('category_counts.csv'); fr=load_csv('friction_v3_distribution.csv'); risk=load_csv('business_risk_level_summary.csv'); pr=load_csv('priority_distribution.csv'); hv=load_csv('high_value_at_risk_summary.csv'); bill=load_csv('bill_shock_summary.csv'); ml=load_csv('retention_precision_at_k.csv'); rag=load_csv('rag_retrieval_metrics.csv')
    c=st.columns(5)
    with c[0]: metric('Total customers', scale.customer_rows.iloc[0] if not scale.empty else None)
    with c[1]: metric('Total complaints', scale.complaint_rows.iloc[0] if not scale.empty else None)
    with c[2]: metric('Largest category', cat.problem_category.iloc[0] if not cat.empty else None)
    with c[3]:
        hf = fr.loc[fr.friction_level_v3=='high','complaint_count'].sum()/fr.complaint_count.sum() if not fr.empty and fr.complaint_count.sum() else None; metric('High friction rate', f'{hf:.1%}' if hf is not None else None)
    with c[4]: metric('Bill shock rate', f"{float(bill.bill_shock_rate.iloc[0]):.1%}" if not bill.empty else None)
    c=st.columns(5)
    with c[0]: metric('High risk complaints', int(risk.loc[risk.business_risk_level=='high','complaint_count'].sum()) if not risk.empty else None)
    with c[1]: metric('P0/P1 complaints', int(pr.loc[pr.priority_level.isin(['P0 Critical','P1 High']),'complaint_count'].sum()) if not pr.empty else None)
    with c[2]: metric('High-value at-risk customers', int(hv.high_value_at_risk_customers.iloc[0]) if not hv.empty else None)
    with c[3]: metric('ML Precision@Top5', f"{float(ml[(ml.split=='test')&(ml.top_k_percent==5)].precision.iloc[0]):.1%}" if not ml.empty and len(ml[(ml.split=='test')&(ml.top_k_percent==5)]) else None)
    with c[4]: metric('RAG Hit@3', f"{float(rag.hit_at_3.iloc[0]):.1%}" if not rag.empty and 'hit_at_3' in rag else None)
    show_df(pr,'Priority distribution'); show_df(risk,'Business risk distribution')
with tabs[1]:
    st.warning('Bu model kesin churn tahmini değildir; no observed billable activity / retention risk önceliklendirmesi yapar.')
    for f,t in [('retention_model_metrics.csv','Model metrics'),('retention_precision_at_k.csv','Precision / Recall / Lift @ Top K'),('retention_decile_lift.csv','Decile lift'),('retention_revenue_capture.csv','Revenue capture'),('retention_feature_importance.csv','Feature importance'),('retention_top_risk_customers.csv','Top risk customers')]: show_df(load_csv(f),t)
with tabs[2]:
    st.write('LLM disabled; using template fallback.' if not llm_configured() else 'LLM enabled; source-constrained generation will be attempted.')
    complaint=st.text_area('Complaint text', 'Faturam beklediğimden yüksek geldi, paketimle uyumlu olmadığını düşünüyorum.')
    category=st.text_input('Optional category','billing'); sub=st.text_input('Optional subcategory',''); k=st.slider('Top-k',1,5,3)
    if st.button('Generate source-grounded response'):
        try:
            data=load_or_build_processed_data(DATA_DIR, rebuild=False); idx=build_reference_index(data['reference']); row={'problem_category':category,'problem_subcategory':sub,'first_customer_message':complaint,'customer_text':complaint}; out=generate_rag_response(idx,row,k)
            st.dataframe(pd.DataFrame(out['retrieved_documents']), use_container_width=True); st.success(out['generated_agent_reply']); st.write('Used document IDs:', out['used_document_ids']); st.write('Retrieval confidence:', out['retrieval_confidence'])
            if out['fallback_used']: st.warning('Fallback used: source evidence may be insufficient or LLM is disabled.')
        except Exception as e: st.error(f'RAG için processed reference verisi gerekli: {e}')
with tabs[3]:
    df=load_csv('rag_eval_sample.csv')
    feats=Path('capstone/data/processed/complaints_final_features.parquet')
    if feats.exists(): df=pd.read_parquet(feats).head(1000)
    if df.empty: st.info('Triage verisi için final pipeline çalıştırılmalı.')
    else:
        i=st.number_input('Sample row',0,len(df)-1,0); r=df.iloc[int(i)]
        st.json({k:str(r.get(k,'')) for k in ['customer_id','billing_year_month','lifecycle_status_v2','customer_segment','problem_category','problem_subcategory','friction_score_v3','friction_level_v3','business_risk_score','business_risk_level','priority_level','recommended_queue','recommended_business_action','has_legal_risk','has_churn_risk','has_billing_dispute','has_technical_outage','potential_agent_mismatch'] if k in df.columns})
with tabs[4]:
    show_df(load_csv('high_value_at_risk_summary.csv'),'High-value at-risk summary'); show_df(load_csv('high_value_priority_distribution.csv'),'High-value priority distribution'); show_df(load_csv('high_value_revenue_by_priority.csv'),'High-value revenue by priority'); show_df(load_csv('high_value_at_risk.csv'),'Top high-value at-risk rows')
with tabs[5]:
    p=OUT/'demo_cases/final_demo_cases.json'
    st.info('Bu kayıtlar temsilî demo örnekleridir; sistemin tamamı tekrar kullanılabilir iş kullanım senaryolarından oluşur.')
    if not p.exists(): st.warning('Demo examples bulunamadı. Final pipeline çalıştırın.')
    else:
        for c in json.loads(p.read_text(encoding='utf-8')):
            with st.expander(c.get('case_name','Demo')): st.json(c)
with tabs[6]:
    for f in ['retention_model_metrics.csv','rag_retrieval_metrics.csv','rag_generation_metrics.csv','priority_distribution.csv','risk_component_summary.csv','friction_v3_distribution.csv','signal_distribution.csv','repeat_complaint_summary.csv']:
        show_df(load_csv(f), f)
    st.markdown('**Bilinen sınırlılıklar:** Resmi churn etiketi yoktur; LLM opsiyoneldir; RAG yanıtı kaynak yoksa fallback kullanmalıdır; bill shock nedensellik olarak yorumlanmaz.')
