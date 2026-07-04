from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from capstone.src.data_loader import load_or_build_processed_data
from capstone.src.lifecycle import build_lifecycle_v2
from capstone.src.feature_engineering import build_customer_profile, monthly_kpi, category_counts, subcategory_counts
from capstone.src.bill_shock import add_bill_shock
from capstone.src.complaint_signals import add_complaint_signals
from capstone.src.friction_scoring import add_friction_v3, export_friction_tables
from capstone.src.risk_scoring import add_risk_priority, export_risk_tables
from capstone.src.routing import add_routing
from capstone.src.retrieval import build_reference_index, evaluate_retrieval
from capstone.src.demo_cases import build_demo_cases
from capstone.src.evaluation import export_final_evaluation
from capstone.src.preprocessing import save_table, ym_to_month_index

def str2bool(x): return str(x).lower() in {'1','true','yes','y'}
def add_repeat_flags(c):
    c=c.copy(); c['month_index']=c.billing_year_month.map(ym_to_month_index); c=c.sort_values(['customer_id','month_index'])
    nxt=c.groupby('customer_id').month_index.shift(-1); c['repeat_complaint']=((nxt-c.month_index).between(0,2)).fillna(False).astype(int); return c

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='capstone/data'); ap.add_argument('--output-dir',default='capstone/outputs'); ap.add_argument('--rebuild',default='false'); args=ap.parse_args()
    out=Path(args.output_dir); [ (out/p).mkdir(parents=True,exist_ok=True) for p in ['tables','demo_cases','final','figures']]
    data=load_or_build_processed_data(args.data_dir,rebuild=str2bool(args.rebuild))
    customer,spending,complaints,reference=data['customer'],data['spending'],data['complaints'],data['reference']
    lifecycle=build_lifecycle_v2(spending); save_table(lifecycle,Path(args.data_dir)/'processed/lifecycle_v2.parquet')
    profile,threshold=build_customer_profile(customer,lifecycle); save_table(profile,Path(args.data_dir)/'processed/customer_profile.parquet')
    save_table(monthly_kpi(spending,complaints),out/'tables/monthly_kpi_summary.csv'); save_table(category_counts(complaints),out/'tables/category_counts.csv'); save_table(subcategory_counts(complaints),out/'tables/subcategory_counts.csv')
    bill=add_bill_shock(spending,complaints); save_table(bill,Path(args.data_dir)/'processed/spending_bill_shock.parquet')
    bill_cols=['customer_id','billing_year_month','previous_bill_amount','rolling_3m_avg_bill','bill_change_abs','bill_change_pct','bill_shock_flag','same_month_billing_complaint','next_month_billing_complaint','no_billable_activity_next_2m']
    enriched=complaints.merge(profile[['customer_id','lifecycle_status_v2','total_revenue','avg_bill','avg_data_usage','high_value_customer','customer_segment']],on='customer_id',how='left').merge(bill[bill_cols],on=['customer_id','billing_year_month'],how='left')
    enriched[['bill_shock_flag','same_month_billing_complaint','next_month_billing_complaint','no_billable_activity_next_2m']]=enriched[['bill_shock_flag','same_month_billing_complaint','next_month_billing_complaint','no_billable_activity_next_2m']].fillna(0)
    enriched=add_repeat_flags(add_complaint_signals(enriched)); enriched=add_friction_v3(enriched); enriched=add_risk_priority(enriched); enriched=add_routing(enriched)
    save_table(enriched,Path(args.data_dir)/'processed/complaints_final_features.parquet')
    export_friction_tables(enriched,args.output_dir); export_risk_tables(enriched,args.output_dir)
    index=build_reference_index(reference); rag=evaluate_retrieval(index,enriched,args.output_dir)
    cases=build_demo_cases(enriched,index,args.output_dir)
    export_final_evaluation(customer,spending,enriched,profile,bill,rag,args.output_dir)
    print(f'Pipeline complete. High-value threshold={threshold:.2f}. Demo cases={len(cases)}. Outputs: {args.output_dir}')
if __name__=='__main__': main()
