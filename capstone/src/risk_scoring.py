import pandas as pd
from .preprocessing import save_table
def add_risk_priority(df):
    o=df.copy(); o['repeat_complaint']=o.get('repeat_complaint',0)
    o['business_risk_score']=(40*o.has_legal_risk+30*o.has_churn_risk+20*o.high_frustration_flag+15*(o.friction_level_v3=='high')+15*o.repeat_complaint+15*o.has_billing_dispute+10*o.has_technical_outage+10*o.high_value_customer+10*o.bill_shock_flag).astype(int)
    o['business_risk_level']=pd.cut(o.business_risk_score,[-1,19,49,999],labels=['low','medium','high']).astype(str)
    o['priority_level']='P3 Low'; o.loc[(o.business_risk_score>=25),'priority_level']='P2 Medium'; o.loc[(o.business_risk_score>=50)| (o.has_churn_risk==1),'priority_level']='P1 High'; o.loc[(o.business_risk_score>=80)|(o.has_legal_risk==1),'priority_level']='P0 Critical'
    return o
def export_risk_tables(df,outdir):
    save_table(df.business_risk_level.value_counts().rename_axis('business_risk_level').reset_index(name='complaint_count'),f'{outdir}/tables/business_risk_level_summary.csv')
    save_table(df.priority_level.value_counts().rename_axis('priority_level').reset_index(name='complaint_count'),f'{outdir}/tables/priority_distribution.csv')
    hv=df[(df.high_value_customer==1)&(df.business_risk_level=='high')]; save_table(hv.head(10000),f'{outdir}/tables/high_value_at_risk.csv')
    save_table(pd.DataFrame([{'high_value_at_risk_customers':hv.customer_id.nunique(),'historical_revenue':hv.total_revenue.sum()}]),f'{outdir}/tables/high_value_at_risk_summary.csv')
