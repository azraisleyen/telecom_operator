import pandas as pd
from .preprocessing import ym_to_month_index
def add_bill_shock(spending, complaints, threshold=.30):
    s=spending.copy().sort_values(['customer_id','billing_year_month']); s['month_index']=s.billing_year_month.map(ym_to_month_index)
    s['previous_bill_amount']=s.groupby('customer_id').bill_amount.shift(1)
    s['rolling_3m_avg_bill']=s.groupby('customer_id').bill_amount.transform(lambda x:x.shift(1).rolling(3,min_periods=1).mean())
    s['bill_change_abs']=s.bill_amount-s.previous_bill_amount; s['bill_change_pct']=s.bill_change_abs/s.previous_bill_amount.replace(0,pd.NA)
    s['bill_shock_flag']=((s.bill_change_pct>=threshold)&s.previous_bill_amount.notna()).astype(int)
    c=complaints[['customer_id','billing_year_month','problem_category']].copy(); c['is_billing']=(c.problem_category=='billing').astype(int); bill=c.groupby(['customer_id','billing_year_month']).is_billing.max().reset_index(name='same_month_billing_complaint')
    s=s.merge(bill,how='left').fillna({'same_month_billing_complaint':0}); s['next_month']=s.month_index+1
    nb=bill.assign(month_index=bill.billing_year_month.map(ym_to_month_index)-1)[['customer_id','month_index','same_month_billing_complaint']].rename(columns={'same_month_billing_complaint':'next_month_billing_complaint'})
    s=s.merge(nb,on=['customer_id','month_index'],how='left').fillna({'next_month_billing_complaint':0})
    months=set(zip(s.customer_id,s.month_index)); s['no_billable_activity_next_2m']=[int((cid,mi+1) not in months and (cid,mi+2) not in months) for cid,mi in zip(s.customer_id,s.month_index)]
    return s
