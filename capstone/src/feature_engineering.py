import pandas as pd
from .preprocessing import safe_divide
def build_customer_profile(customer,lifecycle):
    p=customer.merge(lifecycle,on='customer_id',how='left'); p['age']=2026-p.birth_year
    p['age_band']=pd.cut(p.age,[0,25,35,45,55,65,200],labels=['<=25','26-35','36-45','46-55','56-65','65+'])
    thr=p.total_revenue.quantile(.80); p['high_value_customer']=(p.total_revenue>=thr).astype(int); p['customer_segment']=p.high_value_customer.map({1:'High value (top 20% revenue)',0:'Standard value'}); return p,thr
def monthly_kpi(spending,complaints):
    s=spending.groupby('billing_year_month').agg(active_customers=('customer_id','nunique'),total_revenue=('bill_amount','sum'),avg_data_usage=('data_usage','mean')).reset_index(); s['arpu']=s.total_revenue/s.active_customers
    c=complaints.groupby('billing_year_month').size().reset_index(name='complaint_count'); out=s.merge(c,how='left').fillna({'complaint_count':0}); out['complaints_per_1000_active_customers']=out.complaint_count/out.active_customers*1000; return out
def category_counts(c): return c.problem_category.value_counts().rename_axis('problem_category').reset_index(name='complaint_count')
def subcategory_counts(c): return c.problem_subcategory.value_counts().rename_axis('problem_subcategory').reset_index(name='complaint_count')
